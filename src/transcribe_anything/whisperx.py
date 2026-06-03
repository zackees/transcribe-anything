"""
Runs the WhisperX backend.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import static_ffmpeg  # type: ignore
import static_ffmpeg.run as static_ffmpeg_run  # type: ignore
import webvtt  # type: ignore

from transcribe_anything.generate_speaker_json import Chunk
from transcribe_anything.generate_speaker_json import reduce as reduce_speaker_chunks
from transcribe_anything.util import get_static_ffmpeg_runtime_dir, has_nvidia_smi
from transcribe_anything.whisperx_reqs import get_environment

HERE = Path(__file__).parent

_SUPPORTED_ACTION_FLAGS = {
    "--diarize",
    "--no_align",
    "--return_char_alignments",
    "--speaker_embeddings",
    "--suppress_numerals",
}

_SUPPORTED_BOOL_VALUE_FLAGS = {
    "--condition_on_previous_text",
    "--fp16",
    "--highlight_words",
    "--model_cache_only",
    "--print_progress",
    "--verbose",
}

_SUPPORTED_VALUE_FLAGS = {
    "--align_model",
    "--batch_size",
    "--beam_size",
    "--best_of",
    "--chunk_size",
    "--compression_ratio_threshold",
    "--compute_type",
    "--device",
    "--device_index",
    "--diarize_model",
    "--hotwords",
    "--initial_prompt",
    "--interpolate_method",
    "--length_penalty",
    "--log-level",
    "--logprob_threshold",
    "--max_line_count",
    "--max_line_width",
    "--max_speakers",
    "--min_speakers",
    "--model_dir",
    "--no_speech_threshold",
    "--patience",
    "--segment_resolution",
    "--suppress_tokens",
    "--temperature",
    "--temperature_increment_on_fallback",
    "--threads",
    "--vad_method",
    "--vad_offset",
    "--vad_onset",
}

_CONTROLLED_VALUE_FLAGS = {
    "--hf_token",
    "--language",
    "--model",
    "--output_dir",
    "--output_format",
    "--task",
    "-f",
    "-o",
}

_SKIPPED_ACTION_FLAGS = {
    "--python-version",
    "--version",
    "-P",
    "-V",
}

_KNOWN_FLAGS = _SUPPORTED_ACTION_FLAGS | _SUPPORTED_BOOL_VALUE_FLAGS | _SUPPORTED_VALUE_FLAGS | _CONTROLLED_VALUE_FLAGS | _SKIPPED_ACTION_FLAGS

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}

_FLAG_ALIASES = {
    "--log_level": "--log-level",
}


def _canonical_flag(flag: str) -> str:
    """Return the WhisperX CLI spelling for a supported flag."""
    if flag in _FLAG_ALIASES:
        return _FLAG_ALIASES[flag]
    if flag in _KNOWN_FLAGS:
        return flag
    if flag.startswith("--"):
        candidate = "--" + flag[2:].replace("-", "_")
        if candidate in _KNOWN_FLAGS:
            return candidate
    return flag


def _split_flag_value(arg: str) -> tuple[str, str | None]:
    """Split --flag=value style args without disturbing plain flags."""
    if arg.startswith("--") and "=" in arg:
        flag, value = arg.split("=", 1)
        return flag, value
    return arg, None


def _looks_like_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _looks_like_option(value: str) -> bool:
    if value.startswith("--"):
        return True
    if value.startswith("-") and len(value) > 1 and value[1].isdigit():
        return False
    return value.startswith("-") and not _looks_like_number(value)


def _is_bool_value(value: str) -> bool:
    return value.lower() in _TRUE_VALUES or value.lower() in _FALSE_VALUES


def _as_cli_bool(value: str) -> str:
    return "True" if value.lower() in _TRUE_VALUES else "False"


def _warn_skipped_arg(flag: str) -> None:
    warnings.warn(f"Skipping unsupported WhisperX argument: {flag}")


def _read_following_value(args: list[str], index: int, explicit_value: str | None) -> tuple[str | None, int]:
    """Return an option value and the next index to inspect."""
    if explicit_value is not None:
        return explicit_value, index + 1
    next_index = index + 1
    if next_index >= len(args):
        return None, index + 1
    value = args[next_index]
    if _looks_like_option(value):
        return None, index + 1
    return value, index + 2


def _parse_other_args(other_args: list[str] | None) -> tuple[list[str], str | None]:
    """Normalize supported WhisperX args and extract any passed HF token."""
    if not other_args:
        return [], None

    parsed: list[str] = []
    hf_token: str | None = None
    index = 0
    while index < len(other_args):
        raw_arg = str(other_args[index]).strip()
        if not raw_arg:
            index += 1
            continue
        if not raw_arg.startswith("-"):
            _warn_skipped_arg(raw_arg)
            index += 1
            continue

        raw_flag, explicit_value = _split_flag_value(raw_arg)
        flag = _canonical_flag(raw_flag)

        if flag in _SKIPPED_ACTION_FLAGS:
            warnings.warn(f"Skipping WhisperX informational argument: {flag}")
            index += 1
            continue

        if flag in _CONTROLLED_VALUE_FLAGS:
            value, index = _read_following_value(other_args, index, explicit_value)
            if flag == "--hf_token":
                hf_token = value
            continue

        if flag in _SUPPORTED_ACTION_FLAGS:
            value, next_index = _read_following_value(other_args, index, explicit_value)
            if value is None:
                parsed.append(flag)
                index = next_index
                continue
            if _is_bool_value(value):
                if value.lower() in _TRUE_VALUES:
                    parsed.append(flag)
                index = next_index
                continue
            parsed.append(flag)
            index += 1
            continue

        if flag in _SUPPORTED_BOOL_VALUE_FLAGS:
            value, index = _read_following_value(other_args, index, explicit_value)
            if value is None:
                value = "True"
            elif _is_bool_value(value):
                value = _as_cli_bool(value)
            parsed.extend([flag, value])
            continue

        if flag in _SUPPORTED_VALUE_FLAGS:
            value, index = _read_following_value(other_args, index, explicit_value)
            if value is None:
                warnings.warn(f"Skipping WhisperX argument without value: {flag}")
                continue
            parsed.extend([flag, value])
            continue

        _warn_skipped_arg(raw_flag)
        _, index = _read_following_value(other_args, index, explicit_value)

    return parsed, hf_token


def _has_cli_option(args: list[str], flag: str) -> bool:
    """Return True when a normalized CLI option is already present."""
    return any(arg == flag for arg in args)


def _default_device() -> str:
    """Choose the default WhisperX device for this backend."""
    if sys.platform == "darwin":
        return "cpu"
    return "cuda" if has_nvidia_smi() else "cpu"


def _format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _format_vtt_time(seconds: float) -> str:
    return _format_srt_time(seconds).replace(",", ".")


def _segment_text(segment: dict[str, Any]) -> str:
    text = str(segment.get("text", "")).strip()
    speaker = segment.get("speaker")
    if speaker:
        return f"[{speaker}]: {text}"
    return text


def _segment_bounds(segment: dict[str, Any]) -> tuple[float | None, float | None]:
    start = segment.get("start")
    end = segment.get("end")
    if start is not None and end is not None:
        return float(start), float(end)

    words = segment.get("words")
    if not isinstance(words, list):
        return None, None
    starts = [float(word["start"]) for word in words if isinstance(word, dict) and "start" in word]
    ends = [float(word["end"]) for word in words if isinstance(word, dict) and "end" in word]
    if starts and ends:
        return min(starts), max(ends)
    return None, None


def _json_to_srt(json_data: dict[str, Any]) -> str:
    parts: list[str] = []
    index = 1
    for segment in json_data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        start, end = _segment_bounds(segment)
        if start is None or end is None:
            continue
        text = _segment_text(segment)
        parts.append(f"{index}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}\n")
        index += 1
    return "\n".join(parts)


def _json_to_vtt(json_data: dict[str, Any]) -> str:
    parts = ["WEBVTT\n"]
    for segment in json_data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        start, end = _segment_bounds(segment)
        if start is None or end is None:
            continue
        text = _segment_text(segment)
        parts.append(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}\n{text}\n")
    return "\n".join(parts)


def _json_to_txt(json_data: dict[str, Any]) -> str:
    if isinstance(json_data.get("text"), str):
        return str(json_data["text"]).strip()
    lines: list[str] = []
    for segment in json_data.get("segments", []):
        if isinstance(segment, dict):
            text = _segment_text(segment)
            if text:
                lines.append(text)
    return "\n".join(lines)


def _trim_text_fields(node: Any) -> None:
    if isinstance(node, dict):
        if "text" in node and isinstance(node["text"], str):
            node["text"] = node["text"].strip()
        for value in node.values():
            _trim_text_fields(value)
    elif isinstance(node, list):
        for item in node:
            _trim_text_fields(item)


def _find_output_file(output_dir: Path, suffix: str) -> Path | None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.suffix == suffix)
    if not files:
        return None
    return files[0]


def _write_vtt_from_srt(srt_file: Path, vtt_file: Path, json_data: dict[str, Any]) -> None:
    if srt_file.read_text(encoding="utf-8").strip():
        try:
            webvtt.from_srt(str(srt_file)).save(str(vtt_file))
            return
        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(f"Failed to convert WhisperX SRT to VTT: {exc}")
    vtt_file.write_text(_json_to_vtt(json_data), encoding="utf-8")


def _word_speaker_chunks(segment: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    words = segment.get("words", [])
    if not isinstance(words, list):
        return chunks
    for word in words:
        if not isinstance(word, dict):
            continue
        speaker = word.get("speaker")
        if not speaker or "start" not in word or "end" not in word:
            continue
        text = str(word.get("word", "")).strip()
        chunks.append(Chunk(str(speaker), float(word["start"]), float(word["end"]), text))
    return chunks


def _speaker_json_from_whisperx(json_data: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[Chunk] = []
    for segment in json_data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        speaker = segment.get("speaker")
        start, end = _segment_bounds(segment)
        if speaker and start is not None and end is not None:
            chunks.append(Chunk(str(speaker), start, end, str(segment.get("text", "")).strip()))
            continue
        chunks.extend(_word_speaker_chunks(segment))
    if not chunks:
        return []
    reduced = reduce_speaker_chunks(chunks)
    return [chunk.to_json() for chunk in reduced]


def _normalize_outputs(whisperx_output_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = _find_output_file(whisperx_output_dir, ".json")
    if json_file is None:
        raise FileNotFoundError(f"WhisperX did not produce a JSON output in {whisperx_output_dir}")
    json_data = json.loads(json_file.read_text(encoding="utf-8"))
    _trim_text_fields(json_data)

    out_json = output_dir / "out.json"
    out_json.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    out_srt = output_dir / "out.srt"
    srt_file = _find_output_file(whisperx_output_dir, ".srt")
    if srt_file is not None:
        shutil.copyfile(srt_file, out_srt)
    else:
        out_srt.write_text(_json_to_srt(json_data), encoding="utf-8")

    out_vtt = output_dir / "out.vtt"
    vtt_file = _find_output_file(whisperx_output_dir, ".vtt")
    if vtt_file is not None:
        shutil.copyfile(vtt_file, out_vtt)
    else:
        _write_vtt_from_srt(out_srt, out_vtt, json_data)

    out_txt = output_dir / "out.txt"
    txt_file = _find_output_file(whisperx_output_dir, ".txt")
    if txt_file is not None:
        shutil.copyfile(txt_file, out_txt)
    else:
        out_txt.write_text(_json_to_txt(json_data), encoding="utf-8")

    speaker_file = output_dir / "speaker.json"
    speaker_json = _speaker_json_from_whisperx(json_data)
    if speaker_json:
        speaker_file.write_text(json.dumps(speaker_json, ensure_ascii=False, indent=2), encoding="utf-8")
    elif speaker_file.exists():
        speaker_file.unlink()


def run_whisperx(  # pylint: disable=too-many-arguments
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    hugging_face_token: str | None = None,
    other_args: list[str] | None = None,
) -> None:
    """Run WhisperX through its isolated environment."""
    ffmpeg_cache = get_static_ffmpeg_runtime_dir()
    static_ffmpeg_run.LOCK_FILE = str(ffmpeg_cache / "lock.file")
    static_ffmpeg.add_paths(download_dir=str(ffmpeg_cache / static_ffmpeg_run.get_platform_key()))
    iso_env = get_environment()
    passthrough_args, arg_hf_token = _parse_other_args(other_args)
    hf_token = hugging_face_token or arg_hf_token

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        whisperx_output_dir = Path(temp_dir) / "whisperx"
        whisperx_output_dir.mkdir(parents=True, exist_ok=True)

        cmd_list = [
            "whisperx",
            str(input_wav),
            "--output_dir",
            str(whisperx_output_dir),
            "--output_format",
            "all",
            "--task",
            task or "transcribe",
        ]
        if model:
            cmd_list.extend(["--model", model])
        if language:
            cmd_list.extend(["--language", language])
        if not _has_cli_option(passthrough_args, "--device"):
            cmd_list.extend(["--device", _default_device()])
        if hf_token:
            cmd_list.extend(["--hf_token", hf_token])
        cmd_list.extend(passthrough_args)

        cmd_list = [str(item).strip() for item in cmd_list if str(item).strip()]
        cmd = subprocess.list2cmdline(cmd_list)
        sys.stderr.write(f"Running:\n  {cmd}\n")
        proc = iso_env.open_proc(  # pylint: disable=consider-using-with
            cmd_list,
            shell=False,
            universal_newlines=True,
            encoding="utf-8",
            env=env,
        )
        while True:
            rtn = proc.poll()
            if rtn is None:
                time.sleep(0.25)
                continue
            if rtn != 0:
                raise OSError(f"Failed to execute {cmd}\n")
            break
        proc.wait()

        _normalize_outputs(whisperx_output_dir, output_dir)
