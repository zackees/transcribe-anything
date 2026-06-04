# pylint: skip-file
# flake8: noqa

"""
Runs whisper api.
"""

import json  # type: ignore
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import wave
from pathlib import Path
from typing import Any, Optional

import static_ffmpeg  # type: ignore
import static_ffmpeg.run as static_ffmpeg_run  # type: ignore
import webvtt  # type: ignore

from transcribe_anything.cuda_available import CudaInfo
from transcribe_anything.generate_speaker_json import generate_speaker_json
from transcribe_anything.insanley_fast_whisper_reqs import get_environment
from transcribe_anything.util import get_static_ffmpeg_runtime_dir, print_cuda_diagnostics

HERE = Path(__file__).parent
CUDA_INFO: Optional[CudaInfo] = None
FLASH_ATTENTION_VERIFIED = False

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}

FLASH_ATTENTION_PROBE_CODE = r"""
import json
import platform
import sys

result = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
}

try:
    import torch

    result["torch_version"] = torch.__version__
    result["torch_cuda"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")

    capability = torch.cuda.get_device_capability(0)
    result["gpu_name"] = torch.cuda.get_device_name(0)
    result["gpu_capability"] = f"sm_{capability[0]}{capability[1]}"
    if capability[0] < 8:
        raise RuntimeError("FlashAttention-2 CUDA requires NVIDIA Ampere or newer (SM80+)")

    import flash_attn
    import flash_attn_2_cuda  # noqa: F401

    result["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    result["flash_attn_2_cuda"] = True

    from transformers import AutoModelForSpeechSeq2Seq, WhisperConfig
    from transformers.utils import is_flash_attn_2_available

    result["transformers_flash_attn_2_available"] = bool(is_flash_attn_2_available())
    if not result["transformers_flash_attn_2_available"]:
        raise RuntimeError("Transformers reports FlashAttention2 as unavailable")

    config = WhisperConfig(
        vocab_size=64,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        num_mel_bins=80,
        max_source_positions=16,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = AutoModelForSpeechSeq2Seq.from_config(
        config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    ).to("cuda")
    result["whisper_attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    if result["whisper_attn_implementation"] != "flash_attention_2":
        raise RuntimeError(
            "Whisper model did not select flash_attention_2; "
            f"got {result['whisper_attn_implementation']!r}"
        )
    print(json.dumps(result, sort_keys=True))
except Exception as exc:
    result["error"] = repr(exc)
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    raise
"""


def _k2_stub_root() -> Path:
    """Directory containing the ``k2`` stub package shipped with this project.

    Adding this directory to ``PYTHONPATH`` lets the venv's Python satisfy
    ``import k2`` (needed by ``speechbrain.integrations.k2_fsa`` on
    speechbrain>=1.0) on platforms where the real ``k2`` is not installable
    — notably Windows, where the upstream package ships no wheels. See
    issue #69.
    """
    return HERE / "_k2_stub"


def _prepare_subprocess_env(base_env: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``base_env`` augmented for the insane backend.

    Adds the k2 stub directory to ``PYTHONPATH`` at the END so a real
    ``k2`` installed in the venv (e.g. on Linux + CUDA where wheels
    exist) still wins normal import resolution.
    """
    env = dict(base_env)
    stub = str(_k2_stub_root())
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if stub not in parts:
        parts.append(stub)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _split_flag_value(arg: str) -> tuple[str, str | None]:
    """Split --flag=value args while preserving plain flags."""
    if arg.startswith("--") and "=" in arg:
        flag, value = arg.split("=", 1)
        return flag, value
    return arg, None


def _bool_arg_value(value: str | None) -> bool:
    """Normalize a CLI boolean value."""
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Expected a boolean value for --flash, got {value!r}")


def _prepare_insane_args(other_args: list[str] | None, *, force_flash: bool) -> list[str]:
    """Return backend args, forcing --flash True for the insane-flash backend."""
    args = [str(arg) for arg in (other_args or [])]
    if not force_flash:
        return args

    prepared: list[str] = []
    index = 0
    saw_flash = False
    while index < len(args):
        raw_arg = args[index]
        flag, explicit_value = _split_flag_value(raw_arg)
        if flag != "--flash":
            prepared.append(raw_arg)
            index += 1
            continue

        saw_flash = True
        if explicit_value is not None:
            value = explicit_value
            index += 1
        elif index + 1 < len(args) and not args[index + 1].startswith("-"):
            value = args[index + 1]
            index += 2
        else:
            value = None
            index += 1
        if not _bool_arg_value(value):
            raise ValueError("--device insane-flash requires FlashAttention; remove '--flash False' or use --device insane.")

    if saw_flash:
        sys.stderr.write("Normalizing --flash True for --device insane-flash.\n")
    prepared.extend(["--flash", "True"])
    return prepared


def verify_flash_attention_available(iso_env: Any) -> None:
    """Verify that the flash env can import and select FlashAttention2."""
    global FLASH_ATTENTION_VERIFIED  # pylint: disable=global-statement
    if FLASH_ATTENTION_VERIFIED:
        return

    result = iso_env.run(
        ["python", "-c", FLASH_ATTENTION_PROBE_CODE],
        shell=False,
        universal_newlines=True,
        encoding="utf-8",
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        details = "\n".join(part for part in [stdout, stderr] if part)
        raise RuntimeError(f"insane-flash FlashAttention capability probe failed:\n{details}")
    if stdout:
        sys.stderr.write(f"insane-flash FlashAttention probe passed: {stdout}\n")
    FLASH_ATTENTION_VERIFIED = True


def get_cuda_info() -> CudaInfo:
    """Get the computing device."""
    global CUDA_INFO  # pylint: disable=global-statement
    if CUDA_INFO is None:
        env = get_environment()
        py_file = HERE / "cuda_available.py"
        # tempfile
        with tempfile.TemporaryDirectory() as dir_name:
            temp = Path(dir_name) / "stdout.txt"
            abs_name = temp.absolute()
            try:
                env.run(
                    [str(py_file), "-o", str(abs_name)],
                    shell=False,
                    check=True,
                    universal_newlines=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 1:  # 1 is the expected return code
                    raise
                pass
                # print(f"Failed to run python {py_file} -o {abs_name}: {exc}")
                # print(f"stdout: {exc.stdout}")
                # print(f"stderr: {exc.stderr}")
            # stdout = cp.stdout
            stdout = temp.read_text(encoding="utf-8")
            try:
                CUDA_INFO = CudaInfo.from_json_str(stdout)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to decode json: {exc}") from exc
            assert CUDA_INFO is not None, f"Expected CUDA_INFO to be set, but the stdout was {stdout}"
    return CUDA_INFO


def get_device_id() -> str:
    """Get the device id."""
    # on mac, we just return "mps"
    if sys.platform == "darwin":
        return "mps"
    cuda_info = get_cuda_info()
    if not cuda_info.cuda_available:
        print_cuda_diagnostics(expected_cuda="12.8")
        raise ValueError("CUDA is not available. Run 'transcribe-anything --clear-nvidia-cache' if hardware changed. " "See diagnostic output above for details.")
    device_id = cuda_info.cuda_devices[0].device_id
    return f"{device_id}"


def get_batch_size() -> int | None:
    """Returns the batch size."""
    if sys.platform == "darwin":
        return 1
    return None


def convert_time_to_srt_format(timestamp: float) -> str:
    """Converts timestamp in seconds to SRT time format (hours:minutes:seconds,milliseconds)."""
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:02},{milliseconds:03}"


def convert_to_webvtt(srt_file: Path, out_webvtt_file: Path) -> None:
    """Convert to webvtt format."""
    STYLE_ELEMENT = """STYLE
    ::::cue {
    line: 80%;
    }
    """
    assert srt_file.suffix == ".srt"
    assert out_webvtt_file.suffix == ".vtt"
    webvtt.from_srt(str(srt_file)).save(str(out_webvtt_file))
    content = out_webvtt_file.read_text(encoding="utf-8")
    content = content.replace("WEBVTT\n\n", f"WEBVTT\n\n{STYLE_ELEMENT}")
    out_webvtt_file.write_text(content, encoding="utf-8")


def convert_json_to_srt(json_data: dict[str, Any], duration: float) -> str:
    """Converts JSON data from speech-to-text tool to SRT format."""
    srt_content = ""
    num_chunks = len(json_data["chunks"])
    for index, chunk in enumerate(json_data["chunks"], start=1):
        # start_time, end_time = chunk["timestamp"]
        time_pair = chunk["timestamp"]
        start_time = time_pair[0]
        end_time = time_pair[1]
        try:
            if start_time is None and end_time is None:
                print(f"Skipping chunk {index} because both start and end time are None.")
                stack_trace = traceback.format_stack()
                print("Stack trace: ", stack_trace)
                continue
            if end_time is None:
                # assert index == num_chunks
                if index != num_chunks:
                    print(f"Setting end time to duration because it's None for chunk {index}.")
                end_time = duration  # Sometimes happens at the end
            try:
                start_time_str = convert_time_to_srt_format(start_time)
            except Exception as exc:
                print(f"Failed to convert start time {start_time} to srt format: {exc}")
                stack_trace = traceback.format_stack()
                print("Stack trace: ", stack_trace)
                continue
            try:
                end_time_str = convert_time_to_srt_format(end_time)
            except Exception as exc:
                print(f"Failed to convert end time {end_time} to srt format: {exc}")
                stack_trace = traceback.format_stack()
                print("Stack trace: ", stack_trace)
                continue
        except Exception as exc:
            print(f"Failed to convert times for chunk {index}: {exc}")
            stack_trace = traceback.format_stack()
            print("Stack trace: ", stack_trace)
            continue
        try:
            text = str(chunk["text"]).strip()
            srt_content += f"{index}\n{start_time_str} --> {end_time_str}\n{text}\n\n"
        except Exception as exc:
            print(f"Failed to add chunk {index} to srt content: {exc}")
            stack_trace = traceback.format_stack()
            print("Stack trace: ", stack_trace)
            continue
    return srt_content


def convert_json_to_text(json_data: dict[str, Any]) -> str:
    """Converts JSON data from speech-to-text tool to text."""
    return json_data["text"]


def get_wave_duration(wave_file: Path) -> float:
    """Returns the duration of a wave file."""
    with wave.open(str(wave_file), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / float(rate)
        return duration


def trim_text_chunks(json_data: dict[str, Any]) -> None:
    """'text' chunks seem to have an extra space at the beginning, remove it."""

    # visit all the nodes in the json data, when we see one that has a 'text' key,
    # then apply a trim.
    def visit(node: dict[str, Any]) -> None:
        if isinstance(node, dict):
            if "text" in node:
                node["text"] = node["text"].strip()
            for key, value in node.items():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(json_data)


def run_insanely_fast_whisper(
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    hugging_face_token: str | None = None,
    other_args: list[str] | None = None,
    flash: bool = False,
    align: bool = False,
    align_model: str | None = None,
) -> None:
    """Runs insanely fast whisper.

    When ``align`` is true, runs WhisperX's wav2vec2 forced-alignment pass
    on the transcript to replace HF Whisper's segment-level timestamps
    with phoneme-precise word-level timestamps. Reuses the WhisperX
    iso-env (no new deps in the insane env). Best-effort: unsupported
    language or env-build failure logs a warning and leaves the original
    output untouched.
    """
    # ffmpeg paths have to be installed or else the backend tool will fail.
    ffmpeg_cache = get_static_ffmpeg_runtime_dir()
    static_ffmpeg_run.LOCK_FILE = str(ffmpeg_cache / "lock.file")
    static_ffmpeg.add_paths(download_dir=str(ffmpeg_cache / static_ffmpeg_run.get_platform_key()))
    iso_env = get_environment(flash=flash)
    backend_args = _prepare_insane_args(other_args, force_flash=flash)
    if flash:
        verify_flash_attention_available(iso_env)
    env = _prepare_subprocess_env(dict(os.environ))
    if sys.platform == "darwin":
        # Attempts fixed recommended for the mps machines. This seems
        # to be necessary since a recent update.
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
        # env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device_id = get_device_id()
    cmd_list = []
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "out.json"
    if "/" not in model:
        # Assume it's not a namespace model, so add the namespace.
        model = f"openai/whisper-{model}"
    wave_duration = get_wave_duration(input_wav)
    # if sys.platform == "win32":
    # Set the text mode to UTF-8 on Windows.
    # cmd_list.extend(["cmd.exe", "/c"])
    # cmd_list.extend(["chcp", "65001", "&&"])
    cmd_list += [
        "insanely-fast-whisper",
        "--file-name",
        str(input_wav),
        "--device-id",
        f"{device_id}",
    ]
    if model:
        cmd_list += [
            "--model-name",
            model,
        ]
    cmd_list += [
        "--task",
        task,
        "--transcript-path",
        str(outfile),
    ]
    if hugging_face_token:
        cmd_list += ["--hf-token", hugging_face_token]
        # remove --hf-token from other_args if it's there.
        if "--hf-token" in backend_args:
            idx = backend_args.index("--hf-token")
            idx2 = idx + 1
            backend_args.pop(idx2)
            backend_args.pop(idx)
    if language:
        cmd_list += ["--language", language]

    batch_size: int | None = None
    if backend_args:
        # Check if the other_args contains --batch-size and remove it.
        if "--batch-size" in backend_args:
            idx = backend_args.index("--batch-size")
            idx2 = idx + 1
            batch_size = int(backend_args[idx2])
            backend_args.pop(idx2)
            backend_args.pop(idx)
    batch_size = get_batch_size() or batch_size
    if batch_size is not None:
        cmd_list += ["--batch-size", f"{batch_size}"]
    if backend_args:
        cmd_list.extend(backend_args)
    # Remove the empty strings.
    cmd_list = [x.strip() for x in cmd_list if x.strip()]
    cmd = subprocess.list2cmdline(cmd_list)
    # Mask --hf-token's value before any logging/error so the token doesn't
    # leak into stdout, error responses, or downstream observability tools.
    cmd_safe = re.sub(r"(--hf[-_]token)\s+\S+", r"\1 <REDACTED>", cmd)
    sys.stderr.write(f"Running:\n  {cmd_safe}\n")
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
            time.sleep(0.1)
            continue
        if rtn != 0:
            msg = f"Failed to execute {cmd_safe}\n "
            raise OSError(msg)
        break
    proc.wait()
    assert outfile.exists(), f"Expected {outfile} to exist."
    json_text = outfile.read_text(encoding="utf-8")
    json_data = json.loads(json_text)
    trim_text_chunks(json_data)

    if align:
        from transcribe_anything.insane_align import apply_forced_alignment

        json_data = apply_forced_alignment(
            json_data,
            input_wav=input_wav,
            language=language,
            align_model=align_model,
        )
        trim_text_chunks(json_data)

    json_data_str = json.dumps(json_data, indent=2)

    if hugging_face_token:
        print("### HUGGING FACE TOKEN IS ACTIVE - GENERATING SPEAKER JSON ###")
        # Speaker diarization is active so generate the file
        try:
            speaker_json = generate_speaker_json(json_data)
            speaker_json_str = json.dumps(speaker_json, indent=2)
            speaker_json_file = output_dir / "speaker.json"
            speaker_json_file.write_text(speaker_json_str, encoding="utf-8")
        except Exception as exc:
            warnings.warn(f"Failed to generate speaker json beause of exception: {exc}")
    else:
        print("### HUGGING FACE TOKEN IS NOT ACTIVE - NO SPEAKER JSON GENERATED ###")

    # now write the pretty formatted json data back to the text file.
    outfile.write_text(json_data_str, encoding="utf-8")
    try:
        srt_content = convert_json_to_srt(json_data, wave_duration)
        srt_file = output_dir / "out.srt"
    except Exception as exc:
        print(f"Failed to convert to srt: {exc}")
        print("Json data: ", json_data_str)
        error_file = Path("transcribe-anything-error.json")
        error_file.write_text(json_text, encoding="utf-8")
        raise
    try:
        txt_content = convert_json_to_text(json_data)
    except Exception as exc:
        error_file = Path("transcribe-anything-error.json")
        error_file.write_text(json_text, encoding="utf-8")
        raise
    # Disable srt_wrapping because it breaks conversion to webvtt.
    srt_file.write_text(srt_content, encoding="utf-8")
    # srt_wrap(srt_file)
    # srt_content = srt_file.read_text(encoding="utf-8")
    txt_file = output_dir / "out.txt"
    txt_file.write_text(txt_content, encoding="utf-8")
    convert_to_webvtt(srt_file, output_dir / "out.vtt")
    # print srt file
    # print(srt_content)
