"""
Runs whisper api with Apple MPS support.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import webvtt  # type: ignore
from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

HERE = Path(__file__).parent
CUDA_AVAILABLE: Optional[bool] = None


def get_environment() -> IsoEnv:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "whisper_darwin"
    content_lines: list[str] = []

    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")
    content_lines.append("[project]")
    content_lines.append('name = "project"')
    content_lines.append('version = "0.1.0"')
    content_lines.append('requires-python = "==3.11.*"')
    content_lines.append("dependencies = [")
    content_lines.append('  "whisper-mps",')
    content_lines.append('  "webvtt-py",')
    content_lines.append("]")
    content = "\n".join(content_lines)
    pyproject_toml = PyProjectToml(content)
    args = IsoEnvArgs(venv_dir, build_info=pyproject_toml)
    env = IsoEnv(args)
    return env


def _format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format."""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _json_to_srt(json_data: Dict[str, Any]) -> str:
    """Convert whisper-mps JSON output to SRT format."""
    srt_content = ""

    if "segments" not in json_data:
        # If no segments, try to create a single segment from the full text
        if "text" in json_data:
            srt_content = "1\n00:00:00,000 --> 00:01:00,000\n" + json_data["text"] + "\n\n"
        return srt_content

    for i, segment in enumerate(json_data["segments"], start=1):
        start_time = segment.get("start", 0)
        end_time = segment.get("end", start_time + 5)  # Default to 5 seconds if no end time
        text = segment.get("text", "").strip()

        if text:  # Only include non-empty segments
            srt_content += f"{i}\n"
            srt_content += f"{_format_timestamp(start_time)} --> {_format_timestamp(end_time)}\n"
            srt_content += f"{text}\n\n"

    return srt_content


def run_whisper_mac_english(  # pylint: disable=too-many-arguments
    input_wav: Path,
    model: str,
    output_dir: Path,
) -> None:
    """Runs whisper with Apple MPS acceleration.

    This function executes the whisper-mps command to transcribe audio using Apple's
    Metal Performance Shaders (MPS) for acceleration on Apple Silicon hardware.
    It then processes the JSON output to generate SRT, VTT, and TXT files.
    """
    input_wav_abs = input_wav.resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    env = get_environment()

    # whisper-mps saves the output JSON in the same directory as the input file
    output_json = input_wav_abs.parent / "output.json"

    # Prepare command
    cmd_list = [
        "whisper-mps",
        "--file-name",
        input_wav.name,  # cwd is set to the same directory as the input file
    ]

    if model:
        cmd_list.extend(["--model", model])

    # Execute command
    cmd = subprocess.list2cmdline(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = env.open_proc(cmd_list, shell=False, cwd=input_wav_abs.parent)
    while True:
        rtn = proc.poll()
        if rtn is None:
            time.sleep(0.25)
            continue
        if rtn != 0:
            raise OSError(f"Failed to execute {cmd}")
        break

    # Process output files
    if not output_json.exists():
        raise FileNotFoundError(f"whisper-mps did not generate the expected output file: {output_json}")

    # Read the JSON output
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse whisper-mps output JSON: {e}")

    # Generate output files
    # 1. SRT file
    srt_content = _json_to_srt(json_data)
    srt_file = output_dir / "out.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt_content)

    # 2. Text file
    txt_file = output_dir / "out.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(json_data.get("text", ""))

    # 3. JSON file
    json_out_file = output_dir / "out.json"
    with open(json_out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 4. VTT file
    try:
        vtt_file = output_dir / "out.vtt"
        webvtt.from_srt(srt_file).save(vtt_file)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to convert SRT to VTT: {e}\n")
