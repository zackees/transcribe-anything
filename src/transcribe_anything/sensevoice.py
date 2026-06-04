"""
SenseVoice (FunASR) backend.

Wraps the FunASR ``AutoModel(model="iic/SenseVoiceSmall", ...)`` pipeline
behind the same contract as the other backends: write ``out.{json,srt,vtt,txt}``
and (when diarizing) ``speaker.json`` to ``output_dir``.

The actual inference runs inside an isolated venv via ``sensevoice_runner.py``;
the host process never imports ``funasr``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import static_ffmpeg  # type: ignore
import static_ffmpeg.run as static_ffmpeg_run  # type: ignore

from transcribe_anything.sensevoice_reqs import get_environment
from transcribe_anything.util import (
    get_static_ffmpeg_runtime_dir,
    has_nvidia_smi,
)

HERE = Path(__file__).parent

_DIARIZE_FLAGS = {"--diarize", "-d"}


def _default_device() -> str:
    if has_nvidia_smi():
        return "cuda:0"
    if sys.platform == "darwin":
        return "mps"
    return "cpu"


def _extract_flag_value(other_args: list[str] | None, flag: str) -> tuple[list[str], str | None]:
    """Pop ``flag`` (and its value) from ``other_args``, returning the residual list and the value."""
    if not other_args:
        return [], None
    args = list(other_args)
    if flag not in args:
        return args, None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        del args[idx]
        return args, None
    value = args[idx + 1]
    del args[idx : idx + 2]
    return args, value


def _extract_flag(other_args: list[str] | None, flags: set[str]) -> tuple[list[str], bool]:
    if not other_args:
        return [], False
    args: list[str] = []
    found = False
    for arg in other_args:
        if arg in flags:
            found = True
            continue
        args.append(arg)
    return args, found


def run_sensevoice(  # pylint: disable=too-many-arguments,too-many-locals
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    hugging_face_token: str | None = None,
    other_args: list[str] | None = None,
) -> None:
    """Run SenseVoice through its isolated environment.

    ``model`` is accepted for interface parity but ignored (the FunASR
    AutoModel always loads ``iic/SenseVoiceSmall``). ``task`` is also
    ignored: SenseVoice is transcription-only (no translate task).
    """
    del task  # SenseVoice has no translate task.
    del model  # Single hard-coded model id.

    ffmpeg_cache = get_static_ffmpeg_runtime_dir()
    static_ffmpeg_run.LOCK_FILE = str(ffmpeg_cache / "lock.file")
    static_ffmpeg.add_paths(download_dir=str(ffmpeg_cache / static_ffmpeg_run.get_platform_key()))

    passthrough, diarize = _extract_flag(other_args, _DIARIZE_FLAGS)
    passthrough, device_override = _extract_flag_value(passthrough, "--device")
    passthrough, hub_override = _extract_flag_value(passthrough, "--hub")
    passthrough, language_override = _extract_flag_value(passthrough, "--language")

    if passthrough:
        sys.stderr.write(
            "Warning: ignoring unsupported SenseVoice args: " + " ".join(passthrough) + "\n"
        )

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    output_dir.mkdir(parents=True, exist_ok=True)
    iso_env = get_environment()

    runner = HERE / "sensevoice_runner.py"
    if not runner.exists():
        raise FileNotFoundError(f"SenseVoice runner script missing: {runner}")

    final_language = language_override or language or "auto"
    cmd_list: list[str] = [
        str(runner),
        "--input-wav",
        str(input_wav),
        "--output-dir",
        str(output_dir),
        "--device",
        device_override or _default_device(),
        "--language",
        final_language,
        "--hub",
        hub_override or "ms",
    ]
    if diarize:
        cmd_list.append("--diarize")
    if hugging_face_token:
        cmd_list.extend(["--hf-token", hugging_face_token])

    cmd_str = subprocess.list2cmdline(cmd_list)
    safe_cmd_str = cmd_str
    if hugging_face_token:
        safe_cmd_str = safe_cmd_str.replace(hugging_face_token, "<REDACTED>")
    sys.stderr.write(f"Running:\n  {safe_cmd_str}\n")

    try:
        iso_env.run(
            cmd_list,
            shell=False,
            check=True,
            universal_newlines=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise OSError(f"Failed to execute {safe_cmd_str}\n") from exc
