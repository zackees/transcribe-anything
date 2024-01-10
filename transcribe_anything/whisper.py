
"""
Runs whisper api.
"""


import sys
import time
from pathlib import Path
import subprocess
from typing import Optional


def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# for insanely fast whisper, use:
#   pipx install insanely-fast-whisper --python python3.11


def run_insanely_fast_whisper(  # pylint: disable=too-many-arguments
    input_wav: Path,
    device: str,  # pylint: disable=unused-argument
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    other_args: Optional[list[str]]
) -> None:
    """Runs insanely fast whisper."""
    cmd_list = []
    model = f"openai/whisper-{model}"
    if sys.platform == "win32":
        # Set the text mode to UTF-8 on Windows.
        cmd_list.extend(["chcp", "65001", "&&"])
    cmd_list += [
        "insanely-fast-whisper",
        "--file-name", str(input_wav),
        "--device-id", "0",
        "--model-name", model,
        "--task", task,
        "--language", language,
        "--transcript-path", str(output_dir),
    ]
    if other_args:
        cmd_list.extend(other_args)
    # Remove the empty strings.
    cmd_list = [x.strip() for x in cmd_list if x.strip()]
    cmd = " ".join(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd, shell=True, universal_newlines=True,
        encoding="utf-8"
    )
    while True:
        rtn = proc.poll()
        if rtn is None:
            time.sleep(0.25)
            continue
        if rtn != 0:
            msg = f"Failed to execute {cmd}\n "
            raise OSError(msg)
        break


def run_whisper(  # pylint: disable=too-many-arguments
    input_wav: Path,
    device: str,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    other_args: Optional[list[str]]
) -> None:
    """Runs whisper."""

    cmd_list = []
    if sys.platform == "win32":
        # Set the text mode to UTF-8 on Windows.
        cmd_list.extend(["chcp", "65001", "&&"])

    cmd_list.extend(
        [
            "whisper",
            f'"{input_wav}"',
            "--device",
            device,
            model,
            f'--output_dir "{output_dir}"',
            task,
            language,
        ]
    )
    if other_args:
        cmd_list.extend(other_args)
    # Remove the empty strings.
    cmd_list = [x.strip() for x in cmd_list if x.strip()]
    cmd = " ".join(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd, shell=True, universal_newlines=True,
        encoding="utf-8"
    )
    while True:
        rtn = proc.poll()
        if rtn is None:
            time.sleep(0.25)
            continue
        if rtn != 0:
            msg = f"Failed to execute {cmd}\n "
            raise OSError(msg)
        break
