"""
Runs whisper api.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

HERE = Path(__file__).parent
CUDA_AVAILABLE: Optional[bool] = None

# whisper-mps --file-name tests/localfile/video.mp4


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
    content_lines.append("]")
    content = "\n".join(content_lines)
    pyproject_toml = PyProjectToml(content)
    args = IsoEnvArgs(venv_dir, build_info=pyproject_toml)
    env = IsoEnv(args)
    return env


def run_whisper_mac_english(  # pylint: disable=too-many-arguments
    input_wav: Path,
    model: str,
    output_dir: Path,
) -> None:
    """Runs whisper."""
    input_wav_abs = input_wav.resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    env = get_environment()
    cmd_list = []
    cmd_list.append("whisper-mps")
    cmd_list.append("--file-name")
    cmd_list.append(input_wav.name)  # cwd is set to the same directory as the input file.
    if model:
        cmd_list.append("--model")
        cmd_list.append(model)

    # Remove the empty strings.
    cmd_list = [str(x).strip() for x in cmd_list if str(x).strip()]
    # cmd = " ".join(cmd_list)
    cmd = subprocess.list2cmdline(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = env.open_proc(cmd_list, shell=False, cwd=input_wav_abs.parent)
    while True:
        rtn = proc.poll()
        if rtn is None:
            time.sleep(0.25)
            continue
        if rtn != 0:
            msg = f"Failed to execute {cmd}\n "
            raise OSError(msg)
        break
