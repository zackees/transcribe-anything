
"""
Runs whisper api.
"""

import shutil
import sys
import time
from pathlib import Path
import subprocess
from typing import Optional

from isolated_environment import IsolatedEnvironment  # type: ignore

HERE = Path(__file__).parent
ENV: Optional[IsolatedEnvironment] = None
CUDA_AVAILABLE: Optional[bool] = None

# Set the versions
TENSOR_VERSION = "2.1.2"
CUDA_VERSION = "cu121"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"


# for insanely fast whisper, use:
#   pipx install insanely-fast-whisper --python python3.11

def has_nvidia_smi() -> bool:
    """Returns True if nvidia-smi is installed."""
    return shutil.which("nvidia-smi") is not None


def get_environment() -> IsolatedEnvironment:
    """Returns the environment."""
    global ENV  # pylint: disable=global-statement
    if ENV is not None:
        return ENV
    venv_dir = HERE / "venv" / "whisper"
    env = IsolatedEnvironment(HERE / "venv" / "whisper")
    if not venv_dir.exists():
        env.install_environment()
        if has_nvidia_smi():
            env.pip_install(f"torch=={TENSOR_VERSION}", extra_index=EXTRA_INDEX_URL)
        else:
            env.pip_install(f"torch=={TENSOR_VERSION}")
        env.pip_install("openai-whisper")
    ENV = env
    return env


def get_computing_device() -> str:
    """Get the computing device."""
    global CUDA_AVAILABLE  # pylint: disable=global-statement
    if CUDA_AVAILABLE is None:
        iso_env = get_environment()
        env = iso_env.environment()
        py_file = HERE / "cuda_available.py"
        rtn = subprocess.run([
            "python", py_file
        ], check=False, env=env).returncode
        CUDA_AVAILABLE = rtn == 0
    return "cuda" if CUDA_AVAILABLE else "cpu"

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

    iso_env = get_environment()
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
        env=iso_env.environment(),
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
