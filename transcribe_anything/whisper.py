"""
Runs whisper api.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from isolated_environment import isolated_environment  # type: ignore

HERE = Path(__file__).parent
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


def get_environment() -> dict[str, Any]:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "whisper"
    deps = [
        "openai-whisper",
    ]
    if has_nvidia_smi():
        deps.append(
            f"torch=={TENSOR_VERSION}+{CUDA_VERSION} --extra-index-url {EXTRA_INDEX_URL}"
        )
    else:
        deps.append(f"torch=={TENSOR_VERSION}")
    env = isolated_environment(venv_dir, deps)
    return env


def get_computing_device() -> str:
    """Get the computing device."""
    global CUDA_AVAILABLE  # pylint: disable=global-statement
    if CUDA_AVAILABLE is None:
        env = get_environment()
        py_file = HERE / "cuda_available.py"
        rtn = subprocess.run(
            ["python", py_file], shell=False, check=False, env=env
        ).returncode
        CUDA_AVAILABLE = rtn == 0
    return "cuda" if CUDA_AVAILABLE else "cpu"


def run_whisper(  # pylint: disable=too-many-arguments
    input_wav: Path,
    device: str,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    other_args: list[str] | None = None,
) -> None:
    """Runs whisper."""
    env = get_environment()
    cmd_list = []
    if sys.platform == "win32":
        # Set the text mode to UTF-8 on Windows.
        cmd_list.extend(["chcp", "65001", "&&"])
    cmd_list.append("whisper")
    cmd_list.append(f'"{input_wav}"')
    cmd_list.append("--device")
    cmd_list.append(device)
    cmd_list.append("--model")
    cmd_list.append(model)
    cmd_list.append(f'--output_dir "{output_dir}"')
    cmd_list.append("--task")
    cmd_list.append(task)
    if language:
        cmd_list.append(f'--language "{language}"')

    if other_args:
        cmd_list.extend(other_args)
    # Remove the empty strings.
    cmd_list = [x.strip() for x in cmd_list if x.strip()]
    cmd = " ".join(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd, shell=True, universal_newlines=True, env=env, encoding="utf-8"
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
