# pylint: skip-file
# flake8: noqa

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

# Set the versions
TENSOR_VERSION = "2.1.2"
CUDA_VERSION = "cu121"
TENSOR_CUDA_VERSION = f"{TENSOR_VERSION}+{CUDA_VERSION}"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"

def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# for insanely fast whisper, use:
#   pipx install insanely-fast-whisper --python python3.11

def has_nvidia_smi() -> bool:
    """Returns True if nvidia-smi is installed."""
    return shutil.which("nvidia-smi") is not None


def install_whisper_if_necessary() -> None:
    """Installs whisper if necessary."""
    install_gpu = has_nvidia_smi()
    gpu_requirements = [
        "torch==2.1.2",
        "openai-whisper"
    ]
    TENSOR_VERSION = "2.1.2"
    CUDA_VERSION = "cu121"
    TENSOR_CUDA_VERSION = f"{TENSOR_VERSION}+{CUDA_VERSION}"
    EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"

    # Installing using pipx
    try:
        # Step 1: Install Python 3.11 (handled externally, not via Python script)
        # Step 2: Install insanely-fast-whisper using pipx
        subprocess.run(["pipx", "install", "insanely-fast-whisper", "--python", "python3.11"], check=True)

        # Steps 3-5: Injecting packages into the pipx environment
        subprocess.run(["pipx", "inject", "insanely-fast-whisper", "torch==2.1.2"], check=True)
        subprocess.run(["pipx", "inject", "insanely-fast-whisper", "openai-whisper"], check=True)
        subprocess.run(["pipx", "inject", "insanely-fast-whisper", "transformers"], check=True)

        print("Whisper installation and configuration complete.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during installation: {e}", file=sys.stderr)



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
