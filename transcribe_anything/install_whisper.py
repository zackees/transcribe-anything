"""
Installs whisper in an isolated environment.
"""

import os
import shutil
from pathlib import Path

from isolated_environment import IsolatedEnvironment  # type: ignore

TENSOR_VERSION = "2.1.2"
CUDA_VERSION = "cu121"
TENSOR_CUDA_VERSION = f"{TENSOR_VERSION}+{CUDA_VERSION}"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"

HERE = Path(os.path.abspath(os.path.dirname(__file__)))


def has_nvidia_smi() -> bool:
    """Returns True if nvidia-smi is installed."""
    return shutil.which("nvidia-smi") is not None


def unit_test() -> None:
    """Unit test."""
    from tempfile import TemporaryDirectory  # pylint: disable=import-outside-toplevel

    with TemporaryDirectory() as tmpdir:
        print(f"Using temporary directory {tmpdir}")
        iso_env = IsolatedEnvironment(HERE / "transcribe_anything_env")
        iso_env.install_environment()
        iso_env.pip_install("torch==2.1.2", EXTRA_INDEX_URL)
        iso_env.pip_install("openai-whisper")
        iso_env.run(["whisper", "--help"])


if __name__ == "__main__":
    unit_test()
