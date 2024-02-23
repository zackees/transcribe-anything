"""
Requirements for the insanely fast whisper.
"""

import sys
from pathlib import Path
from typing import Any

from isolated_environment import isolated_environment  # type: ignore

from transcribe_anything.util import has_nvidia_smi

HERE = Path(__file__).parent

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


def get_environment() -> dict[str, Any]:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "insanely_fast_whisper"
    deps = [
        "openai-whisper",
        "insanely-fast-whisper==0.0.13 --ignore-requires-python",
        "torchaudio==2.1.2",
        "pytorch-lightning==2.1.4",
        "torchmetrics~=1.3.0",
        "srtranslator==0.2.6",
    ]
    if has_nvidia_smi():
        deps.append(f"torch=={TENSOR_CUDA_VERSION} --extra-index-url {EXTRA_INDEX_URL}")
    else:
        deps.append(f"torch=={TENSOR_VERSION}")
    if sys.platform != "darwin":
        # Add the windows specific dependencies.
        deps.append("intel-openmp==2024.0.2")
    env = isolated_environment(venv_dir, deps)
    return env
