"""
Requirements for the insanely fast whisper.
"""

import os
import sys
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.flash_attention_wheels import (
    SUPPORTED_PYTHON_TAG,
    get_flash_attention_wheel,
)
from transcribe_anything.util import has_nvidia_smi

HERE = Path(__file__).parent

# Set the versions
TENSOR_VERSION = "2.7.0"
CUDA_VERSION = "cu128"
TENSOR_CUDA_VERSION = f"{TENSOR_VERSION}+{CUDA_VERSION}"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"
PYTHON_VERSION = "==3.11.*"
INSANE_ENV_NAME = "insanely_fast_whisper"
INSANE_FLASH_ENV_NAME = "insanely_fast_whisper_flash"
SHARED_INSANE_BACKEND_ENV_VAR = "TRANSCRIBE_ANYTHING_SHARED_INSANE_BACKEND"


def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_reqs_generic(has_nvidia: bool) -> list[str]:
    """Generate the requirements for the generic case."""
    deps = [
        # Whisper long-form/chunk timestamp fixes need >=4.53.0 (PRs #34537, #35750)
        "transformers==4.55.4",
        "pyannote.audio==3.3.2",
        "openai-whisper==20240930",
        "insanely-fast-whisper==0.0.15",
        "datasets==2.17.1",
        "pytorch-lightning==2.5.0",
        "torchmetrics==1.6.1",
        "srtranslator==0.3.5",
        # "numpy==2.2.0",
        "safeIO==1.2",
        "llvmlite==0.44.0",
        "numba==0.61.0",
    ]

    content_lines: list[str] = []

    for dep in deps:
        content_lines.append(dep)
    if has_nvidia:
        content_lines.append(f"torch=={TENSOR_CUDA_VERSION}")
        content_lines.append(f"torchaudio=={TENSOR_CUDA_VERSION}")
        # torch 2.7.0+cu128 dlopens libcusparseLt.so.0 at import on Linux; the
        # lib is NOT bundled with the torch wheel. nvidia-cusparselt-cu12 ships
        # it. Without this dep, `import torch` raises ImportError and the
        # insane backend silently falls back to CPU (see issue #35).
        if sys.platform.startswith("linux"):
            content_lines.append("nvidia-cusparselt-cu12")
    else:
        content_lines.append(f"torch=={TENSOR_VERSION}")
        content_lines.append(f"torchaudio=={TENSOR_VERSION}")
    if sys.platform != "darwin":
        # Add the windows specific dependencies.
        content_lines.append("intel-openmp==2024.0.3")

    return content_lines


def build_pyproject_toml(has_nvidia: bool, flash: bool = False) -> str:
    """Build the uv pyproject content for the isolated insane backend env."""
    dep_lines = _get_reqs_generic(has_nvidia)
    if flash:
        wheel = get_flash_attention_wheel(has_nvidia=has_nvidia, python_tag=SUPPORTED_PYTHON_TAG)
        dep_lines.append(wheel.requirement)
    # filter out empty lines
    dep_lines = [line.strip() for line in dep_lines if line.strip()]

    content_lines: list[str] = []

    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")

    content_lines.append("[project]")
    content_lines.append('name = "project"')
    content_lines.append('version = "0.1.0"')
    content_lines.append(f'requires-python = "{PYTHON_VERSION}"')
    content_lines.append("dependencies = [")
    for dep in dep_lines:
        content_lines.append(f'  "{dep}",')
    content_lines.append("]")

    # Constrain setuptools < 82 for build isolation because
    # openai-whisper==20240930 imports pkg_resources, which was removed in
    # setuptools 82. Mirrors the same guard in whisper.py.
    content_lines.append("")
    content_lines.append("[tool.uv]")
    content_lines.append('build-constraint-dependencies = ["setuptools<82"]')

    if has_nvidia:
        content_lines.append("[tool.uv.sources]")
        content_lines.append("torch = [")
        content_lines.append("  { index = 'pytorch-cu128' },")
        content_lines.append("]")
        content_lines.append("torchaudio = [")
        content_lines.append("  { index = 'pytorch-cu128' },")
        content_lines.append("]")
        content_lines.append("[[tool.uv.index]]")
        content_lines.append('name = "pytorch-cu128"')
        content_lines.append(f'url = "{EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")

    return "\n".join(content_lines)


def _use_shared_flash_backend(has_nvidia: bool, flash: bool) -> bool:
    """Return True when normal insane should reuse the flash-capable env."""
    if flash:
        return True
    value = os.environ.get(SHARED_INSANE_BACKEND_ENV_VAR, "").strip().lower()
    return has_nvidia and value in {"1", "true", "yes", "y", "on", "flash", "insane-flash"}


def get_environment(has_nvidia: bool | None = None, flash: bool = False) -> IsoEnv:
    """Returns the isolated insane or insane-flash environment."""
    if has_nvidia is None:
        has_nvidia = has_nvidia_smi()
    flash = _use_shared_flash_backend(has_nvidia=has_nvidia, flash=flash)
    env_name = INSANE_FLASH_ENV_NAME if flash else INSANE_ENV_NAME
    venv_dir = HERE / "venv" / env_name
    content = build_pyproject_toml(has_nvidia=has_nvidia, flash=flash)
    build_info = PyProjectToml(content)
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    env = IsoEnv(args)
    return env
