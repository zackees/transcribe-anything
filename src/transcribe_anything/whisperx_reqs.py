"""
Requirements for the WhisperX backend.
"""

import sys
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import get_runtime_venv_dir, has_nvidia_smi

HERE = Path(__file__).parent

WHISPERX_VERSION = "3.8.6"
PYTHON_VERSION = "==3.11.*"
TORCH_VERSION = "2.8.0"
TORCHAUDIO_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
CUDA_VERSION = "cu128"
XPU_VERSION = "xpu"
CUDA_EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"
XPU_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/xpu"


def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _torch_requirement(package: str, version: str, has_nvidia: bool, use_xpu: bool = False) -> str:
    """Return the torch-family requirement for CPU, CUDA, or XPU installs."""
    if use_xpu:
        return f"{package}=={version}+{XPU_VERSION}"
    if has_nvidia:
        return f"{package}=={version}+{CUDA_VERSION}"
    return f"{package}=={version}"


def _get_reqs_generic(has_nvidia: bool, use_xpu: bool = False) -> list[str]:
    """Generate WhisperX backend requirements."""
    return [
        f"whisperx=={WHISPERX_VERSION}",
        _torch_requirement("torch", TORCH_VERSION, has_nvidia, use_xpu=use_xpu),
        _torch_requirement("torchaudio", TORCHAUDIO_VERSION, has_nvidia, use_xpu=use_xpu),
        _torch_requirement("torchvision", TORCHVISION_VERSION, has_nvidia, use_xpu=use_xpu),
    ]


def build_pyproject_toml(has_nvidia: bool, use_xpu: bool = False) -> str:
    """Build the uv pyproject content for the isolated WhisperX env."""
    dep_lines = _get_reqs_generic(has_nvidia, use_xpu=use_xpu)
    dep_lines = [line.strip() for line in dep_lines if line.strip()]

    content_lines: list[str] = []
    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")

    content_lines.append("[project]")
    content_lines.append('name = "transcribe-anything-whisperx-backend"')
    content_lines.append('version = "0.1.0"')
    content_lines.append(f'requires-python = "{PYTHON_VERSION}"')
    content_lines.append("dependencies = [")
    for dep in dep_lines:
        content_lines.append(f'  "{dep}",')
    content_lines.append("]")

    if has_nvidia and not use_xpu:
        content_lines.append("")
        content_lines.append("[tool.uv.sources]")
        for package in ("torch", "torchaudio", "torchvision"):
            content_lines.append(f"{package} = [")
            content_lines.append("  { index = 'pytorch-cu128' },")
            content_lines.append("]")
        content_lines.append("")
        content_lines.append("[[tool.uv.index]]")
        content_lines.append('name = "pytorch-cu128"')
        content_lines.append(f'url = "{CUDA_EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")
    elif use_xpu:
        content_lines.append("")
        content_lines.append("[tool.uv]")
        content_lines.append('index-strategy = "unsafe-best-match"')
        content_lines.append('override-dependencies = [')
        content_lines.append('    "torchcodec; python_version < \'3.0\'",')
        content_lines.append(']')
        content_lines.append("")
        content_lines.append("[tool.uv.sources]")
        for package in ("torch", "torchaudio", "torchvision"):
            content_lines.append(f"{package} = [")
            content_lines.append("  { index = 'pytorch-xpu' },")
            content_lines.append("]")
        content_lines.append("")
        content_lines.append("[[tool.uv.index]]")
        content_lines.append('name = "pytorch-xpu"')
        content_lines.append(f'url = "{XPU_EXTRA_INDEX_URL}"')

    return "\n".join(content_lines)


def get_environment(has_nvidia: bool | None = None, use_xpu: bool = False) -> IsoEnv:
    """Returns the isolated WhisperX environment."""
    venv_dir = get_runtime_venv_dir("whisperx-xpu" if use_xpu else "whisperx")
    if has_nvidia is None:
        has_nvidia = has_nvidia_smi()
    build_info = PyProjectToml(build_pyproject_toml(has_nvidia, use_xpu=use_xpu))
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    env = IsoEnv(args)
    return env
