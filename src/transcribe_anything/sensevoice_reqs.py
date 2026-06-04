"""
Requirements for the SenseVoice (FunASR) backend.
"""

import sys
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import get_runtime_venv_dir, has_nvidia_smi

HERE = Path(__file__).parent

FUNASR_VERSION = "1.1.14"
MODELSCOPE_VERSION = "1.18.0"
PYTHON_VERSION = "==3.11.*"
TORCH_VERSION = "2.4.1"
TORCHAUDIO_VERSION = "2.4.1"
CUDA_VERSION = "cu124"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"


def _torch_requirement(package: str, version: str, has_nvidia: bool) -> str:
    """Return the torch-family requirement for CPU or CUDA installs."""
    if has_nvidia:
        return f"{package}=={version}+{CUDA_VERSION}"
    return f"{package}=={version}"


def _get_reqs_generic(has_nvidia: bool) -> list[str]:
    """Generate SenseVoice/FunASR backend requirements.

    Pins follow FunASR's own compatibility window:
    - torch 2.4.x is the latest line FunASR 1.1.x has been validated against;
      newer torch (2.6+) trips kaldi-native-fbank wheels on some platforms.
    - numpy<2.0: FunASR + onnxruntime transitive deps break under numpy 2.x.
    """
    return [
        f"funasr=={FUNASR_VERSION}",
        f"modelscope>={MODELSCOPE_VERSION}",
        _torch_requirement("torch", TORCH_VERSION, has_nvidia),
        _torch_requirement("torchaudio", TORCHAUDIO_VERSION, has_nvidia),
        "soundfile>=0.12",
        "librosa>=0.10",
        "numpy<2.0",
    ]


def build_pyproject_toml(has_nvidia: bool) -> str:
    """Build the uv pyproject content for the isolated SenseVoice env."""
    dep_lines = _get_reqs_generic(has_nvidia)
    dep_lines = [line.strip() for line in dep_lines if line.strip()]

    content_lines: list[str] = []
    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")

    content_lines.append("[project]")
    content_lines.append('name = "transcribe-anything-sensevoice-backend"')
    content_lines.append('version = "0.1.0"')
    content_lines.append(f'requires-python = "{PYTHON_VERSION}"')
    content_lines.append("dependencies = [")
    for dep in dep_lines:
        content_lines.append(f'  "{dep}",')
    content_lines.append("]")

    if has_nvidia:
        content_lines.append("")
        content_lines.append("[tool.uv.sources]")
        for package in ("torch", "torchaudio"):
            content_lines.append(f"{package} = [")
            content_lines.append(f"  {{ index = 'pytorch-{CUDA_VERSION}' }},")
            content_lines.append("]")
        content_lines.append("")
        content_lines.append("[[tool.uv.index]]")
        content_lines.append(f'name = "pytorch-{CUDA_VERSION}"')
        content_lines.append(f'url = "{EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")

    return "\n".join(content_lines)


def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_environment(has_nvidia: bool | None = None) -> IsoEnv:
    """Returns the isolated SenseVoice environment."""
    venv_dir = get_runtime_venv_dir("sensevoice")
    if has_nvidia is None:
        has_nvidia = has_nvidia_smi()
    build_info = PyProjectToml(build_pyproject_toml(has_nvidia))
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    env = IsoEnv(args)
    return env
