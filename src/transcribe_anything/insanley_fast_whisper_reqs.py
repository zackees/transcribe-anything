"""
Requirements for the insanely fast whisper.
"""

import sys
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import has_nvidia_smi

HERE = Path(__file__).parent

# Set the versions
TENSOR_VERSION = "2.6.0"
CUDA_VERSION = "cu126"
TENSOR_CUDA_VERSION = f"{TENSOR_VERSION}+{CUDA_VERSION}"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"


def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_reqs_generic(has_nvidia: bool) -> list[str]:
    """Generate the requirements for the generic case."""
    deps = [
        "transformers==4.46.3",  # 4.47.X has problems with mac mps driver see fix: https://github.com/huggingface/transformers/pull/35295
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
    else:
        content_lines.append(f"torch=={TENSOR_VERSION}")
        content_lines.append(f"torchaudio=={TENSOR_VERSION}")
    if sys.platform != "darwin":
        # Add the windows specific dependencies.
        content_lines.append("intel-openmp==2024.0.3")

    return content_lines


def get_environment(has_nvidia: bool | None = None) -> IsoEnv:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "insanely_fast_whisper"
    # has_nvidia = has_nvidia_smi()
    if has_nvidia is None:
        has_nvidia = has_nvidia_smi()
    # We used to use pip to compile args, but it was hurting developement so now
    # we always use the generic requirements.
    dep_lines = _get_reqs_generic(has_nvidia)
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
    content_lines.append('requires-python = "==3.11.*"')
    content_lines.append("dependencies = [")
    for dep in dep_lines:
        content_lines.append(f'  "{dep}",')
    content_lines.append("]")

    if has_nvidia:
        content_lines.append("[tool.uv.sources]")
        content_lines.append("torch = [")
        content_lines.append("  { index = 'pytorch-cu126' },")
        content_lines.append("]")
        content_lines.append("torchaudio = [")
        content_lines.append("  { index = 'pytorch-cu126' },")
        content_lines.append("]")
        content_lines.append("[[tool.uv.index]]")
        content_lines.append('name = "pytorch-cu126"')
        content_lines.append(f'url = "{EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")

    content = "\n".join(content_lines)

    build_info = PyProjectToml(content)
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    env = IsoEnv(args)
    return env
