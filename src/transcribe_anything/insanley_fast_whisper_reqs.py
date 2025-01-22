"""
Requirements for the insanely fast whisper.
"""

import sys
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

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


def get_environment() -> IsoEnv:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "insanely_fast_whisper"
    deps = [
        "openai-whisper==20240930",
        "insanely-fast-whisper==0.0.15",
        "torchaudio==2.1.2",
        "datasets==2.17.1",
        "pytorch-lightning==2.1.4",
        "torchmetrics~=1.3.0",
        "srtranslator==0.2.6",
        "numpy==1.26.4",
    ]

    content_lines: list[str] = []

    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")

    content_lines.append("[project]")
    content_lines.append('name = "project"')
    content_lines.append('version = "0.1.0"')
    content_lines.append('requires-python = "==3.10.*"')
    content_lines.append("dependencies = [")
    for dep in deps:
        content_lines.append(f'  "{dep}",')
    needs_extra_index = False
    if has_nvidia_smi():
        needs_extra_index = True
        content_lines.append(f'  "torch=={TENSOR_CUDA_VERSION}",')
    else:
        content_lines.append(f'  "torch=={TENSOR_VERSION}",')
    if sys.platform != "darwin":
        # Add the windows specific dependencies.
        content_lines.append('  "intel-openmp==2024.0.3",')
    content_lines.append("]")
    content_lines.append("")

    # if has_nvidia_smi():
    #     deps.append(f"torch=={TENSOR_CUDA_VERSION} --extra-index-url {EXTRA_INDEX_URL}")
    # else:
    #     deps.append(f"torch=={TENSOR_VERSION}")
    # if sys.platform != "darwin":
    #     # Add the windows specific dependencies.
    #     deps.append("intel-openmp==2024.0.3")

    if needs_extra_index:
        # [tool.uv.sources]
        #   torch = [
        #   { index = "pytorch-cu121", marker = "platform_system == 'Windows'" },
        # ]
        content_lines.append("[tool.uv.sources]")
        content_lines.append("torch = [")
        content_lines.append("  { index = 'pytorch-cu121' },")
        content_lines.append("]")

        # [[tool.uv.index]]
        # name = "pytorch-cu121"
        # url = "https://download.pytorch.org/whl/cu121"
        # explicit = true

        content_lines.append("[[tool.uv.index]]")
        content_lines.append('name = "pytorch-cu121"')
        content_lines.append(f'url = "{EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")

        # deps.append(f"--extra-index-url {EXTRA_INDEX_URL}")

    content = "\n".join(content_lines)
    build_info = PyProjectToml(content)
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    env = IsoEnv(args)
    return env
