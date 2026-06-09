"""
Requirements for the realtime-streaming backend (#124).

Built on top of the WS protocol skeleton landed in #123. The iso-env
mirrors the per-backend pattern used by :mod:`whisperx_reqs` and
:mod:`sensevoice_reqs`: ``faster-whisper`` is the decoder, ``silero-vad``
chunks the inbound PCM at voice-activity boundaries, ``onnxruntime``
runs silero-vad's small ONNX model on CPU even on GPU hosts.

The streaming backend deliberately does NOT pull in ``whisper`` or
``insanely-fast-whisper`` — it is its own backend pinned to
``small.en`` for the live-captioning latency budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import get_runtime_venv_dir, has_nvidia_smi

HERE = Path(__file__).parent

PYTHON_VERSION = "==3.11.*"
STREAM_ENV_NAME = "stream"

FASTER_WHISPER_VERSION = "1.1.0"
SILERO_VAD_VERSION = "5.1.2"


def get_current_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _ctranslate2_requirement(has_nvidia: bool) -> str:
    """Pin ctranslate2 to a CUDA-bearing build when nvidia-smi is present.

    ``faster-whisper`` pulls ctranslate2 as a transitive dep, but its
    default wheel is CPU-only on Linux. The CUDA build is published as
    ``ctranslate2`` with a ``+cuda`` local version; on Windows the regular
    wheel includes CUDA support. We declare ctranslate2 explicitly so the
    iso-env resolution doesn't accidentally land on the CPU wheel.
    """
    # Both wheels currently use the same upstream version; CUDA support
    # is selected at runtime based on whether the system has CUDA libs.
    return "ctranslate2>=4.4.0,<5.0.0"


def _get_reqs(has_nvidia: bool) -> list[str]:
    return [
        f"faster-whisper=={FASTER_WHISPER_VERSION}",
        f"silero-vad=={SILERO_VAD_VERSION}",
        _ctranslate2_requirement(has_nvidia),
        "onnxruntime>=1.17.0",
        "numpy>=1.26.0,<3.0.0",
    ]


def build_pyproject_toml(has_nvidia: bool) -> str:
    deps = _get_reqs(has_nvidia)
    lines: list[str] = []
    lines.append("[build-system]")
    lines.append('requires = ["setuptools", "wheel"]')
    lines.append('build-backend = "setuptools.build_meta"')
    lines.append("")
    lines.append("[project]")
    lines.append('name = "transcribe-anything-stream-backend"')
    lines.append('version = "0.1.0"')
    lines.append(f'requires-python = "{PYTHON_VERSION}"')
    lines.append("dependencies = [")
    for dep in deps:
        lines.append(f'  "{dep}",')
    lines.append("]")
    return "\n".join(lines)


def get_environment(has_nvidia: bool | None = None) -> IsoEnv:
    """Return the streaming-backend iso-env (built on first call)."""
    venv_dir = get_runtime_venv_dir(STREAM_ENV_NAME)
    if has_nvidia is None:
        has_nvidia = has_nvidia_smi()
    build_info = PyProjectToml(build_pyproject_toml(has_nvidia))
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    return IsoEnv(args)
