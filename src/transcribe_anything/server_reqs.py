"""
Requirements for the daemon (FastAPI server) backend.

Following the same pattern as ``whisperx_reqs.py`` and
``insanley_fast_whisper_reqs.py``: the server runs in its own isolated
``uv``-managed virtualenv built on first use, so users get the daemon
without having to install FastAPI / uvicorn themselves.

The iso-env contains the host package's base runtime dependencies plus
the FastAPI/uvicorn server stack. The host package itself is imported
from PYTHONPATH (set by the runner) — the iso-env supplies the deps,
the host source supplies the code.
"""

from __future__ import annotations

from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import get_runtime_venv_dir

HERE = Path(__file__).parent

PYTHON_VERSION = "==3.11.*"
SERVER_ENV_NAME = "server"

# These are the minimal deps the server iso-env needs. The host package's
# base deps are pulled in too so `import transcribe_anything.api` works
# inside the iso-env when its source dir is on PYTHONPATH.
_SERVER_DEPS = [
    "fastapi>=0.110.0,<1.0.0",
    "uvicorn>=0.27.0,<1.0.0",
    "python-multipart>=0.0.9",
    "httpx>=0.27.0",
    # WS /v1/stream (#122) — FastAPI/Starlette dispatches WebSocket via
    # the ``websockets`` library; uvicorn's [standard] extra also pulls
    # it in, but the iso-env declares it explicitly so dev/test envs
    # (--no-iso-env) don't have to guess.
    "websockets>=12.0",
    # Mirror of the host base deps so `import transcribe_anything.*` works
    # against the host source via PYTHONPATH.
    "static-ffmpeg>=3.0",
    "yt-dlp>=2025.1.15",
    "appdirs>=1.4.4",
    "disklru>=1.0.7",
    "FileLock",
    "webvtt-py==0.4.6",
    "uv-iso-env>=1.0.44",
    "python-dotenv>=1.0.1",
]


def build_pyproject_toml() -> str:
    """Build the uv pyproject content for the isolated server env."""
    lines: list[str] = []
    lines.append("[build-system]")
    lines.append('requires = ["setuptools", "wheel"]')
    lines.append('build-backend = "setuptools.build_meta"')
    lines.append("")
    lines.append("[project]")
    lines.append('name = "transcribe-anything-server-backend"')
    lines.append('version = "0.1.0"')
    lines.append(f'requires-python = "{PYTHON_VERSION}"')
    lines.append("dependencies = [")
    for dep in _SERVER_DEPS:
        lines.append(f'  "{dep}",')
    lines.append("]")
    return "\n".join(lines)


def get_environment() -> IsoEnv:
    """Return the daemon's iso-env (built on first call)."""
    venv_dir = get_runtime_venv_dir(SERVER_ENV_NAME)
    build_info = PyProjectToml(build_pyproject_toml())
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    return IsoEnv(args)
