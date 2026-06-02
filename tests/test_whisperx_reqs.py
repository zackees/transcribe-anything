"""Unit tests for the isolated WhisperX backend requirements."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEAVY_BACKEND_DEPS = {
    "ctranslate2",
    "faster-whisper",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "whisperx",
}


def _package_name(requirement: str) -> str:
    """Return the normalized package name from a requirement string."""
    return re.split(r"\s|<|>|=|!|~|;|\[", requirement, maxsplit=1)[0].lower().replace("_", "-")


def test_top_level_dependencies_do_not_leak_whisperx_stack() -> None:
    """WhisperX and its heavy AI stack must stay out of top-level installs."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    dep_names = {_package_name(dep) for dep in deps}

    leaked = sorted(HEAVY_BACKEND_DEPS.intersection(dep_names))
    assert leaked == [], f"WhisperX backend dependencies belong in whisperx_reqs.py, not pyproject.toml: {leaked}"


def test_whisperx_reqs_include_pinned_backend_stack() -> None:
    """The isolated env should include pinned WhisperX and torch dependencies."""
    from transcribe_anything.whisperx_reqs import _get_reqs_generic

    deps = _get_reqs_generic(has_nvidia=False)
    dep_names = {_package_name(dep) for dep in deps}

    assert any(dep.startswith("whisperx==") for dep in deps), deps
    assert "torch" in dep_names
    assert "torchaudio" in dep_names
    assert "whisperx" in dep_names


def test_whisperx_reqs_select_cuda_torch_when_nvidia_is_present() -> None:
    """CUDA installs should use CUDA torch wheels; CPU installs should not."""
    from transcribe_anything.whisperx_reqs import _get_reqs_generic

    with mock.patch.object(sys, "platform", "linux"):
        cuda_deps = _get_reqs_generic(has_nvidia=True)
        cpu_deps = _get_reqs_generic(has_nvidia=False)

    assert any(dep.startswith("torch==") and "+cu" in dep for dep in cuda_deps), cuda_deps
    assert any(dep.startswith("torchaudio==") and "+cu" in dep for dep in cuda_deps), cuda_deps
    assert any(dep.startswith("torch==") and "+cu" not in dep for dep in cpu_deps), cpu_deps
    assert any(dep.startswith("torchaudio==") and "+cu" not in dep for dep in cpu_deps), cpu_deps


def test_get_environment_uses_dedicated_whisperx_venv() -> None:
    """WhisperX must use its own isolated env, not the insane backend env."""
    import transcribe_anything.whisperx_reqs as reqs

    captured: dict[str, Any] = {}

    class FakePyProjectToml:
        def __init__(self, content: str) -> None:
            captured["content"] = content
            self.content = content

    class FakeIsoEnvArgs:
        def __init__(self, venv_path: Path, build_info: Any) -> None:
            captured["venv_path"] = venv_path
            captured["build_info"] = build_info
            self.venv_path = venv_path
            self.build_info = build_info

    class FakeIsoEnv:
        def __init__(self, args: Any) -> None:
            captured["args"] = args
            self.args = args

    with mock.patch.object(reqs, "PyProjectToml", FakePyProjectToml):
        with mock.patch.object(reqs, "IsoEnvArgs", FakeIsoEnvArgs):
            with mock.patch.object(reqs, "IsoEnv", FakeIsoEnv):
                with mock.patch.object(reqs, "has_nvidia_smi", return_value=False):
                    env = reqs.get_environment()

    assert isinstance(env, FakeIsoEnv)
    venv_path = Path(captured["venv_path"])
    assert venv_path.name == "whisperx"
    assert "venv" in venv_path.parts
    assert "insanely_fast_whisper" not in str(venv_path)
    assert "whisperx==" in captured["content"]
    assert 'requires-python = "==3.11.*"' in captured["content"]
