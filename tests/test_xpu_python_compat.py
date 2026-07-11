"""Regression tests for XPU environments launched from newer host Pythons."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from iso_env import IsoEnvArgs, PyProjectToml

from transcribe_anything.xpu_iso_env import XpuIsoEnv


def _args(tmp_path: Path) -> IsoEnvArgs:
    return IsoEnvArgs(
        venv_path=tmp_path / "xpu-env",
        build_info=PyProjectToml('[project]\nname = "test"\nversion = "0.0.0"\nrequires-python = "==3.11.*"'),
    )


@pytest.mark.parametrize("method", ["run", "open_proc"])
def test_xpu_env_requests_compatible_python_during_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    """Issue #134: never resolve XPU wheels with the CPython 3.14 host ABI."""
    args = _args(tmp_path)
    env = XpuIsoEnv(args, python="3.11")
    observed: list[list[str]] = []
    sentinel = object()

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        observed.append(cmd)
        if "compile" in cmd:
            (args.venv_path / "requirements.compiled.txt").write_text("torch==2.7.1+xpu\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("transcribe_anything.xpu_iso_env.shutil.which", lambda name: "uv")
    monkeypatch.setattr("transcribe_anything.xpu_iso_env.subprocess.run", fake_run)
    monkeypatch.setattr(f"iso_env.IsoEnv.{method}", lambda *args, **kwargs: sentinel)

    assert getattr(env, method)(["python", "-V"]) is sentinel
    assert observed[0] == ["uv", "venv", "--python", "3.11"]
    assert observed[1][:4] == ["uv", "pip", "compile", "pyproject.toml"]
    assert observed[1][4:6] == ["--python", "3.11"]


def test_xpu_env_rejects_false_installed_marker_after_compile_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dependency compile failure must be fatal and must not be cached."""
    args = _args(tmp_path)
    env = XpuIsoEnv(args, python="3.11")
    args.venv_path.mkdir(parents=True)
    installed = args.venv_path / "installed"
    installed.touch()

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        if "compile" in cmd:
            raise subprocess.CalledProcessError(1, cmd, stderr="no cp314 wheel")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("transcribe_anything.xpu_iso_env.shutil.which", lambda name: "uv")
    monkeypatch.setattr("transcribe_anything.xpu_iso_env.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="no cp314 wheel"):
        env.open_proc(["python", "-V"])

    assert not installed.exists()


@pytest.mark.parametrize(
    ("module_name", "expected_python"),
    [
        ("transcribe_anything.whisper", "3.10"),
        ("transcribe_anything.insanley_fast_whisper_reqs", "3.11"),
        ("transcribe_anything.whisperx_reqs", "3.11"),
    ],
)
def test_xpu_environment_factories_pin_the_backend_python(module_name: str, expected_python: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = __import__(module_name, fromlist=["get_environment"])
    monkeypatch.setattr(module, "get_runtime_venv_dir", lambda name: tmp_path / name)
    if hasattr(module, "has_nvidia_smi"):
        monkeypatch.setattr(module, "has_nvidia_smi", lambda: False)

    env = module.get_environment(use_xpu=True)

    assert isinstance(env, XpuIsoEnv)
    assert env.python == expected_python
