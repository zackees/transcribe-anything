"""Unit tests for the isolated insane-flash backend requirements."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from transcribe_anything.flash_attention_wheels import (
    FLASH_ATTN_VERSION,
    UnsupportedFlashAttentionWheel,
    get_flash_attention_wheel,
)


def test_windows_flash_attention_wheel_candidate_is_pinned() -> None:
    wheel = get_flash_attention_wheel(has_nvidia=True, system="Windows", machine="AMD64", python_tag="cp311")

    assert FLASH_ATTN_VERSION == "2.8.3"
    assert wheel.filename == "flash_attn-2.8.3+cu128torch2.7-cp311-cp311-win_amd64.whl"
    assert wheel.sha256 == "ee22b69054b067de658e4a85183fc0d494b495770c8ff557e2d85b34f1f477fb"
    assert "mjun0812/flash-attention-prebuild-wheels" in wheel.url
    assert wheel.requirement.startswith("flash-attn @ https://")
    assert wheel.requirement.endswith(f"#sha256={wheel.sha256}")


def test_linux_x86_flash_attention_uses_manylinux_primary_wheel() -> None:
    wheel = get_flash_attention_wheel(has_nvidia=True, system="Linux", machine="x86_64", python_tag="cp311")

    assert "manylinux_2_24_x86_64.manylinux_2_28_x86_64" in wheel.filename
    assert wheel.sha256 == "f6e22ca58419150a54499c6df209b9dbf51346829b959876edd3f37fae1d7a03"
    assert wheel.primary is True


def test_linux_aarch64_flash_attention_wheel_candidate_is_pinned() -> None:
    wheel = get_flash_attention_wheel(has_nvidia=True, system="Linux", machine="aarch64", python_tag="cp311")

    assert wheel.filename == "flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_aarch64.whl"
    assert wheel.sha256 == "3ddd94bd0868905569378a716907724c3ed503c3390c57dca49a5c039be3a2ff"


def test_macos_flash_attention_is_unsupported() -> None:
    with pytest.raises(UnsupportedFlashAttentionWheel, match="not supported on macOS"):
        get_flash_attention_wheel(has_nvidia=True, system="Darwin", machine="arm64", python_tag="cp311")


def test_cpu_flash_attention_is_unsupported_with_actionable_tuple() -> None:
    with pytest.raises(UnsupportedFlashAttentionWheel, match="NVIDIA CUDA"):
        get_flash_attention_wheel(has_nvidia=False, system="Linux", machine="x86_64", python_tag="cp311")


def test_insane_flash_pyproject_contains_direct_wheel_hash() -> None:
    from transcribe_anything import flash_attention_wheels
    from transcribe_anything.insanley_fast_whisper_reqs import build_pyproject_toml

    with (
        mock.patch.object(flash_attention_wheels.platform, "system", return_value="Windows"),
        mock.patch.object(flash_attention_wheels.platform, "machine", return_value="AMD64"),
        mock.patch.object(flash_attention_wheels, "current_python_tag", return_value="cp311"),
    ):
        content = build_pyproject_toml(has_nvidia=True, flash=True)

    assert "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels" in content
    assert "#sha256=ee22b69054b067de658e4a85183fc0d494b495770c8ff557e2d85b34f1f477fb" in content
    assert "torch==2.7.0+cu128" in content
    assert "torchaudio==2.7.0+cu128" in content
    assert "https://download.pytorch.org/whl/cu128" in content


def test_non_flash_insane_pyproject_does_not_install_flash_attention() -> None:
    from transcribe_anything.insanley_fast_whisper_reqs import build_pyproject_toml

    content = build_pyproject_toml(has_nvidia=True, flash=False)

    assert "flash-attn" not in content


def test_get_environment_uses_dedicated_flash_venv() -> None:
    import transcribe_anything.insanley_fast_whisper_reqs as reqs
    from transcribe_anything import flash_attention_wheels

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

    with (
        mock.patch.object(reqs, "PyProjectToml", FakePyProjectToml),
        mock.patch.object(reqs, "IsoEnvArgs", FakeIsoEnvArgs),
        mock.patch.object(reqs, "IsoEnv", FakeIsoEnv),
        mock.patch.object(flash_attention_wheels.platform, "system", return_value="Windows"),
        mock.patch.object(flash_attention_wheels.platform, "machine", return_value="AMD64"),
        mock.patch.object(flash_attention_wheels, "current_python_tag", return_value="cp311"),
    ):
        env = reqs.get_environment(has_nvidia=True, flash=True)

    assert isinstance(env, FakeIsoEnv)
    venv_path = Path(captured["venv_path"])
    assert venv_path.name == "insanely_fast_whisper_flash"
    assert "venv" in venv_path.parts
    assert "flash-attn @" in captured["content"]


def test_shared_insane_backend_env_var_reuses_flash_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe_anything.insanley_fast_whisper_reqs as reqs
    from transcribe_anything import flash_attention_wheels

    captured: dict[str, Any] = {}

    class FakePyProjectToml:
        def __init__(self, content: str) -> None:
            captured["content"] = content

    class FakeIsoEnvArgs:
        def __init__(self, venv_path: Path, build_info: Any) -> None:
            captured["venv_path"] = venv_path
            self.venv_path = venv_path
            self.build_info = build_info

    class FakeIsoEnv:
        def __init__(self, args: Any) -> None:
            self.args = args

    monkeypatch.setenv("TRANSCRIBE_ANYTHING_SHARED_INSANE_BACKEND", "flash")
    with (
        mock.patch.object(reqs, "PyProjectToml", FakePyProjectToml),
        mock.patch.object(reqs, "IsoEnvArgs", FakeIsoEnvArgs),
        mock.patch.object(reqs, "IsoEnv", FakeIsoEnv),
        mock.patch.object(flash_attention_wheels.platform, "system", return_value="Windows"),
        mock.patch.object(flash_attention_wheels.platform, "machine", return_value="AMD64"),
        mock.patch.object(flash_attention_wheels, "current_python_tag", return_value="cp311"),
    ):
        reqs.get_environment(has_nvidia=True, flash=False)

    assert Path(captured["venv_path"]).name == "insanely_fast_whisper_flash"
    assert "flash-attn @" in captured["content"]


def test_shared_insane_backend_env_var_does_not_force_flash_without_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe_anything.insanley_fast_whisper_reqs as reqs

    captured: dict[str, Any] = {}

    class FakePyProjectToml:
        def __init__(self, content: str) -> None:
            captured["content"] = content

    class FakeIsoEnvArgs:
        def __init__(self, venv_path: Path, build_info: Any) -> None:
            captured["venv_path"] = venv_path
            self.venv_path = venv_path
            self.build_info = build_info

    class FakeIsoEnv:
        def __init__(self, args: Any) -> None:
            self.args = args

    monkeypatch.setenv("TRANSCRIBE_ANYTHING_SHARED_INSANE_BACKEND", "flash")
    with (
        mock.patch.object(reqs, "PyProjectToml", FakePyProjectToml),
        mock.patch.object(reqs, "IsoEnvArgs", FakeIsoEnvArgs),
        mock.patch.object(reqs, "IsoEnv", FakeIsoEnv),
    ):
        reqs.get_environment(has_nvidia=False, flash=False)

    assert Path(captured["venv_path"]).name == "insanely_fast_whisper"
    assert "flash-attn" not in captured["content"]
