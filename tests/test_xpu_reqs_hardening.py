"""XPU pyproject hardening tests (issue #128).

The pytorch-xpu extra index must be `explicit = true` (only used for packages
pinned via [tool.uv.sources]) and must not rely on the weaker
`unsafe-best-match` index strategy — mirroring the existing pytorch-cu128
config.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src" / "transcribe_anything"


def _assert_hardened_xpu_index(content: str) -> None:
    """The pytorch-xpu index must be explicit and unsafe-best-match gone."""
    assert "unsafe-best-match" not in content, "index-strategy = unsafe-best-match must not be used"
    idx = content.find('name = "pytorch-xpu"')
    assert idx != -1, "pytorch-xpu index must be declared"
    assert "explicit = true" in content[idx:], "pytorch-xpu index must set explicit = true"
    # The generated content must stay valid TOML.
    parsed = tomllib.loads(content)
    indexes = parsed["tool"]["uv"]["index"]
    xpu_index = next(i for i in indexes if i["name"] == "pytorch-xpu")
    assert xpu_index["url"] == "https://download.pytorch.org/whl/xpu"
    assert xpu_index["explicit"] is True
    # pytorch-triton-xpu (torch+xpu's triton backend) is quarantined on PyPI
    # and only exists on the pytorch-xpu index, so it must be declared and
    # pinned there for resolution to succeed with an explicit index.
    sources = parsed["tool"]["uv"]["sources"]
    assert sources["pytorch-triton-xpu"] == [{"index": "pytorch-xpu"}]
    deps = parsed["project"]["dependencies"]
    assert any(d.startswith("pytorch-triton-xpu") for d in deps)
    # XPU wheels are linux/windows only; universal resolution must not
    # attempt the mac split.
    environments = parsed["tool"]["uv"]["environments"]
    assert set(environments) == {"sys_platform == 'win32'", "sys_platform == 'linux'"}


def test_whisper_xpu_pyproject_is_hardened() -> None:
    from transcribe_anything.whisper import build_pyproject_toml

    _assert_hardened_xpu_index(build_pyproject_toml(has_nvidia=False, use_xpu=True))


def test_insane_xpu_pyproject_is_hardened() -> None:
    from transcribe_anything.insanley_fast_whisper_reqs import build_pyproject_toml

    _assert_hardened_xpu_index(build_pyproject_toml(has_nvidia=False, use_xpu=True))


def test_whisperx_xpu_pyproject_is_hardened() -> None:
    from transcribe_anything.whisperx_reqs import build_pyproject_toml

    _assert_hardened_xpu_index(build_pyproject_toml(has_nvidia=False, use_xpu=True))


def test_whisper_cuda_pyproject_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hardening the XPU path must not disturb the CUDA config."""
    import transcribe_anything.whisper as whisper

    # The CUDA extra index is only emitted off-mac; force that path so the
    # assertion holds on macOS CI runners too.
    monkeypatch.setattr(whisper, "IS_MAC", False)
    content = whisper.build_pyproject_toml(has_nvidia=True, use_xpu=False)
    idx = content.find('name = "pytorch-cu128"')
    assert idx != -1
    assert "explicit = true" in content[idx:]
    assert "unsafe-best-match" not in content


def test_xpu_whisper_flash_flag_is_store_true() -> None:
    """argparse `type=bool` treats any non-empty string ("False") as truthy."""
    src = (SRC / "_xpu_whisper.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if "add_argument" in line:
            assert "type=bool" not in line, f"raw type=bool parses 'False' as True: {line.strip()}"


def test_whisperx_has_no_dead_intel_gpu_helper() -> None:
    """_has_intel_gpu() was dead code; it must be removed or actually used."""
    src = (SRC / "whisperx.py").read_text(encoding="utf-8")
    defined = "_has_intel_gpu" in src
    if defined:
        # If it exists it must be called somewhere other than its definition.
        assert src.count("_has_intel_gpu") > 1, "_has_intel_gpu defined but never called"
