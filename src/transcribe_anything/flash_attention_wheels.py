"""Pinned FlashAttention wheel manifest for the insane-flash backend."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

FLASH_ATTN_VERSION = "2.8.3"
SUPPORTED_PYTHON_TAG = "cp311"
SUPPORTED_TORCH_MINOR = "2.7"
SUPPORTED_CUDA_TAG = "cu128"


class UnsupportedFlashAttentionWheel(RuntimeError):
    """Raised when no controlled FlashAttention wheel exists for a host tuple."""


@dataclass(frozen=True)
class FlashAttentionWheel:
    """A checked FlashAttention wheel candidate for one platform tuple."""

    system: str
    machine: str
    python_tag: str
    torch_minor: str
    cuda_tag: str
    filename: str
    url: str
    sha256: str
    source: str
    release: str
    primary: bool = True

    @property
    def requirement(self) -> str:
        """Return a PEP 508 direct URL requirement with a hash fragment."""
        return f"flash-attn @ {self.url}#sha256={self.sha256}"


FLASH_ATTN_WHEELS: tuple[FlashAttentionWheel, ...] = (
    FlashAttentionWheel(
        system="windows",
        machine="x86_64",
        python_tag=SUPPORTED_PYTHON_TAG,
        torch_minor=SUPPORTED_TORCH_MINOR,
        cuda_tag=SUPPORTED_CUDA_TAG,
        filename="flash_attn-2.8.3+cu128torch2.7-cp311-cp311-win_amd64.whl",
        url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.11/flash_attn-2.8.3%2Bcu128torch2.7-cp311-cp311-win_amd64.whl",
        sha256="ee22b69054b067de658e4a85183fc0d494b495770c8ff557e2d85b34f1f477fb",
        source="mjun0812/flash-attention-prebuild-wheels",
        release="v0.7.11",
    ),
    FlashAttentionWheel(
        system="linux",
        machine="x86_64",
        python_tag=SUPPORTED_PYTHON_TAG,
        torch_minor=SUPPORTED_TORCH_MINOR,
        cuda_tag=SUPPORTED_CUDA_TAG,
        filename="flash_attn-2.8.3+cu128torch2.7-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.7-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        sha256="f6e22ca58419150a54499c6df209b9dbf51346829b959876edd3f37fae1d7a03",
        source="mjun0812/flash-attention-prebuild-wheels",
        release="v0.7.16",
    ),
    FlashAttentionWheel(
        system="linux",
        machine="x86_64",
        python_tag=SUPPORTED_PYTHON_TAG,
        torch_minor=SUPPORTED_TORCH_MINOR,
        cuda_tag=SUPPORTED_CUDA_TAG,
        filename="flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl",
        url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.7-cp311-cp311-linux_x86_64.whl",
        sha256="562ada63800388bfe9733e37feb09992d59a12a386430f40a2b119c0ff68c6ad",
        source="mjun0812/flash-attention-prebuild-wheels",
        release="v0.7.16",
        primary=False,
    ),
    FlashAttentionWheel(
        system="linux",
        machine="aarch64",
        python_tag=SUPPORTED_PYTHON_TAG,
        torch_minor=SUPPORTED_TORCH_MINOR,
        cuda_tag=SUPPORTED_CUDA_TAG,
        filename="flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_aarch64.whl",
        url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3%2Bcu128torch2.7-cp311-cp311-linux_aarch64.whl",
        sha256="3ddd94bd0868905569378a716907724c3ed503c3390c57dca49a5c039be3a2ff",
        source="mjun0812/flash-attention-prebuild-wheels",
        release="v0.6.4",
    ),
)


def current_python_tag() -> str:
    """Return the current CPython ABI tag used by wheel filenames."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def normalize_system(system: str | None = None) -> str:
    """Normalize OS names into manifest keys."""
    value = (system or platform.system()).strip().lower()
    if value in {"win32", "windows"}:
        return "windows"
    if value.startswith("linux"):
        return "linux"
    if value in {"darwin", "macos", "mac"}:
        return "darwin"
    return value


def normalize_machine(machine: str | None = None) -> str:
    """Normalize architecture names into manifest keys."""
    value = (machine or platform.machine()).strip().lower()
    if value in {"amd64", "x64", "x86-64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "aarch64"
    return value


def expected_tuple_text(
    *,
    system: str | None = None,
    machine: str | None = None,
    python_tag: str | None = None,
    torch_minor: str = SUPPORTED_TORCH_MINOR,
    cuda_tag: str = SUPPORTED_CUDA_TAG,
) -> str:
    """Return a concise tuple string for diagnostics."""
    return f"os={normalize_system(system)} arch={normalize_machine(machine)} " f"python={python_tag or current_python_tag()} torch={torch_minor} cuda={cuda_tag}"


def get_flash_attention_wheel(
    *,
    has_nvidia: bool,
    system: str | None = None,
    machine: str | None = None,
    python_tag: str | None = None,
    torch_minor: str = SUPPORTED_TORCH_MINOR,
    cuda_tag: str = SUPPORTED_CUDA_TAG,
) -> FlashAttentionWheel:
    """Return the primary pinned wheel for the current supported tuple."""
    normalized_system = normalize_system(system)
    normalized_machine = normalize_machine(machine)
    selected_python_tag = python_tag or current_python_tag()
    tuple_text = expected_tuple_text(
        system=normalized_system,
        machine=normalized_machine,
        python_tag=selected_python_tag,
        torch_minor=torch_minor,
        cuda_tag=cuda_tag,
    )
    if normalized_system == "darwin":
        raise UnsupportedFlashAttentionWheel("insane-flash is not supported on macOS because this backend requires CUDA FlashAttention. " f"Use --device mlx on Apple Silicon. Tuple: {tuple_text}.")
    if not has_nvidia:
        raise UnsupportedFlashAttentionWheel("insane-flash requires NVIDIA CUDA hardware and nvidia-smi. " f"No controlled FlashAttention wheel selected for {tuple_text}.")
    matches = [
        wheel
        for wheel in FLASH_ATTN_WHEELS
        if wheel.system == normalized_system and wheel.machine == normalized_machine and wheel.python_tag == selected_python_tag and wheel.torch_minor == torch_minor and wheel.cuda_tag == cuda_tag
    ]
    primary_matches = [wheel for wheel in matches if wheel.primary]
    if primary_matches:
        return primary_matches[0]
    if matches:
        return matches[0]
    raise UnsupportedFlashAttentionWheel(
        "No pinned FlashAttention wheel is available for insane-flash. "
        f"Tuple: {tuple_text}. Supported tuples are Windows x86_64, Linux x86_64, "
        "and Linux aarch64 on Python cp311 with torch2.7/cu128."
    )
