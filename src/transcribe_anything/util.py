"""
Determines whether this device is cpu or gpu.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

PROCESS_TIMEOUT = 4 * 60 * 60

# Cache file for NVIDIA detection to ensure consistency across runs
_NVIDIA_CACHE_FILE = Path.home() / ".transcribe_anything_nvidia_cache.json"
_NVIDIA_DETECTION_CACHE = None


def is_mac_arm() -> bool:
    """Returns true if mac arm like m1, m2, etc."""
    if platform.system() != "Darwin":
        return False  # Not a Mac
    else:

        # Using uname to get the machine hardware name can indicate the architecture
        machine = os.uname().machine  # type: ignore[attr-defined]

        # ARM architectures can be 'arm64' or 'aarch64' depending on the platform
        return machine in ["arm64", "aarch64"]


def is_mac() -> bool:
    """Returns True if the OS is macOS."""
    return platform.system() == "Darwin"


def sanitize_filename(string: str) -> str:
    """
    Sanitize a string to be used as a filename.

    If minimal_change is set to true, then we only strip the bare minimum of
    characters that are problematic for filesystems (namely, ':', '/' and '\x00', '\n').
    """
    string = unescape(string)
    string = unquote(string)
    string = re.sub(r"<(?P<tag>.+?)>(?P<in>.+?)<(/(?P=tag))>", r"\g<in>", string)
    string = string.replace(":", "_").replace("/", "_").replace("\x00", "_")
    string = re.sub(r'[\n\\\*><?"|\t]', "", string)
    string = string.strip()
    while string.endswith("_"):
        string = string[:-1]
    while string.startswith("_"):
        string = string[1:]
    return string


def chop_double_extension(path_name) -> str:
    """takes in a path like out.mp3.txt and returns out.mp3"""
    # Split the path name on "."
    parts = path_name.split(".")
    ext = parts[-1]
    # If there are fewer than two parts, return the original path name
    while len(parts) > 1:
        parts = parts[:-1]
    # Otherwise, return the second-to-last part followed by the last part
    return ".".join(parts + [ext])


def _get_system_fingerprint() -> str:
    """Get a fingerprint of the system to detect hardware changes."""
    # Include platform info and check for nvidia-smi existence
    platform_info = f"{platform.system()}-{platform.machine()}-{platform.version()}"
    nvidia_smi_exists = shutil.which("nvidia-smi") is not None
    return f"{platform_info}-nvidia_smi:{nvidia_smi_exists}"


def _load_nvidia_cache() -> dict:
    """Load the NVIDIA detection cache from disk."""
    try:
        if _NVIDIA_CACHE_FILE.exists():
            with open(_NVIDIA_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load NVIDIA cache: {e}", file=sys.stderr)
    return {}


def _save_nvidia_cache(cache_data: dict) -> None:
    """Save the NVIDIA detection cache to disk."""
    try:
        with open(_NVIDIA_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except OSError as e:
        print(f"Warning: Failed to save NVIDIA cache: {e}", file=sys.stderr)


def has_nvidia_smi() -> bool:
    """
    Returns True if nvidia-smi is installed.

    This function caches the result based on system fingerprint to ensure
    consistency across runs and avoid triggering unnecessary reinstalls
    in uv-iso-env environments.
    """
    global _NVIDIA_DETECTION_CACHE

    # Get current system fingerprint
    current_fingerprint = _get_system_fingerprint()

    # Load cache if not already loaded
    if _NVIDIA_DETECTION_CACHE is None:
        _NVIDIA_DETECTION_CACHE = _load_nvidia_cache()

    # Check if we have a cached result for this system fingerprint
    if current_fingerprint in _NVIDIA_DETECTION_CACHE:
        cached_result = _NVIDIA_DETECTION_CACHE[current_fingerprint]
        # print(f"Debug: Using cached NVIDIA detection result: {cached_result} for fingerprint: {current_fingerprint}", file=sys.stderr)
        return cached_result

    # Perform actual detection
    nvidia_available = shutil.which("nvidia-smi") is not None

    # Cache the result
    _NVIDIA_DETECTION_CACHE[current_fingerprint] = nvidia_available
    _save_nvidia_cache(_NVIDIA_DETECTION_CACHE)

    print(f"Debug: Detected NVIDIA availability: {nvidia_available} for fingerprint: {current_fingerprint}", file=sys.stderr)
    return nvidia_available


def clear_nvidia_cache() -> None:
    """Clear the NVIDIA detection cache. Useful for testing or when hardware changes."""
    global _NVIDIA_DETECTION_CACHE
    _NVIDIA_DETECTION_CACHE = None
    try:
        if _NVIDIA_CACHE_FILE.exists():
            _NVIDIA_CACHE_FILE.unlink()
            print("NVIDIA detection cache cleared.", file=sys.stderr)
    except OSError as e:
        print(f"Warning: Failed to clear NVIDIA cache: {e}", file=sys.stderr)


@dataclass
class NvidiaDriverInfo:
    """Information parsed from nvidia-smi output."""

    driver_version: str
    cuda_version: str  # Max CUDA version supported by the driver


_NVIDIA_DRIVER_INFO_CACHE: Optional[NvidiaDriverInfo] = None
_NVIDIA_DRIVER_INFO_CHECKED: bool = False


def get_nvidia_driver_info() -> Optional[NvidiaDriverInfo]:
    """
    Parse nvidia-smi output to get driver version and supported CUDA version.
    Returns None if nvidia-smi is not available or output cannot be parsed.
    Results are cached in memory.
    """
    global _NVIDIA_DRIVER_INFO_CACHE, _NVIDIA_DRIVER_INFO_CHECKED
    if _NVIDIA_DRIVER_INFO_CHECKED:
        return _NVIDIA_DRIVER_INFO_CACHE

    _NVIDIA_DRIVER_INFO_CHECKED = True

    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        output = result.stdout

        # Parse driver version: "Driver Version: 560.35.03"
        driver_match = re.search(r"Driver Version:\s+([\d.]+)", output)
        # Parse CUDA version: "CUDA Version: 13.0"
        cuda_match = re.search(r"CUDA Version:\s+([\d.]+)", output)

        if driver_match and cuda_match:
            _NVIDIA_DRIVER_INFO_CACHE = NvidiaDriverInfo(
                driver_version=driver_match.group(1),
                cuda_version=cuda_match.group(1),
            )
            return _NVIDIA_DRIVER_INFO_CACHE
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass

    return None


def print_cuda_diagnostics(expected_cuda: str = "12.8") -> None:
    """Print CUDA diagnostic information to stderr for troubleshooting."""
    info = get_nvidia_driver_info()
    if info is None:
        print(
            "CUDA Diagnostics: Could not query nvidia-smi. " "Ensure NVIDIA drivers are installed and nvidia-smi is on PATH.",
            file=sys.stderr,
        )
        return

    print("CUDA Diagnostics:", file=sys.stderr)
    print(f"  NVIDIA Driver Version: {info.driver_version}", file=sys.stderr)
    print(f"  Driver CUDA Version:   {info.cuda_version} (max supported by driver)", file=sys.stderr)
    print(f"  PyTorch CUDA Version:  {expected_cuda} (bundled with PyTorch wheels)", file=sys.stderr)

    driver_major = int(info.cuda_version.split(".")[0])
    expected_major = int(expected_cuda.split(".")[0])

    if driver_major > expected_major:
        print(
            f"  NOTE: Your driver supports CUDA {info.cuda_version} but PyTorch uses CUDA {expected_cuda} wheels.",
            file=sys.stderr,
        )
        print(
            "  This should be backward-compatible, but if system CUDA libraries are on",
            file=sys.stderr,
        )
        print(
            "  LD_LIBRARY_PATH they may override PyTorch's bundled libraries and cause errors.",
            file=sys.stderr,
        )
        print(
            "  Try: unset LD_LIBRARY_PATH (or remove CUDA paths from it) and retry.",
            file=sys.stderr,
        )
    elif driver_major < expected_major:
        print(
            f"  ERROR: Driver CUDA {info.cuda_version} is older than PyTorch's CUDA {expected_cuda}.",
            file=sys.stderr,
        )
        print("  Update your NVIDIA driver to one that supports CUDA 12.8+.", file=sys.stderr)

    print("  If hardware changed recently, try: transcribe-anything --clear-nvidia-cache", file=sys.stderr)
