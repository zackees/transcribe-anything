"""
Utilities for srt translation, including srt wrap.
"""

import subprocess
import warnings
from pathlib import Path
from typing import Any

from isolated_environment import isolated_environment  # type: ignore

HERE = Path(__file__).parent
WRAP_SRT_PY = HERE / "srt_wrap.py"


def get_environment() -> dict[str, Any]:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "srttranslator"
    env = isolated_environment(
        venv_dir, ["srtranslator==0.2.6", "requests==2.28.1", "urllib3==1.26.13"]
    )
    return env


def srt_wrap_to_string(srt_file: Path) -> str:
    """Wrap lines in a srt file."""
    env = get_environment()
    process = subprocess.run(
        ["python", str(WRAP_SRT_PY), str(srt_file)],
        env=env,
        capture_output=True,
        text=True,
        shell=False,
        check=True,
    )
    out = process.stdout
    return out


def srt_wrap(srt_file: Path) -> None:
    """Wrap lines in a srt file."""
    try:
        assert WRAP_SRT_PY.exists()
        out = srt_wrap_to_string(srt_file)
        srt_file.write_text(out, encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        warnings.warn(f"Failed to run srt_wrap: {exc}")
        return
