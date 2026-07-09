"""
Utilities for srt translation, including srt wrap.
"""

import subprocess
import warnings
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, Requirements  # type: ignore

from transcribe_anything.util import get_runtime_venv_dir

# from isolated_environment import isolated_environment  # type: ignore


HERE = Path(__file__).parent
WRAP_SRT_PY = HERE / "srt_wrap.py"


def get_environment() -> IsoEnv:
    """Returns the environment."""
    venv_path = get_runtime_venv_dir("srttranslator")
    # srtranslator pins selenium==4.7.2, which caps urllib3 to 1.26.x; keep
    # the patched floors within that cap (PYSEC-2023-192/212, PYSEC-2026-1995,
    # requests PYSEC-2023-74/2026-187x, CVE-2026-25645). Only SrtFile parsing
    # is used here — the selenium translation machinery is never launched.
    reqs_text = "\n".join(["srtranslator==0.3.9", "requests>=2.33.0", "urllib3>=1.26.19"])
    reqs = Requirements(
        reqs_text,
        python_version="==3.11.*",
    )
    args = IsoEnvArgs(
        venv_path=venv_path,
        build_info=reqs,
    )
    env = IsoEnv(args)
    return env


def srt_wrap_to_string(srt_file: Path) -> str:
    """Wrap lines in a srt file."""
    env = get_environment()
    cmd_list = [
        str(WRAP_SRT_PY),
        str(srt_file),
    ]
    try:
        cp: subprocess.CompletedProcess = env.run(
            cmd_list,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr
        error_msg = f"Failed to run srt_wrap: {exc}"
        error_msg += f"\n{stderr}"
        warnings.warn(error_msg)
        raise
    out = cp.stdout
    return out


def srt_wrap(srt_file: Path) -> None:
    """Wrap lines in a srt file."""
    try:
        assert WRAP_SRT_PY.exists()
        out = srt_wrap_to_string(srt_file)
        srt_file.write_text(out, encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        warnings.warn(f"Failed to run srt_wrap: {exc}")
