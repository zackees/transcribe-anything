"""
Utilities for srt translation, including srt wrap.
"""

import subprocess
import warnings
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, Requirements  # type: ignore

# from isolated_environment import isolated_environment  # type: ignore


HERE = Path(__file__).parent
WRAP_SRT_PY = HERE / "srt_wrap.py"


def get_environment() -> IsoEnv:
    """Returns the environment."""
    venv_path = HERE / "venv" / "srttranslator"
    reqs_text = "\n".join(["srtranslator==0.3.9", "requests==2.28.1", "urllib3==1.26.13"])
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
