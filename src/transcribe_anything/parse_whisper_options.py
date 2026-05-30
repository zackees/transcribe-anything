"""
Parses whisper options.
"""

import re
import subprocess
from typing import Any

from transcribe_anything import logger
from transcribe_anything.whisper import get_environment

PATTERN = r"\s+\[--(.*?)\]"


def _parse_item(item: str) -> tuple[str, Any]:
    """Parses item into a key,value pair."""
    value: Any = None
    key, value = item.split(" ", 1)
    if "{" in value:
        value = value.replace("{", "").replace("}", "").split(",")
        value = [v.strip() for v in value if v.strip()]
    return (key, value)


def parse_whisper_options() -> dict:
    """Parses the whisper options.

    Notes on stream handling (issues #40 and #52):

    - stderr is intentionally NOT redirected so uv's first-run download/build
      progress flows to the terminal; capturing it makes the call look frozen.
    - stdout is captured because we parse it.
    - On non-zero exit we raise RuntimeError with an actionable message and
      the captured stderr (when available) instead of a bare CalledProcessError
      that hides the underlying cause.
    """
    env = get_environment()
    result = env.run(
        ["whisper", "--help"],
        shell=False,
        universal_newlines=True,
        encoding="utf-8",
        check=False,
        stdout=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()
        suffix = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
        raise RuntimeError(
            f"`whisper --help` failed with exit {result.returncode}. "
            "This usually means the whisper environment failed to install or "
            "the bundled whisper binary cannot run on this platform."
            f"{suffix}"
        )
    stdout = result.stdout or ""
    lines = stdout.splitlines()
    data = {}
    for line in lines:
        items = re.findall(PATTERN, line)
        if not items:
            continue
        for item in items:
            try:
                key, value = _parse_item(item)
                data[key] = value
            except Exception as exc:  # pylint: disable=broad-except
                logger.log_error(f"Failed to parse {items}: {exc}")
                continue
    return data


if __name__ == "__main__":
    from pprint import pprint

    pprint(parse_whisper_options())
