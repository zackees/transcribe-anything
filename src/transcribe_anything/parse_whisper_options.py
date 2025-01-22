"""
Parses whisper options.
"""

import re
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
    """Parses the whisper options."""
    env = get_environment()
    stdout = env.run(
        ["whisper", "--help"],
        shell=False,
        universal_newlines=True,
        encoding="utf-8",
        check=True,
        capture_output=True,
    ).stdout
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
