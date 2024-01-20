"""
Determines whether this device is cpu or gpu.
"""

import os
import platform
import re
from html import unescape
from urllib.parse import unquote

PROCESS_TIMEOUT = 4 * 60 * 60


def is_mac_arm():
    """Returns true if mac arm like m1, m2, etc."""
    if platform.system() != "Darwin":
        return False  # Not a Mac

    # Using uname to get the machine hardware name can indicate the architecture
    machine = os.uname().machine  # pylint: disable=no-member

    # ARM architectures can be 'arm64' or 'aarch64' depending on the platform
    return machine in ["arm64", "aarch64"]


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
