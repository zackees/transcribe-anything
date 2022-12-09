"""
Determines whether this device is cpu or gpu.
"""

import re
from html import unescape
from urllib.parse import unquote
import torch  # pylint: disable=import-outside-toplevel


def get_computing_device() -> str:
    """Get the computing device."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def sanitize_path(string: str) -> str:
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
