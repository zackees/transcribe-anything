"""
Determines whether this device is cpu or gpu.
"""

import torch  # pylint: disable=import-outside-toplevel


def get_computing_device() -> str:
    """Get the computing device."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"
