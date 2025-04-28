"""
Main entry point.
"""

import logging
import os

from transcribe_anything.insanley_fast_whisper_reqs import get_environment

# for resource loading


logger = logging.getLogger("transcribe_anything")
logger.setLevel(logging.INFO)

# _MODEL = os.environ.get("TRANSCRIBE_ANYTHING_MODEL", "large-v3")
# _DEVICE = os.environ.get("TRANSCRIBE_ANYTHING_DEVICE", "insane")
# _BATCH_SIZE = os.environ.get("TRANSCRIBE_ANYTHING_BATCH_SIZE", 8)


def main() -> int:
    """Main entry point for the transcribe_anything package."""
    # locate the bundled sample.mp3
    print("Installing transcribe_anything (device insane) environment...")
    get_environment()
    print("Installing static ffmpeg...")
    os.system("ffmpeg_static -version")
    return 0


if __name__ == "__main__":

    main()
