"""
Main entry point.
"""

import logging
import os
import sys

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
    env = get_environment(has_nvidia=True)
    if sys.platform == "win32":
        env.run(["python", "-c", "import os; print(os.getcwd())"])
    else:
        env.run(["pwd"])
    print("Installing static ffmpeg...")
    os.system("static_ffmpeg -version")
    return 0


if __name__ == "__main__":

    main()
