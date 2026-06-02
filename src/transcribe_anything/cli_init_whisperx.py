"""
Main entry point for initializing the WhisperX environment.
"""

import logging
import os
import sys

logger = logging.getLogger("transcribe_anything")
logger.setLevel(logging.INFO)


def main() -> int:
    """Main entry point for the transcribe_anything WhisperX environment initializer."""
    from transcribe_anything.whisperx_reqs import get_environment

    print("Installing transcribe_anything (device whisperx) environment...")
    env = get_environment()
    if sys.platform == "win32":
        env.run(["python", "-c", "import os; print(os.getcwd())"])
    else:
        env.run(["pwd"])
    print("Installing static ffmpeg...")
    os.system("static_ffmpeg -version")
    return 0


if __name__ == "__main__":
    main()
