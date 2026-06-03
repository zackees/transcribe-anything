"""Initialize the isolated insane-flash backend environment."""

import logging
import os
import sys

from transcribe_anything.insanley_fast_whisper_reqs import get_environment

logger = logging.getLogger("transcribe_anything")
logger.setLevel(logging.INFO)


def main() -> int:
    """Main entry point for pre-installing the insane-flash backend."""
    print("Installing transcribe_anything (device insane-flash) environment...")
    env = get_environment(has_nvidia=True, flash=True)
    if sys.platform == "win32":
        env.run(["python", "-c", "import os; print(os.getcwd())"])
    else:
        env.run(["pwd"])
    print("Installing static ffmpeg...")
    os.system("static_ffmpeg -version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
