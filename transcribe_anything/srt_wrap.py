# pylint: skip-file

"""
Wraps the srt file.
"""

# designed to be run in an isolated environment.

import shutil
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

from srtranslator import SrtFile  # type: ignore


# srtranslator==0.2.6
def srt_wrap(srt_file: Path) -> str:
    """Wrap lines in a srt file."""
    with tempfile.TemporaryDirectory() as tempdir:
        out_file = Path(tempdir) / "out.srt"
        shutil.copy(srt_file, out_file)
        srt = SrtFile(str(srt_file))
        srt.wrap_lines()
        srt.save(str(out_file))
        string = out_file.read_text(encoding="utf-8")
        return string


def create_args() -> Namespace:
    """Create args."""
    parser = ArgumentParser(description="Wrap lines in a srt file.")
    parser.add_argument("input_srt", help="The srt file to wrap.")
    args = parser.parse_args()
    return args


def main() -> int:
    """Main entry point for the command line tool."""
    args = create_args()
    out = srt_wrap(args.input_srt)
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
