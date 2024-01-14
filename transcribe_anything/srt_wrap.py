"""
Wraps the srt file.
"""

# designed to be run in an isolated environment.

from argparse import ArgumentParser, Namespace
import sys
from srtranslator import SrtFile  # type: ignore


# srtranslator==0.2.6
def srt_wrap(srt_file: str, out_file: str) -> None:
    """Wrap lines in a srt file."""
    srt = SrtFile(srt_file)
    srt.wrap_lines()
    srt.save(out_file)


def create_args() -> Namespace:
    """Create args."""
    parser = ArgumentParser(description="Wrap lines in a srt file.")
    parser.add_argument("src_srt_file", help="The srt file to wrap.")
    parser.add_argument("dst_srt_file", help="The output file.", nargs="?")
    args = parser.parse_args()
    return args


def main() -> int:
    """Main entry point for the command line tool."""
    args = create_args()
    srt_wrap(args.src_srt_file, args.dst_srt_file or args.src_srt_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
