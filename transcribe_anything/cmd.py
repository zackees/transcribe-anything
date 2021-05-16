"""
    Entry point for running the transcribe-anything prgram.
"""

import argparse
import sys

from transcribe_anything.api import transcribe


def main() -> None:
    """Main entry point for the command line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url_or_file",
        help="Provide file path or url (includes youtube/facebook/twitter/etc)",
    )
    parser.add_argument(
        "--out",
        help="Output text file name",
        default=None,
    )
    args = parser.parse_args()
    out = transcribe(url_or_file=args.url_or_file)
    if args.out:
        with open(args.out, "wt") as fd:
            fd.write(out)
    else:
        sys.stdout.write(f"{out}\n")


if __name__ == "__main__":
    main()
    sys.exit(0)
