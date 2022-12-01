"""
    Entry point for running the transcribe-anything prgram.
"""

import argparse
import sys

from transcribe_anything.api import transcribe
from transcribe_anything.util import get_computing_device


def main() -> None:
    """Main entry point for the command line tool."""
    device = get_computing_device()
    help_str = f'transcribe_anything is using a "{device}" device'
    parser = argparse.ArgumentParser(description=help_str)
    parser.add_argument(
        "url_or_file",
        help="Provide file path or url (includes youtube/facebook/twitter/etc)",
    )
    parser.add_argument(
        "--output_dirname",
        help="Provide output directory name",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Provide model name",
        default="small",
    )
    args = parser.parse_args()
    transcribe(url_or_file=args.url_or_file, output_dirname=args.output_dirname, model=args.model)


if __name__ == "__main__":
    main()
    sys.exit(0)
