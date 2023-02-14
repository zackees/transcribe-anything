"""
    Entry point for running the transcribe-anything prgram.
"""

import argparse
import sys

from transcribe_anything.api import transcribe
from transcribe_anything.util import get_computing_device
from transcribe_anything.parse_whisper_options import parse_whisper_options


def main() -> None:
    """Main entry point for the command line tool."""
    whisper_options = parse_whisper_options()
    device = get_computing_device()
    help_str = (
        f'transcribe_anything is using a "{device}" device.'
        " Any unrecognized args are assumed to be for whisper"
        " ai and will be passed as is to whisper ai."
    )
    parser = argparse.ArgumentParser(description=help_str)
    parser.add_argument(
        "url_or_file",
        help="Provide file path or url (includes youtube/facebook/twitter/etc)",
    )
    parser.add_argument(
        "--output_dir",
        help="Provide output directory name,d efaults to the filename of the file.",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="name of the Whisper model to us",
        default="small",
        choices=whisper_options["model"],
    )
    parser.add_argument(
        "--task",
        help="whether to perform transcription or translation",
        default="transcribe",
        choices=whisper_options["task"],
    )
    parser.add_argument(
        "--language",
        help="language to the target audio is in, default None will auto-detect",
        default=None,
        choices=[None] + whisper_options["language"],
    )
    # keep_audio
    parser.add_argument(
        "--no-keep-audio",
        help="whether to keep the audio file after processing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--device",
        help="device to use for processing, None will auto select CUDA if available or else CPU",
        default=None,
        choices=[None, "cpu", "cuda"],
    )
    # add extra options that are passed into the transcribe function
    args, unknown = parser.parse_known_args()
    print(f"Unknown args: {unknown}")
    print(f"Running transcribe_audio on {args.url_or_file}")
    transcribe(
        url_or_file=args.url_or_file,
        output_dir=args.output_dir,
        model=args.model if args.model != "None" else None,
        task=args.task,
        language=args.language if args.language != "None" else None,
        keep_audio=not args.no_keep_audio,
        device=args.device,
        other_args=unknown,
    )


if __name__ == "__main__":
    # push sys argv prior to call
    sys.argv.append("test.mp3")
    sys.argv.append('--initial_prompt "What is your name?"')
    main()
    sys.exit(0)
