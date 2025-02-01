"""
Entry point for running the transcribe-anything prgram.
"""

# flake8: noqa E501
# pylint: disable=too-many-branches,import-outside-toplevel

import argparse
import json
import os
import platform
import sys
import traceback
from pathlib import Path

# appdirs is used to get the cache directory
from appdirs import user_cache_dir  # type: ignore

from transcribe_anything.parse_whisper_options import parse_whisper_options
from transcribe_anything.whisper import get_computing_device

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
WHISPER_OPTIONS = HERE / "WHISPER_OPTIONS.json"

os.environ["PYTHONIOENCODING"] = "utf-8"

WHISPER_MODEL_OPTIONS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-legacy",
    "large",
    "large-v2",
    "large-v3",
    "distil-whisper/distil-large-v2",
]


def get_whisper_options() -> dict:
    """Get whisper options.""" ""
    if WHISPER_OPTIONS.exists():
        whisper_options = parse_whisper_options()
        string = json.dumps(whisper_options, indent=4)
        WHISPER_OPTIONS.write_text(string)
        return whisper_options
    file_age = os.path.getmtime(WHISPER_OPTIONS)
    if file_age > 60 * 60 * 24 * 7:  # 1 week
        whisper_options = parse_whisper_options()
        string = json.dumps(whisper_options, indent=4)
        WHISPER_OPTIONS.write_text(string)
    return whisper_options


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    whisper_options = get_whisper_options()
    device = get_computing_device()
    help_str = f'transcribe_anything is using a "{device}" device.' " Any unrecognized args are assumed to be for whisper" " ai and will be passed as is to whisper ai."
    parser = argparse.ArgumentParser(description=help_str)
    parser.add_argument(
        "url_or_file",
        help="Provide file path or url (includes youtube/facebook/twitter/etc)",
        nargs="?",
    )
    parser.add_argument(
        "--query-gpu-json-path",
        help=("Query the GPU and store it in the given path," " warning takes a long time on first load!"),
        type=Path,
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
        choices=WHISPER_MODEL_OPTIONS,
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
    choices = [None, "cpu", "cuda", "insane"]
    if platform.system() == "Darwin":
        choices.append("mps")
    parser.add_argument(
        "--device",
        help="device to use for processing, None will auto select CUDA if available or else CPU",
        default=None,
        choices=choices,
    )
    parser.add_argument(
        "--hf_token",
        help="huggingface token to use for downloading models",
        default=None,
    )
    parser.add_argument(
        "--save_hf_token",
        help="save huggingface token to a file for future use",
        action="store_true",
    )
    parser.add_argument(
        "--diarization_model",
        help=("Name of the pretrained model/ checkpoint to perform diarization." + " (default: pyannote/speaker-diarization). Only works for --device insane."),
        default="pyannote/speaker-diarization-3.1",
    )
    parser.add_argument(
        "--timestamp",
        help=("Whisper supports both chunked as well as word level timestamps. (default: chunk)." + " Only works for --device insane."),
        choices=["chunk", "word"],
        default=None,
    )
    parser.add_argument(
        "--embed",
        help="whether to embed the translation file into the output file",
        action="store_true",
    )
    # add extra options that are passed into the transcribe function
    args, unknown = parser.parse_known_args()
    if args.url_or_file is None and args.query_gpu_json_path is None:
        print("No file or url provided")
        parser.print_help()
        sys.exit(1)
    args.unknown = unknown
    return args


def main() -> int:
    """Main entry point for the command line tool."""
    args = parse_arguments()
    unknown = args.unknown
    if args.query_gpu_json_path is not None:
        from transcribe_anything.insanely_fast_whisper import get_cuda_info

        json_str = get_cuda_info().to_json_str()
        path: Path = args.query_gpu_json_path
        path.write_text(json_str, encoding="utf-8")
        return 0
    if args.model == "large-legacy":
        args.model = "large"
    elif args.model == "large":
        print("Defaulting to large-v3 model for --model large," + " use --model large-legacy for the old model")
        args.model = "large-v3"
    elif args.model is None and args.device == "insane":
        print("Defaulting to large-v3 model for --device insane")
        args.model = "large-v3"

    hf_token_path = Path(user_cache_dir(), "hf_token.txt")
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN", None)
        if args.hf_token is None and hf_token_path.exists():
            # read from file
            args.hf_token = hf_token_path.read_text(encoding="utf-8").strip() or None
        if args.hf_token is None:
            args.diarization_model = None
    if args.save_hf_token:
        hf_token_path.write_text(args.hf_token or "", encoding="utf-8")
        print("Saved huggingface token to", hf_token_path)

    # For now, just stuff --diarization_model and --timestamp into unknown
    if args.diarization_model:
        if args.device != "insane":
            print("--diarization_model only works with --device insane. Ignoring --diarization_model")
        else:
            unknown.append("--diarization_model")
            unknown.append(args.diarization_model)

    if args.timestamp:
        if args.device != "insane":
            print("--timestamp only works with --device insane. Ignoring --timestamp")
        else:
            # unknown.append(f"--timestamp {args.timestamp}")
            unknown.append("--timestamp")
            unknown.append(args.timestamp)

    if unknown:
        print(f"Args passed to whisper backend: {unknown}")
    print(f"Running transcribe_audio on {args.url_or_file}")
    try:
        from transcribe_anything.api import transcribe

        transcribe(
            url_or_file=args.url_or_file,
            output_dir=args.output_dir,
            model=args.model if args.model != "None" else None,
            task=args.task,
            language=args.language if args.language != "None" else None,
            device=args.device,
            embed=args.embed,
            hugging_face_token=args.hf_token,
            other_args=unknown,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        stack = traceback.format_exc()
        sys.stderr.write(f"Error: {e}\n{stack}\nwhile processing {args.url_or_file}\n")
        return 1
    return 0


if __name__ == "__main__":
    # push sys argv prior to call
    here = Path(os.path.abspath(os.path.dirname(__file__)))
    project_root = here.parent.parent
    localfile_dir = project_root / "tests" / "localfile"
    os.chdir(localfile_dir)
    # sys.argv.append("test.wav")
    # sys.argv.append("--model")
    # sys.argv.append("large")
    # sys.argv.append('--initial_prompt "What is your name?"')
    sys.argv.append("video.mp4")
    sys.argv.append("--language")
    sys.argv.append("en")
    sys.argv.append("--model")
    sys.argv.append("tiny")
    sys.exit(main())
