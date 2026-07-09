"""
Drop-in replacement for insanely-fast-whisper CLI that supports XPU device.
"""

import argparse
import json

import torch
from transformers import pipeline

# XPU doesn't support float64.  Alias float64 → float32 globally for
# the entire subprocess so every `.to(torch.float64)` in Whisper's
# generation code (including the four `device.type == "mps"` checks in
# generation_whisper.py) silently becomes float32.  Safe for inference:
# we never need double precision for timestamp arithmetic.
torch.float64 = torch.float32
torch.double = torch.float32

parser = argparse.ArgumentParser(description="ASR with XPU support")
parser.add_argument("--file-name", required=True, type=str)
parser.add_argument("--device-id", required=False, default="xpu", type=str)
parser.add_argument("--transcript-path", required=False, default="output.json", type=str)
parser.add_argument("--model-name", required=False, default="openai/whisper-large-v3", type=str)
parser.add_argument("--task", required=False, default="transcribe", type=str, choices=["transcribe", "translate"])
parser.add_argument("--language", required=False, default="None", type=str)


def _str_to_bool(value: str) -> bool:
    """argparse type=bool treats any non-empty string ("False") as truthy."""
    return value.strip().lower() in ("1", "true", "yes", "on")


parser.add_argument("--batch-size", required=False, default=24, type=int)
# Accepted for CLI parity with insanely-fast-whisper (both bare "--flash" and
# "--flash True/False"); flash attention is not used on XPU.
parser.add_argument("--flash", required=False, default=False, type=_str_to_bool, nargs="?", const=True)
parser.add_argument("--timestamp", required=False, default="chunk", type=str, choices=["chunk", "word"])
parser.add_argument("--hf-token", required=False, default="no_token", type=str)
parser.add_argument("--diarization_model", required=False, default="pyannote/speaker-diarization-3.1", type=str)
parser.add_argument("--num-speakers", required=False, default=None, type=int)
parser.add_argument("--min-speakers", required=False, default=None, type=int)
parser.add_argument("--max-speakers", required=False, default=None, type=int)


def main():
    args = parser.parse_args()
    device = args.device_id
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    language = None if args.language == "None" else args.language
    generate_kwargs = {"task": args.task, "language": language}
    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")
    outputs = pipe(
        args.file_name,
        chunk_length_s=30,
        batch_size=args.batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )
    # Pipeline returns {"text": …, "chunks": […]}. Pass everything through.
    with open(args.transcript_path, "w", encoding="utf8") as fp:
        json.dump(outputs, fp, ensure_ascii=False)


if __name__ == "__main__":
    main()
