# pylint: skip-file
"""
Runs SenseVoice (FunASR) inside the isolated env, writes out.{json,srt,vtt,txt}
and optional speaker.json to --output-dir.

Invoked by transcribe_anything.sensevoice.run_sensevoice via iso_env.run().
The host process never imports funasr directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _fmt_ts(ms: float, vtt: bool = False) -> str:
    """Format milliseconds as SRT/VTT timestamp."""
    ms = max(int(round(ms)), 0)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms_part = divmod(rem, 1000)
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms_part:03d}"


def _build_segments(result: dict[str, Any], plain_text: str) -> list[dict[str, Any]]:
    """Return a list of {start, end, text, spk?} dicts in milliseconds."""
    sentence_info = result.get("sentence_info")
    if isinstance(sentence_info, list) and sentence_info:
        segments: list[dict[str, Any]] = []
        for s in sentence_info:
            if not isinstance(s, dict):
                continue
            seg: dict[str, Any] = {
                "start": float(s.get("start", 0)),
                "end": float(s.get("end", 0)),
                "text": str(s.get("text", "")).strip(),
            }
            if "spk" in s and s["spk"] is not None:
                seg["spk"] = s["spk"]
            segments.append(seg)
        return segments
    return [{"start": 0.0, "end": 0.0, "text": plain_text}]


def _write_srt_vtt(segments: list[dict[str, Any]], srt_path: Path, vtt_path: Path, diarize: bool) -> None:
    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]
    for idx, seg in enumerate(segments, 1):
        text = seg["text"]
        spk = seg.get("spk") if diarize else None
        prefix = f"Speaker {spk}: " if spk is not None else ""
        body = f"{prefix}{text}".strip()
        start = _fmt_ts(seg["start"])
        end = _fmt_ts(seg["end"])
        srt_lines.append(f"{idx}\n{start} --> {end}\n{body}\n")
        vtt_lines.append(f"{_fmt_ts(seg['start'], vtt=True)} --> {_fmt_ts(seg['end'], vtt=True)}\n{body}\n")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="SenseVoice transcription runner.")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 | cpu | mps")
    parser.add_argument("--language", default="auto", help="auto|zh|en|yue|ja|ko|nospeech")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--hub", default="ms", choices=["ms", "hf"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--max-segment-ms", type=int, default=30000)
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--merge-length-s", type=int, default=15)
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    # FunASR + ModelScope are imported inside main() so that --help works
    # without pulling them in.
    from funasr import AutoModel  # type: ignore
    from funasr.utils.postprocess_utils import (
        rich_transcription_postprocess,  # type: ignore
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "model": "iic/SenseVoiceSmall" if args.hub == "ms" else "FunAudioLLM/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": args.max_segment_ms},
        "device": args.device,
        "hub": args.hub,
        "trust_remote_code": False,
        "disable_update": True,
    }
    if args.diarize:
        # Both spk_model and punc_model are required to populate
        # sentence_info with per-speaker segments.
        kwargs["spk_model"] = "cam++"
        kwargs["punc_model"] = "ct-punc"

    model = AutoModel(**kwargs)
    result_list = model.generate(
        input=args.input_wav,
        cache={},
        language=args.language,
        use_itn=True,
        batch_size_s=args.batch_size_s,
        merge_vad=True,
        merge_length_s=args.merge_length_s,
    )
    if not result_list:
        sys.stderr.write("SenseVoice produced no result\n")
        return 1
    result = result_list[0]

    raw_text = result.get("text", "")
    plain_text = rich_transcription_postprocess(raw_text).strip()
    segments = _build_segments(result, plain_text)
    for seg in segments:
        seg["text"] = rich_transcription_postprocess(seg.get("text", "")).strip()

    (out_dir / "out.json").write_text(
        json.dumps({"text": plain_text, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "out.txt").write_text(plain_text + "\n", encoding="utf-8")
    _write_srt_vtt(segments, out_dir / "out.srt", out_dir / "out.vtt", diarize=args.diarize)

    if args.diarize:
        speaker_payload = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "spk": seg.get("spk"),
                "text": seg["text"],
            }
            for seg in segments
            if seg.get("spk") is not None
        ]
        if speaker_payload:
            (out_dir / "speaker.json").write_text(
                json.dumps(speaker_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
