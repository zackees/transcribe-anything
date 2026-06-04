# pylint: skip-file
"""
WhisperX forced-alignment post-processor for the insane / insane-flash backends.

Runs inside the WhisperX iso-env. Reads the insane backend's ``out.json``
(HF pipeline shape with ``chunks: [{timestamp:[s,e], text}, ...]``), runs
``whisperx.load_align_model`` + ``whisperx.align`` against the source audio,
then writes back an enriched JSON where each chunk gains a ``words`` list
of ``{word, start, end, score}`` and its top-level ``timestamp`` is tightened
to the first/last aligned word boundary.

When the language is unsupported by WhisperX's default-models dict, the
script writes the input JSON back unchanged and exits 0 with a stderr
warning — so the caller can always treat alignment as best-effort.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _convert_chunks_to_segments(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Map insane ``chunks`` to whisperx ``segments``.

    Returns ``(segments, source_index_map)`` where ``segments[i]`` is the
    payload we hand to ``whisperx.align`` and ``source_index_map[i]`` is
    the index back into the original ``chunks`` list so we can write the
    word-level results back to the right chunk.

    Chunks with a ``None`` start or end timestamp are skipped (insanely-
    fast-whisper occasionally emits these at the end of a recording).
    """
    segments: list[dict[str, Any]] = []
    index_map: list[int] = []
    for i, chunk in enumerate(chunks):
        ts = chunk.get("timestamp") or [None, None]
        start = ts[0] if len(ts) > 0 else None
        end = ts[1] if len(ts) > 1 else None
        text = str(chunk.get("text", "")).strip()
        if start is None or end is None or not text:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        segments.append({"start": start_f, "end": end_f, "text": text})
        index_map.append(i)
    return segments, index_map


def _supported_language(language_code: str) -> bool:
    """True if WhisperX has a default alignment model for the language."""
    if not language_code:
        return False
    try:
        from whisperx.alignment import (  # type: ignore
            DEFAULT_ALIGN_MODELS_HF,
            DEFAULT_ALIGN_MODELS_TORCH,
        )
    except ImportError:
        return False
    return language_code in DEFAULT_ALIGN_MODELS_TORCH or language_code in DEFAULT_ALIGN_MODELS_HF


def _merge_alignment_into_chunks(
    chunks: list[dict[str, Any]],
    aligned_segments: list[dict[str, Any]],
    index_map: list[int],
) -> None:
    """Mutate ``chunks`` in place with per-word data + tightened timestamps."""
    for aligned, src_idx in zip(aligned_segments, index_map):
        if src_idx >= len(chunks):
            continue
        words = aligned.get("words") or []
        if not isinstance(words, list):
            continue
        normalized_words: list[dict[str, Any]] = []
        first_start: float | None = None
        last_end: float | None = None
        for w in words:
            if not isinstance(w, dict):
                continue
            w_start = w.get("start")
            w_end = w.get("end")
            if w_start is None or w_end is None:
                # whisperx emits None for words it couldn't align (e.g.
                # numerals it doesn't have phonemes for); keep them in
                # the list without timing so the consumer sees them.
                normalized_words.append(
                    {
                        "word": str(w.get("word", "")),
                        "start": None,
                        "end": None,
                        "score": w.get("score"),
                    }
                )
                continue
            try:
                start_f = float(w_start)
                end_f = float(w_end)
            except (TypeError, ValueError):
                continue
            normalized_words.append(
                {
                    "word": str(w.get("word", "")),
                    "start": start_f,
                    "end": end_f,
                    "score": float(w.get("score", 0.0)) if w.get("score") is not None else None,
                }
            )
            if first_start is None or start_f < first_start:
                first_start = start_f
            if last_end is None or end_f > last_end:
                last_end = end_f
        chunk = chunks[src_idx]
        chunk["words"] = normalized_words
        if first_start is not None and last_end is not None:
            chunk["timestamp"] = [first_start, last_end]


def main() -> int:
    parser = argparse.ArgumentParser(description="Forced-alignment post-pass for the insane backend output.")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--align-model",
        default=None,
        help="Override the wav2vec2 model id. Useful for languages outside whisperx's defaults.",
    )
    args = parser.parse_args()

    src_json = Path(args.input_json)
    out_json = Path(args.output_json)
    src_data = json.loads(src_json.read_text(encoding="utf-8"))

    chunks = src_data.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        sys.stderr.write("insane_align: no chunks to align; passing input through\n")
        out_json.write_text(json.dumps(src_data, indent=2), encoding="utf-8")
        return 0

    language = (args.language or "").strip()
    if args.align_model is None and not _supported_language(language):
        sys.stderr.write(f"insane_align: language={language!r} has no default WhisperX alignment " "model; pass --align-model <hf-model-id> to force one. Skipping alignment.\n")
        out_json.write_text(json.dumps(src_data, indent=2), encoding="utf-8")
        return 0

    segments, index_map = _convert_chunks_to_segments(chunks)
    if not segments:
        sys.stderr.write("insane_align: no alignable chunks (all timestamps None or text empty); skipping\n")
        out_json.write_text(json.dumps(src_data, indent=2), encoding="utf-8")
        return 0

    # Lazy-imported so --help works without the heavy stack.
    import torch  # type: ignore
    import whisperx  # type: ignore

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        sys.stderr.write("insane_align: CUDA requested but not available; falling back to CPU\n")
        device = "cpu"

    sys.stderr.write(f"insane_align: loading wav2vec2 alignment model for language={language!r} on {device}\n")
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
        model_name=args.align_model,
    )

    audio = whisperx.load_audio(str(args.input_wav))
    sys.stderr.write(f"insane_align: aligning {len(segments)} segments\n")
    result = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    aligned_segments = result.get("segments", []) if isinstance(result, dict) else []
    _merge_alignment_into_chunks(chunks, aligned_segments, index_map)

    src_data["aligned"] = True
    src_data["align_language"] = language
    out_json.write_text(json.dumps(src_data, indent=2), encoding="utf-8")
    sys.stderr.write("insane_align: done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
