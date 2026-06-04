"""Unit tests for the insane → WhisperX alignment runner helpers.

These cover the parts of ``insane_align_runner`` that don't need whisperx
itself installed: the chunks→segments conversion, the back-merge into
chunks (which tightens timestamps and adds word-level data), and the
language-support guard.
"""

from __future__ import annotations

import copy
from typing import Any

from transcribe_anything.insane_align_runner import (
    _convert_chunks_to_segments,
    _merge_alignment_into_chunks,
    _supported_language,
)


def test_convert_chunks_strips_text_whitespace() -> None:
    chunks = [{"timestamp": [0.0, 1.0], "text": "  hi  "}]
    segments, index_map = _convert_chunks_to_segments(chunks)
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hi"}]
    assert index_map == [0]


def test_convert_chunks_skips_none_timestamps() -> None:
    chunks: list[dict[str, Any]] = [
        {"timestamp": [0.0, 1.0], "text": "kept"},
        {"timestamp": [None, None], "text": "dropped"},
        {"timestamp": [1.0, None], "text": "also dropped"},
        {"timestamp": [None, 2.0], "text": "also dropped 2"},
    ]
    segments, index_map = _convert_chunks_to_segments(chunks)
    assert len(segments) == 1
    assert segments[0]["text"] == "kept"
    assert index_map == [0]


def test_convert_chunks_skips_empty_text() -> None:
    chunks = [
        {"timestamp": [0.0, 1.0], "text": ""},
        {"timestamp": [1.0, 2.0], "text": "   "},
        {"timestamp": [2.0, 3.0], "text": "real"},
    ]
    segments, _ = _convert_chunks_to_segments(chunks)
    assert len(segments) == 1
    assert segments[0]["text"] == "real"


def test_convert_chunks_coerces_string_timestamps_to_float() -> None:
    chunks = [{"timestamp": ["0.5", "1.25"], "text": "stringy"}]
    segments, _ = _convert_chunks_to_segments(chunks)
    assert segments == [{"start": 0.5, "end": 1.25, "text": "stringy"}]


def test_convert_chunks_drops_unparseable_timestamps() -> None:
    chunks = [{"timestamp": ["nope", 1.0], "text": "bad"}]
    segments, index_map = _convert_chunks_to_segments(chunks)
    assert segments == []
    assert index_map == []


def test_convert_chunks_preserves_source_index_for_back_merge() -> None:
    """index_map must map align-result -> original chunk index, skipping over None-timestamp chunks."""
    chunks: list[dict[str, Any]] = [
        {"timestamp": [None, None], "text": "skip"},
        {"timestamp": [0.0, 1.0], "text": "keep first"},
        {"timestamp": [None, None], "text": "skip"},
        {"timestamp": [1.0, 2.0], "text": "keep second"},
    ]
    _, index_map = _convert_chunks_to_segments(chunks)
    assert index_map == [1, 3]


def test_merge_alignment_tightens_segment_timestamps_to_word_boundaries() -> None:
    chunks = [{"timestamp": [0.0, 5.0], "text": "hello world"}]
    aligned = [
        {
            "words": [
                {"word": "hello", "start": 0.1, "end": 0.6, "score": 0.95},
                {"word": "world", "start": 0.8, "end": 1.4, "score": 0.91},
            ]
        }
    ]
    _merge_alignment_into_chunks(chunks, aligned, [0])
    assert chunks[0]["timestamp"] == [0.1, 1.4]
    assert len(chunks[0]["words"]) == 2


def test_merge_alignment_keeps_words_with_missing_timing() -> None:
    """whisperx emits None for words it couldn't phoneme-align (e.g. numerals)."""
    chunks: list[dict[str, Any]] = [{"timestamp": [0.0, 5.0], "text": "ten apples"}]
    aligned: list[dict[str, Any]] = [
        {
            "words": [
                {"word": "ten", "start": None, "end": None, "score": None},
                {"word": "apples", "start": 0.5, "end": 1.3, "score": 0.94},
            ]
        }
    ]
    _merge_alignment_into_chunks(chunks, aligned, [0])
    assert chunks[0]["timestamp"] == [0.5, 1.3]
    assert len(chunks[0]["words"]) == 2
    assert chunks[0]["words"][0]["start"] is None


def test_merge_alignment_leaves_unaligned_chunks_alone() -> None:
    chunks = [
        {"timestamp": [0.0, 1.0], "text": "first"},
        {"timestamp": [1.0, 2.0], "text": "second"},
    ]
    # Only the first chunk was aligned (index_map mirrors).
    aligned = [{"words": [{"word": "first", "start": 0.1, "end": 0.9, "score": 0.99}]}]
    _merge_alignment_into_chunks(chunks, aligned, [0])
    assert chunks[0]["timestamp"] == [0.1, 0.9]
    # Second chunk untouched — no `words`, original timestamp preserved.
    assert "words" not in chunks[1]
    assert chunks[1]["timestamp"] == [1.0, 2.0]


def test_merge_alignment_handles_index_out_of_range_gracefully() -> None:
    """If index_map points past end of chunks (shouldn't happen, but defensive)."""
    chunks = [{"timestamp": [0.0, 1.0], "text": "only"}]
    aligned = [{"words": [{"word": "only", "start": 0.0, "end": 1.0, "score": 1.0}]}]
    _merge_alignment_into_chunks(chunks, aligned, [99])
    assert "words" not in chunks[0]
    assert chunks[0]["timestamp"] == [0.0, 1.0]


def test_merge_alignment_handles_non_dict_word_entries() -> None:
    chunks = [{"timestamp": [0.0, 2.0], "text": "x"}]
    aligned = [{"words": ["string-garbage", {"word": "x", "start": 0.1, "end": 0.9, "score": 0.9}]}]
    _merge_alignment_into_chunks(chunks, aligned, [0])
    assert len(chunks[0]["words"]) == 1
    assert chunks[0]["timestamp"] == [0.1, 0.9]


def test_supported_language_returns_false_when_whisperx_missing() -> None:
    """In a venv without whisperx installed, the check must fail-safe to False
    rather than raising — so the runner can fall back to passing the input through."""
    # Whisperx is not installed in the host env (it lives in its own iso-env).
    assert _supported_language("en") is False
    assert _supported_language("xx") is False
    assert _supported_language("") is False


def test_merge_alignment_does_not_mutate_aligned_input() -> None:
    """Defensive: the helper should not mutate the aligned-segments list either."""
    chunks = [{"timestamp": [0.0, 5.0], "text": "hello"}]
    aligned = [{"words": [{"word": "hello", "start": 0.1, "end": 0.9, "score": 0.99}]}]
    aligned_snapshot = copy.deepcopy(aligned)
    _merge_alignment_into_chunks(chunks, aligned, [0])
    assert aligned == aligned_snapshot
