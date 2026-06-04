"""Unit tests for the SenseVoice runner helpers (timestamps, segment shaping, SRT/VTT)."""

from __future__ import annotations

import json
from pathlib import Path

import transcribe_anything.sensevoice_runner as runner


def test_fmt_ts_srt_separator() -> None:
    assert runner._fmt_ts(0) == "00:00:00,000"
    assert runner._fmt_ts(1500) == "00:00:01,500"
    assert runner._fmt_ts(3_661_500) == "01:01:01,500"


def test_fmt_ts_vtt_separator() -> None:
    assert runner._fmt_ts(1500, vtt=True) == "00:00:01.500"


def test_fmt_ts_clamps_negative() -> None:
    assert runner._fmt_ts(-1) == "00:00:00,000"


def test_build_segments_uses_sentence_info_when_present() -> None:
    result = {
        "text": "hi there hello",
        "sentence_info": [
            {"start": 0, "end": 1000, "text": "Hi there", "spk": 0},
            {"start": 1500, "end": 2000, "text": "Hello", "spk": 1},
        ],
    }
    segments = runner._build_segments(result, "hi there hello")
    assert len(segments) == 2
    assert segments[0] == {"start": 0.0, "end": 1000.0, "text": "Hi there", "spk": 0}
    assert segments[1]["spk"] == 1


def test_build_segments_falls_back_to_plain_text_when_no_sentence_info() -> None:
    segments = runner._build_segments({"text": "hello"}, "hello")
    assert segments == [{"start": 0.0, "end": 0.0, "text": "hello"}]


def test_build_segments_drops_non_dict_entries_in_sentence_info() -> None:
    result = {
        "sentence_info": [
            "garbage",
            {"start": 0, "end": 100, "text": "ok"},
        ],
    }
    segments = runner._build_segments(result, "ok")
    assert len(segments) == 1
    assert segments[0]["text"] == "ok"


def test_write_srt_vtt_emits_indices_and_timestamps(tmp_path: Path) -> None:
    segments = [
        {"start": 0.0, "end": 1000.0, "text": "first", "spk": 0},
        {"start": 1500.0, "end": 2000.0, "text": "second", "spk": 1},
    ]
    srt_path = tmp_path / "out.srt"
    vtt_path = tmp_path / "out.vtt"
    runner._write_srt_vtt(segments, srt_path, vtt_path, diarize=True)

    srt = srt_path.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,000\nSpeaker 0: first" in srt
    assert "2\n00:00:01,500 --> 00:00:02,000\nSpeaker 1: second" in srt

    vtt = vtt_path.read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.000\nSpeaker 0: first" in vtt
    assert "00:00:01.500 --> 00:00:02.000\nSpeaker 1: second" in vtt


def test_write_srt_vtt_omits_speaker_prefix_when_diarize_off(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 1000.0, "text": "hello", "spk": 0}]
    srt_path = tmp_path / "out.srt"
    vtt_path = tmp_path / "out.vtt"
    runner._write_srt_vtt(segments, srt_path, vtt_path, diarize=False)
    assert "Speaker" not in srt_path.read_text(encoding="utf-8")
    assert "Speaker" not in vtt_path.read_text(encoding="utf-8")


def test_build_segments_handles_missing_spk_field() -> None:
    result = {
        "sentence_info": [
            {"start": 0, "end": 100, "text": "no speaker label"},
        ],
    }
    segments = runner._build_segments(result, "no speaker label")
    assert "spk" not in segments[0]
    # And SRT writer should be happy with that:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        runner._write_srt_vtt(segments, p / "x.srt", p / "x.vtt", diarize=True)
        assert "Speaker" not in (p / "x.srt").read_text(encoding="utf-8")


def test_json_output_round_trips(tmp_path: Path) -> None:
    """Sanity: the segment shape we write is round-trippable."""
    segments = [{"start": 0.0, "end": 1000.0, "text": "hi", "spk": 0}]
    payload = {"text": "hi", "segments": segments}
    (tmp_path / "out.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert loaded == payload
