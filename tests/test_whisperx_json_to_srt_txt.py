"""WhisperX JSON conversion and output normalization tests."""

from __future__ import annotations

import json
from pathlib import Path

WHISPERX_JSON = {
    "segments": [
        {
            "start": 0.0,
            "end": 1.25,
            "text": " Hello world",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                {"word": "world", "start": 0.5, "end": 1.25, "speaker": "SPEAKER_00"},
            ],
        },
        {
            "start": 1.25,
            "end": 2.5,
            "text": " Second line",
            "speaker": "SPEAKER_01",
            "words": [
                {"word": "Second", "start": 1.25, "end": 1.75, "speaker": "SPEAKER_01"},
                {"word": "line", "start": 1.75, "end": 2.5, "speaker": "SPEAKER_01"},
            ],
        },
    ],
    "language": "en",
}


def test_convert_whisperx_json_to_srt() -> None:
    from transcribe_anything.whisperx import _json_to_srt

    srt = _json_to_srt(WHISPERX_JSON)

    assert "1\n00:00:00,000 --> 00:00:01,250\n[SPEAKER_00]: Hello world" in srt
    assert "2\n00:00:01,250 --> 00:00:02,500\n[SPEAKER_01]: Second line" in srt


def test_convert_whisperx_json_to_text_joins_segments() -> None:
    from transcribe_anything.whisperx import _json_to_txt

    txt = _json_to_txt(WHISPERX_JSON)

    assert "[SPEAKER_00]: Hello world" in txt
    assert "[SPEAKER_01]: Second line" in txt


def test_convert_whisperx_json_to_text_prefers_top_level_text() -> None:
    from transcribe_anything.whisperx import _json_to_txt

    data = dict(WHISPERX_JSON)
    data["text"] = " Already normalized transcript "

    assert _json_to_txt(data) == "Already normalized transcript"


def test_normalize_whisperx_outputs_writes_project_output_contract(tmp_path: Path) -> None:
    from transcribe_anything.whisperx import _normalize_outputs

    whisperx_output_dir = tmp_path / "whisperx"
    output_dir = tmp_path / "normalized"
    whisperx_output_dir.mkdir()
    (whisperx_output_dir / "video.json").write_text(json.dumps(WHISPERX_JSON), encoding="utf-8")

    _normalize_outputs(whisperx_output_dir, output_dir)

    assert json.loads((output_dir / "out.json").read_text(encoding="utf-8"))["segments"][0]["text"] == "Hello world"
    txt = (output_dir / "out.txt").read_text(encoding="utf-8")
    assert "[SPEAKER_00]: Hello world" in txt
    assert "[SPEAKER_01]: Second line" in txt

    srt = (output_dir / "out.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,250" in srt
    assert "Hello world" in srt

    vtt = (output_dir / "out.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.250" in vtt

    speaker_json = json.loads((output_dir / "speaker.json").read_text(encoding="utf-8"))
    assert speaker_json[0]["speaker"] == "SPEAKER_00"
    assert speaker_json[0]["timestamp"] == [0.0, 1.25]
    assert speaker_json[0]["text"] == "Hello world"
    assert speaker_json[1]["speaker"] == "SPEAKER_01"
    assert speaker_json[1]["timestamp"] == [1.25, 2.5]
    assert speaker_json[1]["text"] == "Second line"
