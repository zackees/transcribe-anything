"""WhisperX CLI/API routing and argument-forwarding tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

WHISPER_OPTIONS = {
    "task": ["transcribe", "translate"],
    "language": ["en", "fr", "None"],
}

SAMPLE_WHISPERX_JSON = {
    "segments": [
        {"start": 0.0, "end": 1.0, "text": " Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": " world", "speaker": "SPEAKER_00"},
    ]
}


def _value_after(cmd: list[str], option: str) -> str:
    idx = cmd.index(option)
    return cmd[idx + 1]


def _assert_in_order(haystack: list[str], needle: list[str]) -> None:
    pos = 0
    for item in needle:
        try:
            pos = haystack.index(item, pos) + 1
        except ValueError as exc:
            raise AssertionError(f"Expected {needle} in order inside {haystack}") from exc


def test_device_enum_accepts_whisperx() -> None:
    from transcribe_anything.api import Device

    assert str(Device.WHISPERX) == "whisperx"
    assert Device.from_str("whisperx") is Device.WHISPERX


def test_cli_parser_accepts_whisperx_options_and_preserves_unknown_backend_args() -> None:
    from transcribe_anything import _cmd

    argv = [
        "transcribe-anything",
        "sample.wav",
        "--device",
        "whisperx",
        "--model",
        "tiny",
        "--language",
        "en",
        "--batch_size",
        "4",
        "--compute_type",
        "float16",
        "--align_model",
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "--threads",
        "2",
    ]
    with mock.patch.object(sys, "argv", argv):
        with mock.patch.object(_cmd, "get_whisper_options", return_value=WHISPER_OPTIONS):
            args = _cmd.parse_arguments()

    assert args.device == "whisperx"
    assert args.model == "tiny"
    assert args.language == "en"
    assert args.compute_type == "float16"
    assert args.align_model == "WAV2VEC2_ASR_LARGE_LV60K_960H"
    assert args.unknown == ["--batch_size", "4", "--threads", "2"]


def test_cli_main_routes_whisperx_to_api_and_forwards_backend_args() -> None:
    from transcribe_anything import _cmd, api

    calls: list[dict[str, Any]] = []

    def fake_transcribe(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(kwargs["output_dir"] or "out")

    argv = [
        "transcribe-anything",
        "sample.wav",
        "--device",
        "whisperx",
        "--model",
        "tiny",
        "--language",
        "en",
        "--hf_token",
        "hf_test",
        "--batch_size",
        "8",
        "--compute_type",
        "float16",
        "--diarize",
    ]
    with mock.patch.object(sys, "argv", argv):
        with mock.patch.object(_cmd, "get_whisper_options", return_value=WHISPER_OPTIONS):
            with mock.patch.object(api, "transcribe", fake_transcribe):
                assert _cmd.main() == 0

    assert len(calls) == 1
    assert calls[0]["device"] == "whisperx"
    assert calls[0]["model"] == "tiny"
    assert calls[0]["language"] == "en"
    assert calls[0]["hugging_face_token"] == "hf_test"
    assert calls[0]["other_args"] == ["--batch_size", "8", "--compute_type", "float16", "--diarize"]


def test_api_routes_device_whisperx_to_whisperx_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from transcribe_anything import api

    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"fake mp4")
    temp_wav = tmp_path / "input.wav"
    output_dir = tmp_path / "out"
    calls: list[dict[str, Any]] = []

    def fake_fetch_audio(_url_or_file: str, wav_path: str) -> None:
        Path(wav_path).write_bytes(b"fake wav")

    def fake_run_whisperx(**kwargs: Any) -> None:
        calls.append(kwargs)
        backend_dir = Path(kwargs["output_dir"])
        backend_dir.mkdir(parents=True, exist_ok=True)
        (backend_dir / "out.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n", encoding="utf-8")
        (backend_dir / "out.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
        (backend_dir / "out.txt").write_text("Hello\n", encoding="utf-8")
        (backend_dir / "out.json").write_text(json.dumps(SAMPLE_WHISPERX_JSON), encoding="utf-8")

    def fail_backend(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("device='whisperx' should not route to another backend")

    monkeypatch.setattr(api.static_ffmpeg, "add_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "make_temp_wav", lambda: str(temp_wav))
    monkeypatch.setattr(api, "fetch_audio", fake_fetch_audio)
    monkeypatch.setattr(api, "run_whisperx", fake_run_whisperx, raising=False)
    monkeypatch.setattr(api, "run_whisper", fail_backend)
    monkeypatch.setattr(api, "run_insanely_fast_whisper", fail_backend)
    monkeypatch.setattr(api, "run_whisper_mac_mlx", fail_backend)

    result = api.transcribe(
        url_or_file=str(input_file),
        output_dir=str(output_dir),
        model="tiny",
        task="translate",
        language="en",
        device="whisperx",
        hugging_face_token="hf_test",
        other_args=["--batch_size", "4", "--compute_type", "float16"],
    )

    assert result == str(output_dir.resolve())
    assert calls == [
        {
            "input_wav": temp_wav,
            "model": "tiny",
            "output_dir": calls[0]["output_dir"],
            "task": "translate",
            "language": "en",
            "hugging_face_token": "hf_test",
            "other_args": ["--batch_size", "4", "--compute_type", "float16"],
            "use_xpu": False,
        }
    ]
    for filename in ["out.srt", "out.vtt", "out.txt", "out.json"]:
        assert (output_dir / filename).exists()


def test_run_whisperx_uses_isolated_env_and_forwards_backend_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import transcribe_anything.whisperx as whisperx

    input_wav = tmp_path / "clip.wav"
    input_wav.write_bytes(b"fake wav")
    output_dir = tmp_path / "backend-out"
    other_args = ["--batch_size", "4", "--compute_type", "float16", "--diarize"]

    class FakeProc:
        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    class FakeEnv:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def _record(self, cmd: list[Any]) -> None:
            cmd_list = [str(arg) for arg in cmd]
            self.calls.append(cmd_list)
            generated_dir = Path(_value_after(cmd_list, "--output_dir"))
            generated_dir.mkdir(parents=True, exist_ok=True)
            (generated_dir / "clip.json").write_text(json.dumps(SAMPLE_WHISPERX_JSON), encoding="utf-8")

        def open_proc(self, cmd: list[Any], **_kwargs: Any) -> FakeProc:
            self._record(cmd)
            return FakeProc()

        def run(self, cmd: list[Any], **_kwargs: Any) -> SimpleNamespace:
            self._record(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    fake_env = FakeEnv()
    monkeypatch.setattr(whisperx, "get_environment", lambda use_xpu=False: fake_env)
    monkeypatch.setattr(whisperx.static_ffmpeg, "add_paths", lambda *args, **kwargs: None)

    whisperx.run_whisperx(
        input_wav=input_wav,
        model="tiny",
        output_dir=output_dir,
        task="translate",
        language="en",
        hugging_face_token="hf_test",
        other_args=other_args,
    )

    assert len(fake_env.calls) == 1
    cmd = fake_env.calls[0]
    assert cmd[0] == "whisperx"
    assert _value_after(cmd, "--model") == "tiny"
    staged_output_dir = Path(_value_after(cmd, "--output_dir"))
    assert staged_output_dir.name == "whisperx"
    assert staged_output_dir != output_dir
    assert _value_after(cmd, "--output_format") == "all"
    assert _value_after(cmd, "--task") == "translate"
    assert _value_after(cmd, "--language") == "en"
    assert _value_after(cmd, "--hf_token") == "hf_test"
    _assert_in_order(cmd, other_args)
    assert other_args == ["--batch_size", "4", "--compute_type", "float16", "--diarize"]
    for filename in ["out.json", "out.srt", "out.vtt", "out.txt"]:
        assert (output_dir / filename).exists()
