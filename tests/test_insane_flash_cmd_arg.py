"""insane-flash CLI/API routing and backend argument tests."""

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

SAMPLE_INSANE_JSON = {"text": "Hello", "chunks": [{"timestamp": [0.0, 1.0], "text": " Hello"}]}


def _value_after(cmd: list[str], option: str) -> str:
    idx = cmd.index(option)
    return cmd[idx + 1]


def test_device_enum_accepts_insane_flash() -> None:
    from transcribe_anything.api import Device

    assert str(Device.INSANE_FLASH) == "insane-flash"
    assert Device.from_str("insane-flash") is Device.INSANE_FLASH


def test_cli_parser_accepts_insane_flash_and_preserves_backend_args() -> None:
    from transcribe_anything import _cmd

    argv = [
        "transcribe-anything",
        "sample.wav",
        "--device",
        "insane-flash",
        "--model",
        "tiny",
        "--language",
        "en",
        "--timestamp",
        "word",
        "--batch-size",
        "2",
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(_cmd, "get_whisper_options", return_value=WHISPER_OPTIONS),
    ):
        args = _cmd.parse_arguments()

    assert args.device == "insane-flash"
    assert args.timestamp == "word"
    assert args.unknown == ["--batch-size", "2"]


def test_cli_parser_does_not_probe_standard_whisper_env_for_explicit_device() -> None:
    from transcribe_anything import _cmd

    argv = ["transcribe-anything", "sample.wav", "--device", "insane", "--model", "tiny"]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(_cmd, "get_whisper_options", return_value=WHISPER_OPTIONS),
        mock.patch("transcribe_anything.whisper.get_computing_device", side_effect=AssertionError("should not probe standard whisper env")),
    ):
        args = _cmd.parse_arguments()

    assert args.device == "insane"


def test_cli_main_routes_insane_flash_to_api_and_forwards_insane_args(tmp_path: Path) -> None:
    from transcribe_anything import _cmd, api

    calls: list[dict[str, Any]] = []

    def fake_transcribe(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(kwargs["output_dir"] or "out")

    argv = [
        "transcribe-anything",
        "sample.wav",
        "--device",
        "insane-flash",
        "--model",
        "tiny",
        "--language",
        "en",
        "--timestamp",
        "word",
        "--batch-size",
        "8",
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(_cmd, "get_whisper_options", return_value=WHISPER_OPTIONS),
        mock.patch.object(_cmd, "user_cache_dir", return_value=str(tmp_path)),
        mock.patch.dict("os.environ", {}, clear=True),
        mock.patch.object(api, "transcribe", fake_transcribe),
    ):
        assert _cmd.main() == 0

    assert len(calls) == 1
    assert calls[0]["device"] == "insane-flash"
    assert calls[0]["model"] == "tiny"
    assert calls[0]["language"] == "en"
    assert calls[0]["other_args"] == ["--batch-size", "8", "--timestamp", "word"]


def test_api_routes_device_insane_flash_to_insane_backend_with_flash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from transcribe_anything import api

    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"fake mp4")
    temp_wav = tmp_path / "input.wav"
    output_dir = tmp_path / "out"
    calls: list[dict[str, Any]] = []

    def fake_fetch_audio(_url_or_file: str, wav_path: str) -> None:
        Path(wav_path).write_bytes(b"fake wav")

    def fake_run_insane(**kwargs: Any) -> None:
        calls.append(kwargs)
        backend_dir = Path(kwargs["output_dir"])
        backend_dir.mkdir(parents=True, exist_ok=True)
        (backend_dir / "out.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n", encoding="utf-8")
        (backend_dir / "out.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
        (backend_dir / "out.txt").write_text("Hello\n", encoding="utf-8")
        (backend_dir / "out.json").write_text(json.dumps(SAMPLE_INSANE_JSON), encoding="utf-8")

    def fail_backend(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("device='insane-flash' should route only to the insane backend")

    monkeypatch.setattr(api.static_ffmpeg, "add_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "make_temp_wav", lambda: str(temp_wav))
    monkeypatch.setattr(api, "fetch_audio", fake_fetch_audio)
    monkeypatch.setattr(api, "run_insanely_fast_whisper", fake_run_insane)
    monkeypatch.setattr(api, "run_whisper", fail_backend)
    monkeypatch.setattr(api, "run_whisperx", fail_backend, raising=False)
    monkeypatch.setattr(api, "run_whisper_mac_mlx", fail_backend)

    result = api.transcribe(
        url_or_file=str(input_file),
        output_dir=str(output_dir),
        model="tiny",
        task="transcribe",
        language="en",
        device="insane-flash",
        other_args=["--batch-size", "4"],
    )

    assert result == str(output_dir.resolve())
    assert len(calls) == 1
    assert calls[0]["flash"] is True
    assert calls[0]["other_args"] == ["--batch-size", "4"]
    for filename in ["out.srt", "out.vtt", "out.txt", "out.json"]:
        assert (output_dir / filename).exists()


def test_prepare_insane_args_forces_flash_true_without_mutating_input() -> None:
    from transcribe_anything.insanely_fast_whisper import _prepare_insane_args

    args = ["--batch-size", "4", "--flash", "True"]
    prepared = _prepare_insane_args(args, force_flash=True)

    assert args == ["--batch-size", "4", "--flash", "True"]
    assert prepared == ["--batch-size", "4", "--flash", "True"]


def test_prepare_insane_args_rejects_flash_false() -> None:
    from transcribe_anything.insanely_fast_whisper import _prepare_insane_args

    with pytest.raises(ValueError, match="requires FlashAttention"):
        _prepare_insane_args(["--flash=False"], force_flash=True)


def test_run_insanely_fast_whisper_uses_flash_env_and_forces_flash_arg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import transcribe_anything.insanely_fast_whisper as insane

    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"fake wav")
    output_dir = tmp_path / "backend"
    env_calls: list[bool] = []

    class FakeProc:
        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    class FakeEnv:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def open_proc(self, cmd: list[Any], **_kwargs: Any) -> FakeProc:
            cmd_list = [str(arg) for arg in cmd]
            self.commands.append(cmd_list)
            transcript_path = Path(_value_after(cmd_list, "--transcript-path"))
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(json.dumps(SAMPLE_INSANE_JSON), encoding="utf-8")
            return FakeProc()

        def run(self, cmd: list[Any], **_kwargs: Any) -> SimpleNamespace:
            self.commands.append([str(arg) for arg in cmd])
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    fake_env = FakeEnv()

    def fake_get_environment(*, flash: bool = False, has_nvidia: bool | None = None) -> FakeEnv:
        del has_nvidia
        env_calls.append(flash)
        return fake_env

    monkeypatch.setattr(insane.static_ffmpeg, "add_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(insane, "get_environment", fake_get_environment)
    monkeypatch.setattr(insane, "verify_flash_attention_available", lambda _env: None)
    monkeypatch.setattr(insane, "get_device_id", lambda: "0")
    monkeypatch.setattr(insane, "get_batch_size", lambda: None)
    monkeypatch.setattr(insane, "get_wave_duration", lambda _path: 1.0)

    insane.run_insanely_fast_whisper(
        input_wav=input_wav,
        model="tiny",
        output_dir=output_dir,
        task="transcribe",
        language="en",
        other_args=["--batch-size", "4"],
        flash=True,
    )

    assert env_calls == [True]
    cmd = fake_env.commands[0]
    assert cmd[0] == "insanely-fast-whisper"
    assert _value_after(cmd, "--batch-size") == "4"
    assert _value_after(cmd, "--flash") == "True"
    assert (output_dir / "out.srt").exists()
    assert (output_dir / "out.vtt").exists()
    assert (output_dir / "out.txt").exists()
