"""Unit tests for the host-side insane → WhisperX alignment wrapper.

These cover the best-effort fallback behavior: when input JSON has no
chunks, when the runner script is missing, when the runner fails. None
of these tests require WhisperX or CUDA to actually run alignment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

from transcribe_anything import insane_align
from transcribe_anything.insane_align import apply_forced_alignment


def test_apply_forced_alignment_passes_through_when_no_chunks() -> None:
    data = {"text": "no chunks here", "language": "en"}
    result = apply_forced_alignment(data, input_wav=Path("/tmp/nope.wav"), language="en")
    assert result is data  # same dict reference; no copy needed


def test_apply_forced_alignment_passes_through_when_chunks_empty() -> None:
    data = {"text": "", "chunks": []}
    result = apply_forced_alignment(data, input_wav=Path("/tmp/nope.wav"), language="en")
    assert result == data


def test_apply_forced_alignment_passes_through_when_runner_missing(monkeypatch) -> None:
    """If the runner script went missing somehow, we don't kill the whole transcribe call."""
    monkeypatch.setattr(insane_align, "RUNNER", Path("/definitely/missing/insane_align_runner.py"))
    data = {"chunks": [{"timestamp": [0.0, 1.0], "text": "hi"}]}
    result = apply_forced_alignment(data, input_wav=Path("/tmp/x.wav"), language="en")
    assert result == data


def test_apply_forced_alignment_passes_through_when_env_build_fails(monkeypatch) -> None:
    monkeypatch.setattr(insane_align, "get_whisperx_environment", mock.Mock(side_effect=RuntimeError("env busted")))
    data = {"chunks": [{"timestamp": [0.0, 1.0], "text": "hi"}]}
    result = apply_forced_alignment(data, input_wav=Path("/tmp/x.wav"), language="en")
    assert result == data


def test_apply_forced_alignment_passes_through_when_runner_crashes(monkeypatch) -> None:
    """A non-zero exit from the alignment runner must NOT propagate as an exception."""
    class FakeEnv:
        def run(self, *args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

    monkeypatch.setattr(insane_align, "get_whisperx_environment", mock.Mock(return_value=FakeEnv()))
    data = {"chunks": [{"timestamp": [0.0, 1.0], "text": "hi"}]}
    result = apply_forced_alignment(data, input_wav=Path("/tmp/x.wav"), language="en")
    assert result == data


def test_apply_forced_alignment_passes_through_when_runner_writes_no_output(monkeypatch) -> None:
    class FakeEnv:
        def run(self, *args, **kwargs):
            # Successful exit but no output file written.
            return mock.Mock(returncode=0)

    monkeypatch.setattr(insane_align, "get_whisperx_environment", mock.Mock(return_value=FakeEnv()))
    data = {"chunks": [{"timestamp": [0.0, 1.0], "text": "hi"}]}
    result = apply_forced_alignment(data, input_wav=Path("/tmp/x.wav"), language="en")
    assert result == data


def test_apply_forced_alignment_invokes_runner_with_correct_cli_shape(monkeypatch) -> None:
    """When everything goes right, the wrapper writes input JSON, invokes the runner,
    and returns the runner's output."""
    captured: dict[str, list[str]] = {}

    class FakeEnv:
        def run(self, cmd_list, **kwargs):
            captured["cmd_list"] = list(cmd_list)
            # Find --output-json and write an enriched file there.
            out_idx = cmd_list.index("--output-json")
            out_path = Path(cmd_list[out_idx + 1])
            out_path.write_text('{"chunks": [{"timestamp": [0.0, 1.0], "text": "hi", "words": []}], "aligned": true}')
            return mock.Mock(returncode=0)

    monkeypatch.setattr(insane_align, "get_whisperx_environment", mock.Mock(return_value=FakeEnv()))
    data = {"chunks": [{"timestamp": [0.0, 1.0], "text": "hi"}], "text": "hi"}
    result = apply_forced_alignment(
        data,
        input_wav=Path("/some/audio.wav"),
        language="en",
        align_model="my-org/my-w2v",
        device="cpu",
    )

    # Runner output should propagate.
    assert result["aligned"] is True
    assert "words" in result["chunks"][0]

    # CLI shape: arg pairs we promised.
    cmd = captured["cmd_list"]
    assert "--input-wav" in cmd
    assert cmd[cmd.index("--input-wav") + 1] == str(Path("/some/audio.wav"))
    assert "--language" in cmd
    assert cmd[cmd.index("--language") + 1] == "en"
    assert "--device" in cmd
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert "--align-model" in cmd
    assert cmd[cmd.index("--align-model") + 1] == "my-org/my-w2v"


def test_align_device_picks_cpu_when_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(insane_align, "has_nvidia_smi", lambda: False)
    monkeypatch.setattr(insane_align.sys, "platform", "linux")
    assert insane_align._align_device() == "cpu"


def test_align_device_picks_mps_on_darwin_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(insane_align, "has_nvidia_smi", lambda: False)
    monkeypatch.setattr(insane_align.sys, "platform", "darwin")
    assert insane_align._align_device() == "mps"


def test_align_device_picks_cuda_when_nvidia_smi_present(monkeypatch) -> None:
    monkeypatch.setattr(insane_align, "has_nvidia_smi", lambda: True)
    assert insane_align._align_device() == "cuda"
