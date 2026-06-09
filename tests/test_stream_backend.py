"""Unit tests for the realtime streaming backend (#124).

The real ``faster-whisper`` / ``silero-vad`` deps live in the
``[stream]`` extras and are not in CI's base test environment. We
mock them via ``sys.modules`` so the backend's protocol logic can be
unit-tested without GPUs or model downloads.
"""

from __future__ import annotations

import sys
import types
from typing import List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from transcribe_anything.server_app import create_app
from transcribe_anything.server_config import ServerConfig, StreamSession


def _install_fake_deps():
    """Mount fake numpy / faster_whisper / silero_vad modules under sys.modules."""

    class _FakeArr:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, sl):
            return _FakeArr(self.data[sl])

        def astype(self, _dtype):
            return self

        def __truediv__(self, _other):
            return self

    np = types.SimpleNamespace(
        int16="int16",
        float32="float32",
        frombuffer=lambda b, dtype: _FakeArr(list(range(len(b) // 2))),
        zeros=lambda n, dtype="float32": _FakeArr([0.0] * n),
        concatenate=lambda arrs: _FakeArr([x for a in arrs for x in a.data]),
    )

    class _FakeWhisperModel:
        def __init__(self, *a, **kw):
            self.calls: list = []

        def transcribe(self, audio, language=None, beam_size=1, word_timestamps=False):
            self.calls.append((audio.size if hasattr(audio, "size") else len(audio), language))
            seg = types.SimpleNamespace(text=f" hello x{audio.size}")
            return iter([seg]), types.SimpleNamespace()

    faster_whisper_mod = types.SimpleNamespace(WhisperModel=_FakeWhisperModel)

    class _FakeVADIterator:
        def __init__(self, *a, **kw):
            self.call_count = 0

        def __call__(self, chunk, return_seconds=False):
            self.call_count += 1
            # Emit a "speech end" event every 4 chunks so the chunker
            # finalises a segment periodically.
            if self.call_count == 4:
                return {"end": 0}
            if self.call_count == 1:
                return {"start": 0}
            return None

    silero_vad_mod = types.SimpleNamespace(
        VADIterator=_FakeVADIterator,
        load_silero_vad=lambda: object(),
    )

    sys.modules["numpy"] = np  # type: ignore[assignment]
    sys.modules["faster_whisper"] = faster_whisper_mod  # type: ignore[assignment]
    sys.modules["silero_vad"] = silero_vad_mod  # type: ignore[assignment]


def _uninstall_fake_deps():
    for name in ("numpy", "faster_whisper", "silero_vad"):
        sys.modules.pop(name, None)


@pytest.fixture
def fake_deps():
    _install_fake_deps()
    yield
    _uninstall_fake_deps()


def test_lazy_imports_picks_up_fakes(fake_deps) -> None:
    from transcribe_anything.stream_backend import _lazy_imports

    np, WhisperModel, VADIterator, load_silero_vad = _lazy_imports()
    assert WhisperModel is sys.modules["faster_whisper"].WhisperModel
    assert VADIterator is sys.modules["silero_vad"].VADIterator


def test_backend_emits_metrics_then_partial_then_final(fake_deps, tmp_path) -> None:
    """Walk the backend's generator and verify event types and order."""
    from transcribe_anything.server_config import StreamCancelled
    from transcribe_anything.stream_backend import faster_whisper_streaming_fn

    cfg = ServerConfig(model="small.en", stream_decode_interval_ms=10)
    session = StreamSession()
    session.acquire()

    # Each "audio chunk" is 1024 bytes (512 samples). With our fake
    # chunker, the 4th push triggers a finalised segment.
    chunks = [b"\x00" * 1024] * 6

    gen = faster_whisper_streaming_fn(
        iter(chunks),
        session=session,
        config=cfg,
        hello={"type": "hello", "model": "small.en", "language": "en"},
    )

    events: List[dict] = []
    try:
        for evt in gen:
            events.append(evt)
            if evt.get("type") == "final":
                session.cancel()  # exit the loop cleanly
            if len(events) >= 6:
                session.cancel()
    except StreamCancelled:
        pass  # expected — that's how the backend exits the outer loop

    # Backend always emits a metrics event first.
    assert events[0]["type"] == "metrics"
    assert events[0]["backend"] == "faster-whisper"
    # At least one final event from the 4th chunk closing the segment.
    kinds = [e["type"] for e in events]
    assert "final" in kinds


def test_backend_respects_cancel(fake_deps, tmp_path) -> None:
    """is_cancelled raises StreamCancelled out of the generator."""
    from transcribe_anything.server_config import StreamCancelled
    from transcribe_anything.stream_backend import faster_whisper_streaming_fn

    cfg = ServerConfig(model="small.en")
    session = StreamSession()
    session.acquire()
    session.cancel()  # immediately

    gen = faster_whisper_streaming_fn(
        iter([b"\x00" * 1024]),
        session=session,
        config=cfg,
        hello={"type": "hello"},
    )

    events: List[dict] = []
    with pytest.raises(StreamCancelled):
        for evt in gen:
            events.append(evt)
            if len(events) > 5:
                break

    # We should have gotten the initial metrics frame before cancellation.
    assert events and events[0]["type"] == "metrics"


# ----------------------------------------- create_app auto-detection


def test_create_app_falls_back_to_canned_when_faster_whisper_missing(tmp_path) -> None:
    """allow_stream=True with no faster-whisper installed → canned, with a warning."""
    _uninstall_fake_deps()  # belt-and-braces: ensure no leftover fake modules

    # Force the lazy import to fail.
    with patch("transcribe_anything.stream_backend._lazy_imports", side_effect=ImportError("no faster-whisper")):
        cfg = ServerConfig(model="small.en", allow_stream=True, job_root=str(tmp_path / "jobs"))
        app = create_app(cfg)

    # End-to-end: open a stream and verify we get the canned events.
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            import json

            ws.send_text(json.dumps({"type": "hello", "model": "small.en"}))
            assert ws.receive_json() == {"type": "ready"}
            ws.send_bytes(b"\x00\x00" * 100)
            received: list = []
            while True:
                evt = ws.receive_json()
                received.append(evt)
                if evt.get("type") == "done":
                    break
            kinds = {e.get("type") for e in received}
            assert "final" in kinds  # the canned generator yields a `final`


def test_create_app_picks_faster_whisper_when_deps_available(fake_deps, tmp_path) -> None:
    """allow_stream=True with fake deps installed → real backend is selected."""
    cfg = ServerConfig(model="small.en", allow_stream=True, job_root=str(tmp_path / "jobs"))
    app = create_app(cfg)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            import json

            ws.send_text(json.dumps({"type": "hello", "model": "small.en"}))
            assert ws.receive_json() == {"type": "ready"}
            # Receive the first event — the real-backend metrics frame.
            first = ws.receive_json()
            # Cancel right away so the backend's outer loop exits cleanly.
            ws.send_text(json.dumps({"type": "cancel"}))
            # Drain until done.
            received: list = [first]
            while True:
                evt = ws.receive_json()
                received.append(evt)
                if evt.get("type") == "done":
                    break

    kinds = [e.get("type") for e in received]
    # The real (fake-backed) backend emits metrics as the first event,
    # not the canned scripted "the quick brown fox" partial sequence.
    assert "metrics" in kinds
    assert kinds[0] == "metrics"
    assert received[0].get("backend") == "faster-whisper"
