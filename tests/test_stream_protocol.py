"""Tests for the realtime streaming WebSocket protocol (#122 skeleton).

The real backend (faster-whisper) is a follow-up PR. These tests cover
the protocol + StreamSession semantics using the canned in-tree backend.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from transcribe_anything.server_app import create_app
from transcribe_anything.server_config import ServerConfig, StreamSession


def _cfg(tmp_path, *, allow_stream: bool = True, **kw) -> ServerConfig:
    return ServerConfig(
        host="127.0.0.1",
        model="tiny",
        job_root=str(tmp_path / "jobs"),
        allow_stream=allow_stream,
        **kw,
    )


def _hello_frame() -> str:
    return json.dumps({"type": "hello", "model": "small.en", "language": "en", "sample_rate": 16000, "encoding": "pcm16le"})


# ---------------------------------------------------------------- StreamSession


def test_stream_session_acquire_release_cycle() -> None:
    s = StreamSession()
    assert s.acquire() is True
    assert s.is_active is True
    assert s.acquire() is False  # second acquire while held -> False
    s.release()
    assert s.is_active is False
    assert s.acquire() is True  # available again after release


def test_stream_session_cancel_flag() -> None:
    s = StreamSession()
    s.acquire()
    assert s.is_cancelled is False
    s.cancel()
    assert s.is_cancelled is True
    s.release()
    # Next acquire clears the flag.
    s.acquire()
    assert s.is_cancelled is False


def test_stream_session_runtime_seconds() -> None:
    s = StreamSession()
    assert s.runtime_seconds is None
    s.acquire()
    rt = s.runtime_seconds
    assert rt is not None and rt >= 0.0
    s.release()
    assert s.runtime_seconds is None


# ---------------------------------------------------------------- WS endpoint


def test_stream_endpoint_disabled_by_default(tmp_path) -> None:
    cfg = _cfg(tmp_path, allow_stream=False)
    app = create_app(cfg)
    with TestClient(app) as client:
        with pytest.raises(Exception):
            # Should get error frame then close with WS_CLOSE_NOT_ALLOWED.
            with client.websocket_connect("/v1/stream") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert msg["code"] == "stream_disabled"
                # Now receiving past the close should raise.
                ws.receive_json()


def test_stream_endpoint_happy_path_yields_done(tmp_path) -> None:
    cfg = _cfg(tmp_path, allow_stream=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_text(_hello_frame())
            ready = ws.receive_json()
            assert ready == {"type": "ready"}

            # Send some fake audio so the canned backend has something to drain.
            for _ in range(3):
                ws.send_bytes(b"\x00\x00" * 1600)  # 100ms of silence @ 16kHz

            collected = []
            # Read events until done.
            while True:
                msg = ws.receive_json()
                collected.append(msg)
                if msg.get("type") == "done":
                    break

            kinds = [m["type"] for m in collected]
            assert "partial" in kinds
            assert "final" in kinds
            assert kinds[-1] == "done"

            # rev monotonically increases across partial+final events.
            revs = [m["rev"] for m in collected if "rev" in m]
            assert revs == sorted(revs)
            assert len(set(revs)) == len(revs)


def test_stream_endpoint_rejects_non_hello_first_frame(tmp_path) -> None:
    cfg = _cfg(tmp_path, allow_stream=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_text("not json at all")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["code"] == "bad_hello"


def test_stream_endpoint_rejects_wrong_type_first_frame(tmp_path) -> None:
    cfg = _cfg(tmp_path, allow_stream=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_text(json.dumps({"type": "cancel"}))
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["code"] == "bad_hello"


def test_stream_endpoint_busy_when_session_already_acquired(tmp_path) -> None:
    """Pre-acquire the daemon's StreamSession; the WS handler must reject the new connection.

    Done by reaching into ``app.state.stream_session`` directly instead of
    holding a parallel WebSocket open from a worker thread — the latter
    deadlocks the sync TestClient.
    """
    cfg = _cfg(tmp_path, allow_stream=True)
    app = create_app(cfg)
    # Simulate a live session by pre-acquiring before the new connection.
    assert app.state.stream_session.acquire() is True
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/v1/stream") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert msg["code"] == "busy"
    finally:
        app.state.stream_session.release()


def test_stream_endpoint_requires_auth_when_non_loopback(tmp_path) -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret", model="tiny", allow_stream=True, job_root=str(tmp_path))
    app = create_app(cfg)
    with TestClient(app) as client:
        # No header → 4401 close.
        with client.websocket_connect("/v1/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["code"] == "unauthorized"

    with TestClient(app) as client:
        # Correct header → ready frame.
        with client.websocket_connect("/v1/stream", headers={"Authorization": "Bearer secret"}) as ws:
            ws.send_text(_hello_frame())
            msg = ws.receive_json()
            assert msg["type"] == "ready"


def test_stream_endpoint_pluggable_streaming_fn(tmp_path) -> None:
    """A custom streaming_fn replaces the canned backend end-to-end."""

    def _fn(audio_iter, *, session, config, hello):
        # Confirm hello is forwarded; drain a frame; emit a single final.
        assert hello["type"] == "hello"
        assert hello["language"] == "en"
        try:
            next(iter(audio_iter))
        except StopIteration:
            pass
        yield {"type": "final", "text": "custom backend ran", "rev": 1}

    cfg = _cfg(tmp_path, allow_stream=True)
    app = create_app(cfg, streaming_fn=_fn)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_text(_hello_frame())
            assert ws.receive_json() == {"type": "ready"}
            ws.send_bytes(b"\x00" * 100)
            collected = []
            while True:
                msg = ws.receive_json()
                collected.append(msg)
                if msg.get("type") == "done":
                    break
            assert {"type": "final", "text": "custom backend ran", "rev": 1} in collected
