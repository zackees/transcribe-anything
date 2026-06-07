"""Unit tests for the HTTP client used by ``--remote`` mode (issue #107).

We drive the real FastAPI app through ``httpx.ASGITransport`` — no socket,
no subprocess. The actual transcribe() call is mocked.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from transcribe_anything import client as client_mod
from transcribe_anything.client import (
    RemoteTranscriberError,
    resolve_remote_and_token,
    transcribe_remote,
    transcribe_remote_async,
)
from transcribe_anything.server_app import create_app
from transcribe_anything.server_config import ServerConfig


def _fake_transcribe(text: str = "hello"):
    def fn(*, url_or_file: str, output_dir: str, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "out.txt").write_text(text, encoding="utf-8")
        (out / "out.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        (out / "out.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (out / "out.json").write_text(json.dumps({"text": text}), encoding="utf-8")
        return str(out)

    return fn


@pytest.fixture
def asgi_app(tmp_path):
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path / "jobs"))
    return create_app(cfg, transcribe_fn=_fake_transcribe())


@pytest.fixture
def asgi_client_factory(asgi_app, monkeypatch):
    """Monkeypatch httpx.Client used by client.py to talk to our in-process app.

    ``httpx.ASGITransport`` is async-only, so a sync ``httpx.Client`` can't
    use it directly. ``starlette.testclient.TestClient`` is a sync
    ``httpx.Client`` subclass that drives the ASGI app via a worker thread,
    so it presents exactly the surface our sync client expects.
    """

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return TestClient(asgi_app, base_url="http://testserver", headers=kwargs.get("headers") or {})

    monkeypatch.setattr(client_mod.httpx, "Client", factory)
    return factory


def test_transcribe_remote_url_path(asgi_client_factory, tmp_path) -> None:
    out_dir = tmp_path / "out"
    result = transcribe_remote(
        url_or_file="https://example.com/foo.mp3",
        remote="http://testserver",
        output_dir=str(out_dir),
        poll_interval_seconds=0.01,
    )
    assert Path(result).resolve() == out_dir.resolve()
    assert (out_dir / "out.txt").is_file()
    assert (out_dir / "out.srt").is_file()


def test_transcribe_remote_file_upload(asgi_client_factory, tmp_path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    out_dir = tmp_path / "out"
    result = transcribe_remote(
        url_or_file=str(src),
        remote="http://testserver",
        output_dir=str(out_dir),
        poll_interval_seconds=0.01,
    )
    assert Path(result).resolve() == out_dir.resolve()
    assert (out_dir / "out.txt").read_text(encoding="utf-8") == "hello"


def test_transcribe_remote_rejects_missing_file(asgi_client_factory, tmp_path) -> None:
    with pytest.raises(RemoteTranscriberError):
        transcribe_remote(
            url_or_file=str(tmp_path / "nope.wav"),
            remote="http://testserver",
            output_dir=str(tmp_path / "out"),
        )


def test_transcribe_remote_surfaces_daemon_4xx(asgi_client_factory, tmp_path) -> None:
    with pytest.raises(RemoteTranscriberError) as exc:
        transcribe_remote(
            url_or_file="https://example.com/x.mp3",
            remote="http://testserver",
            output_dir=str(tmp_path / "out"),
            # Override locked field -> daemon returns 400.
            other_args=None,
            model="large-v3",  # daemon default is "tiny" and allow_client_model is False
        )
    assert "400" in str(exc.value)


def test_transcribe_remote_propagates_failed_job(monkeypatch, asgi_client_factory, tmp_path) -> None:
    """If the worker's transcribe() raises, the client should raise RemoteTranscriberError."""

    # Rebuild app with a failing transcribe_fn.
    def failing(**_kwargs):
        raise RuntimeError("boom")

    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path / "jobs"))
    app = create_app(cfg, transcribe_fn=failing)

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return TestClient(app, base_url="http://testserver", headers=kwargs.get("headers") or {})

    monkeypatch.setattr(client_mod.httpx, "Client", factory)

    with pytest.raises(RemoteTranscriberError) as exc:
        transcribe_remote(
            url_or_file="https://example.com/x.mp3",
            remote="http://testserver",
            output_dir=str(tmp_path / "out"),
            poll_interval_seconds=0.01,
        )
    assert "failed" in str(exc.value).lower()


def test_resolve_remote_and_token_arg_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_ANYTHING_REMOTE", "http://from-env")
    monkeypatch.setenv("TRANSCRIBE_ANYTHING_TOKEN", "env-token")
    remote, token = resolve_remote_and_token(remote_arg="http://from-arg", token_arg="arg-token")
    assert remote == "http://from-arg"
    assert token == "arg-token"


def test_resolve_remote_and_token_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_ANYTHING_REMOTE", "http://from-env")
    monkeypatch.setenv("TRANSCRIBE_ANYTHING_TOKEN", "env-token")
    remote, token = resolve_remote_and_token(remote_arg=None, token_arg=None)
    assert remote == "http://from-env"
    assert token == "env-token"


def test_resolve_remote_and_token_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("TRANSCRIBE_ANYTHING_REMOTE", raising=False)
    monkeypatch.delenv("TRANSCRIBE_ANYTHING_TOKEN", raising=False)
    remote, token = resolve_remote_and_token(remote_arg=None, token_arg=None)
    assert remote is None
    assert token is None


# --------------------- async client ---------------------


@pytest.fixture
def asgi_async_client_factory(asgi_app, monkeypatch):
    """Monkeypatch ``httpx.AsyncClient`` used by ``transcribe_remote_async``.

    Unlike sync httpx, the async client *does* accept ``ASGITransport``
    directly, so we just inject one pointing at the in-process FastAPI app.
    """
    transport = httpx.ASGITransport(app=asgi_app)
    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        kwargs.setdefault("base_url", "http://testserver")
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", factory)
    return factory


def test_transcribe_remote_async_url_path(asgi_async_client_factory, tmp_path) -> None:
    out_dir = tmp_path / "out"
    result = asyncio.run(
        transcribe_remote_async(
            url_or_file="https://example.com/foo.mp3",
            remote="http://testserver",
            output_dir=str(out_dir),
            poll_interval_seconds=0.01,
        )
    )
    assert Path(result).resolve() == out_dir.resolve()
    assert (out_dir / "out.txt").is_file()
    assert (out_dir / "out.srt").is_file()


def test_transcribe_remote_async_file_upload(asgi_async_client_factory, tmp_path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    out_dir = tmp_path / "out"
    result = asyncio.run(
        transcribe_remote_async(
            url_or_file=str(src),
            remote="http://testserver",
            output_dir=str(out_dir),
            poll_interval_seconds=0.01,
        )
    )
    assert Path(result).resolve() == out_dir.resolve()
    assert (out_dir / "out.txt").read_text(encoding="utf-8") == "hello"


def test_transcribe_remote_async_rejects_missing_file(asgi_async_client_factory, tmp_path) -> None:
    with pytest.raises(RemoteTranscriberError):
        asyncio.run(
            transcribe_remote_async(
                url_or_file=str(tmp_path / "nope.wav"),
                remote="http://testserver",
                output_dir=str(tmp_path / "out"),
            )
        )


def test_transcribe_remote_async_surfaces_daemon_4xx(asgi_async_client_factory, tmp_path) -> None:
    with pytest.raises(RemoteTranscriberError) as exc:
        asyncio.run(
            transcribe_remote_async(
                url_or_file="https://example.com/x.mp3",
                remote="http://testserver",
                output_dir=str(tmp_path / "out"),
                model="large-v3",  # locked field, daemon default is "tiny"
            )
        )
    assert "400" in str(exc.value)
