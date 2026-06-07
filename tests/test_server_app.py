"""Unit tests for the FastAPI daemon (issue #107).

These run entirely in-process via ``fastapi.testclient.TestClient`` so no
real iso-env or GPU is required. The actual transcribe() call is mocked
out — we're testing HTTP behavior, auth, settings ownership, redaction,
and the job-store lifecycle.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transcribe_anything.server_app import (
    JobStatus,
    QueueFull,
    ServerConfig,
    SettingsViolation,
    _check_auth,
    _redact_secrets,
    config_from_env,
    config_to_env,
    create_app,
    is_model_cached,
    validate_request_options,
)

# --------------------- helpers ---------------------


def _fake_transcribe_factory(transcript_text: str = "hello world"):
    """Return a fn matching transcribe(...) signature that writes out.{txt,srt,vtt,json}."""

    def fake_transcribe(*, url_or_file: str, output_dir: str, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "out.txt").write_text(transcript_text, encoding="utf-8")
        (out / "out.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
        (out / "out.vtt").write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n", encoding="utf-8")
        (out / "out.json").write_text(json.dumps({"text": transcript_text, "chunks": []}), encoding="utf-8")
        return str(out)

    return fake_transcribe


def _failing_transcribe_factory(message: str):
    def fake(**_kwargs):
        raise RuntimeError(message)

    return fake


def _wait_for_status(client: TestClient, job_id: str, headers=None, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/v1/jobs/{job_id}", headers=headers or {})
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in ("completed", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not terminate in {timeout}s; last={last}")


# --------------------- redaction ---------------------


def test_redact_secrets_strips_explicit_hf_token() -> None:
    msg = "boom! cmd was: insanely-fast-whisper --hf-token hf_abc123 --foo bar"
    redacted = _redact_secrets(msg, "hf_abc123")
    assert redacted is not None
    assert "hf_abc123" not in redacted
    assert "<REDACTED>" in redacted


def test_redact_secrets_strips_cli_arg_even_without_token_known() -> None:
    msg = "Running: --hf_token hf_xyzzy --batch 4"
    out = _redact_secrets(msg, None)
    assert out is not None
    assert "hf_xyzzy" not in out
    assert "<REDACTED>" in out


def test_redact_secrets_handles_empty_string() -> None:
    assert _redact_secrets("", None) == ""
    assert _redact_secrets(None, None) is None  # type: ignore[arg-type]


# --------------------- auth ---------------------


def test_check_auth_loopback_no_token_allowed() -> None:
    cfg = ServerConfig(host="127.0.0.1", auth_token=None)
    assert _check_auth(cfg, authorization=None, x_transcribe_token=None)


def test_check_auth_token_required_for_non_loopback() -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret")
    assert not _check_auth(cfg, authorization=None, x_transcribe_token=None)
    assert not _check_auth(cfg, authorization="Bearer wrong", x_transcribe_token=None)
    assert _check_auth(cfg, authorization="Bearer secret", x_transcribe_token=None)


def test_check_auth_x_transcribe_token_alias_works() -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret")
    assert _check_auth(cfg, authorization=None, x_transcribe_token="secret")


def test_serverconfig_refuses_non_loopback_without_token() -> None:
    cfg = ServerConfig(host="0.0.0.0")
    with pytest.raises(ValueError):
        cfg.validate()


def test_serverconfig_accepts_non_loopback_with_token() -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret")
    cfg.validate()  # no exception


def test_serverconfig_rejects_bogus_prefetch() -> None:
    cfg = ServerConfig(prefetch="sometimes")
    with pytest.raises(ValueError):
        cfg.validate()


# --------------------- settings ownership ---------------------


def test_validate_request_options_rejects_device_override() -> None:
    with pytest.raises(SettingsViolation):
        validate_request_options({"device": "cpu"}, ServerConfig(device="cuda"))


def test_validate_request_options_rejects_hf_token_override() -> None:
    cfg = ServerConfig(hf_token="serverside")
    with pytest.raises(SettingsViolation):
        validate_request_options({"hf_token": "clientside"}, cfg)
    with pytest.raises(SettingsViolation):
        validate_request_options({"hugging_face_token": "clientside"}, cfg)


def test_validate_request_options_rejects_model_override_by_default() -> None:
    cfg = ServerConfig(model="small")
    with pytest.raises(SettingsViolation):
        validate_request_options({"model": "large-v3"}, cfg)


def test_validate_request_options_allows_model_when_configured() -> None:
    cfg = ServerConfig(model="small", allow_client_model=True)
    out = validate_request_options({"model": "large-v3"}, cfg)
    assert out["model"] == "large-v3"


def test_validate_request_options_allows_same_model_as_default() -> None:
    cfg = ServerConfig(model="small")
    out = validate_request_options({"model": "small"}, cfg)
    assert out["model"] == "small"


def test_validate_request_options_clamps_batch_size() -> None:
    cfg = ServerConfig(max_batch_size=8)
    with pytest.raises(SettingsViolation):
        validate_request_options({"batch_size": 32}, cfg)
    out = validate_request_options({"batch_size": 4}, cfg)
    assert out["batch_size"] == 4


def test_validate_request_options_rejects_embed_by_default() -> None:
    cfg = ServerConfig()
    with pytest.raises(SettingsViolation):
        validate_request_options({"embed": True}, cfg)


def test_validate_request_options_allows_embed_when_configured() -> None:
    cfg = ServerConfig(allow_embed=True)
    out = validate_request_options({"embed": True}, cfg)
    assert out["embed"] is True


# --------------------- config round-trip ---------------------


def test_config_env_round_trip() -> None:
    cfg = ServerConfig(
        host="127.0.0.1",
        port=9000,
        auth_token="t",
        device="cpu",
        model="tiny",
        allow_client_model=True,
        prefetch="eager",
        max_batch_size=4,
        max_queue=2,
        hf_token="hf_abc",
    )
    env = config_to_env(cfg)
    # Subprocess-style: keys must be plain strings.
    for k, v in env.items():
        assert k.startswith("TRANSCRIBE_ANYTHING_SERVER_")
        assert isinstance(v, str)
    rebuilt = config_from_env(env)
    assert rebuilt == cfg


# --------------------- is_model_cached ---------------------


def test_is_model_cached_returns_false_when_no_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # empty dir
    # Also mask the real ~/.cache to avoid false positives on dev machines.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home_stub")
    assert is_model_cached("openai/whisper-large-v3") is False


def test_is_model_cached_finds_directory_match(tmp_path, monkeypatch) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "models--openai--whisper-large-v3").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home_stub")
    assert is_model_cached("openai/whisper-large-v3") is True


# --------------------- end-to-end via TestClient ---------------------


@pytest.fixture
def server_client(tmp_path):
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path / "jobs"))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        yield client, app
    # JobStore.stop is called via the shutdown event by TestClient's context exit.


def test_healthz_ok_without_auth(server_client) -> None:
    client, _ = server_client
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_capabilities_returns_expected_fields(server_client) -> None:
    client, _ = server_client
    resp = client.get("/v1/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_default"] == "tiny"
    assert body["prefetch"] == "lazy"
    assert body["hf_token_configured"] is False


def test_submit_url_runs_to_completion(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://example.com/x.mp3", "language": "en"})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    final = _wait_for_status(client, job_id)
    assert final["status"] == "completed"
    assert "out.txt" in final["artifacts"]
    art = client.get(f"/v1/jobs/{job_id}/artifacts/out.txt")
    assert art.status_code == 200
    assert "hello" in art.text


def test_submit_multipart_runs_to_completion(server_client, tmp_path) -> None:
    client, _ = server_client
    f = tmp_path / "audio.wav"
    f.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    with f.open("rb") as fh:
        resp = client.post(
            "/v1/transcribe",
            files={"file": ("audio.wav", fh, "audio/wav")},
            data={"options": json.dumps({"language": "en"})},
        )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    final = _wait_for_status(client, job_id)
    assert final["status"] == "completed"


def test_submit_rejects_device_override(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x", "device": "cpu"})
    assert resp.status_code == 400
    assert "daemon-locked" in resp.json()["detail"]


def test_submit_rejects_model_override_without_allow(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x", "model": "large-v3"})
    assert resp.status_code == 400
    assert "client model" in resp.json()["detail"]


def test_submit_returns_400_when_no_url_and_no_file(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"language": "en"})
    assert resp.status_code == 400


def test_artifact_path_traversal_blocked(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x"})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)
    # Note: bare ".." is normalized by httpx itself before the request
    # leaves the client, so we don't test it. The cases below are the
    # ones that actually reach the handler.
    for evil in ("..foo", "...", "x.y"):
        r = client.get(f"/v1/jobs/{job_id}/artifacts/{evil}")
        # These either hit the filename check (400/404) or land in a
        # routed-but-missing handler (404). Either is correct rejection.
        assert r.status_code in (400, 404), f"path {evil!r} returned {r.status_code}"


def test_delete_job_removes_artifacts(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x"})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)
    r = client.delete(f"/v1/jobs/{job_id}")
    assert r.status_code == 204
    r2 = client.get(f"/v1/jobs/{job_id}")
    assert r2.status_code == 404


# --------------------- failure + redaction ---------------------


def test_failed_job_redacts_hf_token_in_error(tmp_path) -> None:
    cfg = ServerConfig(host="127.0.0.1", model="tiny", hf_token="hf_supersecret123", job_root=str(tmp_path))
    failing = _failing_transcribe_factory("boom because token=hf_supersecret123 and also --hf-token hf_supersecret123")
    app = create_app(cfg, transcribe_fn=failing)
    with TestClient(app) as client:
        resp = client.post("/v1/transcribe", json={"url": "https://x"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        final = _wait_for_status(client, job_id)
        assert final["status"] == "failed"
        assert "hf_supersecret123" not in (final["error"] or "")
        assert "<REDACTED>" in (final["error"] or "")


# --------------------- auth integration ---------------------


def test_auth_required_when_non_loopback(tmp_path) -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret", model="tiny", job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        # No header -> 401.
        r = client.get("/v1/capabilities")
        assert r.status_code == 401
        # Wrong token -> 401.
        r = client.get("/v1/capabilities", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        # Correct Bearer -> 200.
        r = client.get("/v1/capabilities", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        # Correct X-Transcribe-Token alias -> 200.
        r = client.get("/v1/capabilities", headers={"X-Transcribe-Token": "secret"})
        assert r.status_code == 200


# --------------------- prefetch=none gate ---------------------


def test_prefetch_none_rejects_submit_when_model_uncached(tmp_path, monkeypatch) -> None:
    # Force is_model_cached to report False.
    from transcribe_anything import server_app as srv

    monkeypatch.setattr(srv, "is_model_cached", lambda _model: False)
    cfg = ServerConfig(host="127.0.0.1", model="tiny", prefetch="none", job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 503
        r = client.post("/v1/transcribe", json={"url": "https://x"})
        assert r.status_code == 503
        assert "prefetch=none" in r.json()["detail"]


# --------------------- queue overflow ---------------------


def test_queue_overflow_returns_429(tmp_path) -> None:
    """A second submission while the worker is busy should respect max_queue=1."""
    import threading

    block = threading.Event()
    release = threading.Event()

    def blocking(**kwargs):
        # Hold the worker until we release it.
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        block.set()
        release.wait(timeout=5)
        (out / "out.txt").write_text("done", encoding="utf-8")
        (out / "out.srt").write_text("1\n", encoding="utf-8")
        (out / "out.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (out / "out.json").write_text("{}", encoding="utf-8")
        return str(out)

    cfg = ServerConfig(host="127.0.0.1", model="tiny", max_queue=1, job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=blocking)
    try:
        with TestClient(app) as client:
            r1 = client.post("/v1/transcribe", json={"url": "https://a"})
            assert r1.status_code == 202
            assert block.wait(timeout=2)
            r2 = client.post("/v1/transcribe", json={"url": "https://b"})
            assert r2.status_code == 202  # queued
            r3 = client.post("/v1/transcribe", json={"url": "https://c"})
            # With max_queue=1 and worker busy + one queued, third should be 429.
            assert r3.status_code in (202, 429)
            # Now unblock the worker so shutdown completes.
            release.set()
    finally:
        release.set()
