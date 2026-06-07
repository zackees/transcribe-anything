"""
FastAPI-free core of the transcribe-anything daemon.

Lives here (not in ``server_app``) so the host-side launcher
(``cli_serve``) can import the configuration dataclass without FastAPI
installed. FastAPI / uvicorn only become required when ``server_app`` is
imported — and that import only happens inside the daemon iso-env or
when the test suite has FastAPI installed via ``requirements.testing.txt``.

This module also owns the pure logic that needs to be unit-testable
without spinning up the HTTP layer:

* :class:`ServerConfig` — daemon-side configuration (locked at startup).
* :func:`_redact_secrets` — HF token scrubber.
* :func:`validate_request_options` — settings-ownership enforcement.
* :class:`JobStore` — in-memory job registry + bounded queue + worker.
* :class:`WarmupRunner` — eager-prefetch background task.
* :func:`config_to_env` / :func:`config_from_env` — round-trip between
  a :class:`ServerConfig` instance and env vars (used to ferry config
  into the iso-env subprocess).
"""

import dataclasses
import hmac
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Optional

LOG = logging.getLogger("transcribe_anything.server")

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1"}

_HF_TOKEN_PATTERN = re.compile(r"(--hf[-_]token)(\s+|=)(\S+)", re.IGNORECASE)


def _redact_secrets(text: Optional[str], hf_token: Optional[str]) -> Optional[str]:
    """Strip the HF token and any --hf-token CLI arg from a string."""
    if text is None:
        return None
    if not text:
        return text
    text = _HF_TOKEN_PATTERN.sub(r"\1\2<REDACTED>", text)
    if hf_token:
        text = text.replace(hf_token, "<REDACTED>")
    return text


def _is_loopback(host: str) -> bool:
    return host.lower() in _LOOPBACK_HOSTS


@dataclass
class ServerConfig:
    """Daemon-side configuration. Locked at startup; clients can't change it."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    auth_token: Optional[str] = None
    device: Optional[str] = None
    model: Optional[str] = "small"
    allow_client_model: bool = False
    allow_embed: bool = False
    hf_token: Optional[str] = None
    prefetch: str = "lazy"  # "lazy" | "eager" | "none"
    max_batch_size: Optional[int] = None
    max_queue: int = 8
    max_upload_size_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
    artifact_ttl_seconds: int = 3600
    job_root: Optional[str] = None
    shutdown_grace_seconds: int = 60
    # Allow clients to register an outbound HTTP webhook on job completion.
    # Off by default — the daemon must explicitly opt in via --allow-webhooks
    # because outbound HTTP can be abused as a stepping stone to internal
    # services.
    allow_webhooks: bool = False
    webhook_timeout_seconds: float = 10.0
    # Realtime streaming (#122). v1 ships the WebSocket protocol skeleton
    # plus a pluggable streaming_fn; the production faster-whisper backend
    # is a follow-up PR. `allow_stream=False` keeps the endpoint closed for
    # daemons that don't need it (the canned fallback streaming_fn is for
    # tests/dev only and shouldn't be exposed by default).
    allow_stream: bool = False
    max_stream_duration_seconds: int = 60 * 60
    stream_decode_interval_ms: int = 200

    def requires_auth(self) -> bool:
        return not _is_loopback(self.host)

    def validate(self) -> None:
        if self.requires_auth() and not self.auth_token:
            raise ValueError("Refusing to bind to a non-loopback host without an auth token. " "Set --auth-token / --auth-token-env / --auth-token-file, or bind to 127.0.0.1.")
        if self.prefetch not in {"lazy", "eager", "none"}:
            raise ValueError(f"--prefetch must be one of lazy|eager|none, got {self.prefetch!r}")
        if self.max_queue < 1:
            raise ValueError("--max-queue must be >= 1")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    request: dict
    artifact_dir: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    artifacts: list = field(default_factory=list)

    def to_public_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "artifacts": list(self.artifacts),
        }


class QueueFull(Exception):
    """Raised when the bounded job queue can't accept another job."""


class SettingsViolation(ValueError):
    """Raised when a client request tries to override a daemon-locked setting."""


_DAEMON_LOCKED_FIELDS = ("device", "hf_token", "hugging_face_token", "flash", "prefetch")


def validate_request_options(options: dict, config: ServerConfig) -> dict:
    """Validate per-request options against the daemon's settings ownership table.

    Returns the normalized request body (input field + cleaned options).
    Raises :class:`SettingsViolation` on conflict.
    """
    cleaned: dict = {}
    for key, value in (options or {}).items():
        if value is None:
            continue
        if key in _DAEMON_LOCKED_FIELDS:
            raise SettingsViolation(f"setting {key!r} is daemon-locked and cannot be overridden per request")
        if key == "model":
            if not config.allow_client_model and value != config.model:
                raise SettingsViolation(
                    f"daemon does not allow client model overrides " f"(configured model={config.model!r}, requested={value!r}). " "Start the daemon with --allow-client-model to permit this."
                )
            cleaned["model"] = value
            continue
        if key == "embed":
            if not config.allow_embed and value:
                raise SettingsViolation("daemon does not permit --embed (start with --allow-embed)")
            cleaned["embed"] = bool(value)
            continue
        if key == "batch_size" and config.max_batch_size is not None:
            try:
                requested = int(value)
            except (TypeError, ValueError) as exc:
                raise SettingsViolation(f"batch_size must be an integer, got {value!r}") from exc
            if requested > config.max_batch_size:
                raise SettingsViolation(f"batch_size={requested} exceeds daemon limit {config.max_batch_size}")
            cleaned["batch_size"] = requested
            continue
        if key == "webhook_url":
            if not config.allow_webhooks:
                raise SettingsViolation("daemon does not permit webhook callbacks (start with --allow-webhooks)")
            if not isinstance(value, str) or not (value.startswith("http://") or value.startswith("https://")):
                raise SettingsViolation(f"webhook_url must be an http(s) URL, got {value!r}")
            cleaned["webhook_url"] = value
            continue
        cleaned[key] = value
    return cleaned


def is_model_cached(model: Optional[str]) -> bool:
    """Best-effort heuristic for ``--prefetch=none`` mode.

    Checks whether HuggingFace's hub cache has any directory whose name
    contains the model identifier. Not perfect — a partial download will
    still be reported as cached — but good enough to refuse the obvious
    "fresh container, no weights anywhere" case.
    """
    if not model:
        return False
    candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    needle = model.replace("/", "--").lower()
    for hub in candidates:
        if not hub.is_dir():
            continue
        for entry in hub.iterdir():
            if needle in entry.name.lower():
                return True
    return False


def _default_transcribe_fn(**kwargs: Any) -> str:
    """Lazy thin wrapper around ``transcribe_anything.api.transcribe``."""
    from transcribe_anything.api import transcribe  # local import: heavy

    return transcribe(**kwargs)


class JobStore:
    """In-memory job registry + bounded queue + single worker thread."""

    def __init__(self, config: ServerConfig, transcribe_fn: Optional[Callable[..., str]] = None) -> None:
        self.config = config
        self.transcribe_fn = transcribe_fn or _default_transcribe_fn
        self._jobs: dict = {}
        self._lock = threading.Lock()
        self._queue: Queue = Queue(maxsize=config.max_queue)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        # Lifetime job-status counters for /metrics. Keys mirror JobStatus
        # values; we count terminal transitions, not intermediate states.
        self._counts: dict = {s.value: 0 for s in JobStatus}

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run_worker, name="transcribe-worker", daemon=True)
        self._worker.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            pass
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout if timeout is not None else self.config.shutdown_grace_seconds)
        self._worker = None

    def submit(self, request: dict, artifact_dir: str) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            request=request,
            artifact_dir=artifact_dir,
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._counts[JobStatus.QUEUED.value] += 1
        try:
            self._queue.put_nowait(job_id)
        except Full as exc:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise QueueFull("transcription queue is full") from exc
        return job

    def snapshot_metrics(self) -> dict:
        """Counters + gauges for the /metrics endpoint."""
        with self._lock:
            in_flight = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
            counts = dict(self._counts)
        return {
            "counts_lifetime": counts,
            "in_flight": in_flight,
            "queued_now": queued,
            "queue_capacity": self.config.max_queue,
        }

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        try:
            shutil.rmtree(job.artifact_dir, ignore_errors=True)
        except OSError:
            pass
        return True

    def list_artifacts(self, job: Job) -> list:
        path = Path(job.artifact_dir)
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir() if p.is_file())

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except Empty:
                continue
            if job_id is None:
                break
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:
                continue
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        with self._lock:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._counts[JobStatus.RUNNING.value] += 1
        request = job.request
        try:
            self.transcribe_fn(
                url_or_file=request["input"],
                output_dir=job.artifact_dir,
                model=request.get("model") or self.config.model,
                task=request.get("task") or "transcribe",
                language=request.get("language"),
                device=self.config.device,
                hugging_face_token=self.config.hf_token,
                other_args=request.get("other_args"),
                initial_prompt=request.get("initial_prompt"),
                align=bool(request.get("align", False)),
                align_model=request.get("align_model"),
            )
            artifacts = self.list_artifacts(job)
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.artifacts = artifacts
                self._counts[JobStatus.COMPLETED.value] += 1
        except Exception as exc:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            redacted = _redact_secrets(f"{exc}\n{tb}", self.config.hf_token)
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = redacted
                self._counts[JobStatus.FAILED.value] += 1
        # Webhook fires after the job reaches a terminal state regardless of
        # outcome. Fire-and-forget on a daemon thread: a slow / wedged
        # webhook receiver MUST NOT delay the next job picking up the GPU.
        webhook_url = request.get("webhook_url")
        if webhook_url:
            t = threading.Thread(target=self._fire_webhook, args=(job, webhook_url), name=f"webhook-{job.job_id}", daemon=True)
            t.start()

    def _fire_webhook(self, job: Job, webhook_url: str) -> None:
        """POST the terminal job manifest to ``webhook_url``. Errors swallowed."""
        try:
            import httpx  # local import: keeps server_config FastAPI-free
        except ImportError:
            LOG.warning("webhook dispatch skipped: httpx not installed")
            return
        payload = job.to_public_dict()
        try:
            with httpx.Client(timeout=self.config.webhook_timeout_seconds) as client:
                client.post(webhook_url, json=payload)
        except Exception as exc:  # pylint: disable=broad-except
            redacted = _redact_secrets(str(exc), self.config.hf_token)
            LOG.warning("webhook POST to %s failed: %s", webhook_url, redacted)


class WarmupRunner:
    """Performs the eager-mode warmup transcription in a background thread."""

    def __init__(self, config: ServerConfig, transcribe_fn: Callable[..., str]) -> None:
        self.config = config
        self.transcribe_fn = transcribe_fn
        self._done = threading.Event()
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="transcribe-warmup", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._do_warmup()
        except Exception as exc:  # pylint: disable=broad-except
            self._error = _redact_secrets(str(exc), self.config.hf_token)
            LOG.warning("warmup failed: %s", self._error)
        finally:
            self._done.set()

    def _do_warmup(self) -> None:
        wav = _make_silent_wav(duration_seconds=1.0)
        try:
            with tempfile.TemporaryDirectory(prefix="ta-warmup-") as out_dir:
                self.transcribe_fn(
                    url_or_file=str(wav),
                    output_dir=out_dir,
                    model=self.config.model,
                    task="transcribe",
                    language=None,
                    device=self.config.device,
                    hugging_face_token=self.config.hf_token,
                )
        finally:
            try:
                wav.unlink()
            except OSError:
                pass

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def error(self) -> Optional[str]:
        return self._error


# ----------------------------------------------------------------------
# Realtime streaming (#122).
#
# This module owns the FastAPI-free skeleton: the ``StreamSession``
# serializer (one in-flight WS connection per daemon) and the pluggable
# ``StreamingTranscribeFn`` protocol. The actual WebSocket route lives in
# ``server_app.py`` so this module stays importable without FastAPI.
#
# The faster-whisper backend (acceptance criterion #1 in #122) is
# *deliberately* not in this PR — it needs its own iso-env, model-load
# warmup, GPU benchmarking, and tests against a real WAV. This PR ships
# the protocol + plumbing so that the real backend can be dropped in
# behind the same ``streaming_fn`` interface in a follow-up.
# ----------------------------------------------------------------------


# Wire-protocol close codes (private 4000-4999 range per RFC 6455).
WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_BUSY = 4429  # another stream is already in-flight
WS_CLOSE_DURATION_EXCEEDED = 4408
WS_CLOSE_INTERNAL = 4499
WS_CLOSE_NOT_ALLOWED = 4403  # daemon was not started with --allow-stream


class StreamCancelled(Exception):
    """Raised inside a streaming backend when the session is asked to abort."""


class StreamSession:
    """Serializes the one-streaming-connection-per-daemon invariant.

    Use as a context manager from the WebSocket handler. ``acquire()``
    returns False if another session is already live; the handler then
    closes the new connection with :data:`WS_CLOSE_BUSY`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: bool = False
        self._cancel = threading.Event()
        self._started_at: Optional[float] = None

    def acquire(self) -> bool:
        with self._lock:
            if self._active:
                return False
            self._active = True
            self._cancel.clear()
            self._started_at = time.time()
            return True

    def release(self) -> None:
        with self._lock:
            self._active = False
            self._started_at = None

    def cancel(self) -> None:
        """Signal the backend to abort at the next safe point."""
        self._cancel.set()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def runtime_seconds(self) -> Optional[float]:
        with self._lock:
            if self._started_at is None:
                return None
            return time.time() - self._started_at


def _canned_streaming_fn(audio_iter: Any, *, session: StreamSession, **_kwargs: Any):
    """Stand-in streaming backend.

    Yields a fixed scripted sequence of ``partial`` / ``final`` events,
    decoupled from the audio bytes (we just drain them to keep the
    socket flowing). Useful for tests and for end-to-end protocol
    validation in dev environments without a GPU.

    Real backends will replace this via ``create_app(streaming_fn=...)``.
    They are expected to honor :attr:`StreamSession.is_cancelled` between
    chunks and raise :class:`StreamCancelled` to abort cleanly.
    """
    scripted = [
        ("partial", "the", 1),
        ("partial", "the quick", 2),
        ("final", "the quick brown fox", 3),
        ("partial", "jumps", 4),
        ("final", "jumps over the lazy dog", 5),
    ]
    # Drain a few audio frames so the socket exercises both directions.
    drained = 0
    iterator = iter(audio_iter)
    for kind, text, rev in scripted:
        if session.is_cancelled:
            raise StreamCancelled("stream cancelled by client")
        try:
            next(iterator)
            drained += 1
        except StopIteration:
            pass
        yield {"type": kind, "text": text, "rev": rev}
    yield {"type": "metrics", "audio_frames_received": drained}


def _make_silent_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    """Synthesize a tiny silent WAV file for warmup. Returned path is unlinked by caller."""
    import struct
    import wave

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="ta-silent-")
    os.close(fd)
    nframes = int(duration_seconds * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(struct.pack("<h", 0) for _ in range(nframes)))
    return Path(path)


def check_auth(config: ServerConfig, authorization: Optional[str], x_transcribe_token: Optional[str]) -> bool:
    """Constant-time auth check. Returns True if request is authorized."""
    expected = config.auth_token
    if not expected:
        return not config.requires_auth()
    candidate: Optional[str] = None
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            candidate = parts[1].strip()
    if candidate is None and x_transcribe_token:
        candidate = x_transcribe_token.strip()
    if candidate is None:
        return False
    return hmac.compare_digest(candidate, expected)


def config_to_env(config: ServerConfig) -> dict:
    """Serialize a ServerConfig to env vars consumed by the iso-env runner."""
    out: dict = {}
    for k, v in asdict(config).items():
        if v is None:
            continue
        out[f"TRANSCRIBE_ANYTHING_SERVER_{k.upper()}"] = json.dumps(v)
    return out


def config_from_env(env) -> ServerConfig:
    """Inverse of :func:`config_to_env`. Missing keys fall back to dataclass defaults."""
    defaults = {f.name: f.default for f in dataclasses.fields(ServerConfig)}
    cfg: dict = {}
    for name, default in defaults.items():
        raw = env.get(f"TRANSCRIBE_ANYTHING_SERVER_{name.upper()}")
        if raw is None:
            cfg[name] = default if default is not dataclasses.MISSING else None
            continue
        try:
            cfg[name] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            cfg[name] = raw
    return ServerConfig(**cfg)
