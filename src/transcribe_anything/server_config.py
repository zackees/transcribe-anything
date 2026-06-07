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

    def requires_auth(self) -> bool:
        return not _is_loopback(self.host)

    def validate(self) -> None:
        if self.requires_auth() and not self.auth_token:
            raise ValueError(
                "Refusing to bind to a non-loopback host without an auth token. "
                "Set --auth-token / --auth-token-env / --auth-token-file, or bind to 127.0.0.1."
            )
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
                    f"daemon does not allow client model overrides "
                    f"(configured model={config.model!r}, requested={value!r}). "
                    "Start the daemon with --allow-client-model to permit this."
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
                raise SettingsViolation(
                    f"batch_size={requested} exceeds daemon limit {config.max_batch_size}"
                )
            cleaned["batch_size"] = requested
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
        try:
            self._queue.put_nowait(job_id)
        except Full as exc:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise QueueFull("transcription queue is full") from exc
        return job

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
        except Exception as exc:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            redacted = _redact_secrets(f"{exc}\n{tb}", self.config.hf_token)
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = redacted


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
