"""
FastAPI app for the transcribe-anything daemon.

This module imports FastAPI / uvicorn at module level — so it can ONLY be
imported in environments where those packages are installed:

* The daemon iso-env (built lazily by :mod:`server_reqs`).
* The test environment (FastAPI is in ``requirements.testing.txt``).

The FastAPI-free core (``ServerConfig``, redaction, settings validation,
``JobStore``, etc.) lives in :mod:`server_config` so the host-side
launcher (``cli_serve``) can import config without pulling FastAPI in.

Design constraints (issue #107):

* Local bind (loopback) needs no auth; non-loopback bind requires a token.
* The daemon LOCKS the backend (device + hf_token) at startup. Per-request
  overrides for those fields are rejected. ``model`` is daemon-default
  unless ``--allow-client-model`` was set.
* HF tokens supplied to the daemon must never leak into client responses
  (extends the redaction landed in #93 to the HTTP layer).
* Jobs run on a single background worker (the GPU is single-tenant per
  backend). Queue is bounded; overflow returns 429.
"""

import asyncio
import io
import json
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Iterable, Optional

from fastapi import (  # noqa: F401
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

# Re-export pure-logic surface so test modules and external callers keep
# importing from ``server_app`` after the file split.
from transcribe_anything.server_config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WS_CLOSE_BUSY,
    WS_CLOSE_DURATION_EXCEEDED,
    WS_CLOSE_INTERNAL,
    WS_CLOSE_NOT_ALLOWED,
    WS_CLOSE_UNAUTHORIZED,
    Job,
    JobStatus,
    JobStore,
    QueueFull,
    ServerConfig,
    SettingsViolation,
    StreamCancelled,
    StreamSession,
    WarmupRunner,
    _canned_streaming_fn,
    _default_transcribe_fn,
    _redact_secrets,
)
from transcribe_anything.server_config import check_auth as _check_auth  # noqa: F401
from transcribe_anything.server_config import (
    config_from_env,
    config_to_env,
    is_model_cached,
    validate_request_options,
)


def create_app(
    config: ServerConfig,
    *,
    transcribe_fn: Optional[Callable[..., str]] = None,
    job_root: Optional[Path] = None,
    streaming_fn: Optional[Callable[..., Iterable[dict]]] = None,
) -> FastAPI:
    """Build the FastAPI app. Pure factory — easy to unit-test."""
    config.validate()

    if job_root is None:
        job_root = Path(config.job_root) if config.job_root else Path(tempfile.mkdtemp(prefix="ta-jobs-"))
    job_root.mkdir(parents=True, exist_ok=True)

    store = JobStore(config, transcribe_fn=transcribe_fn)
    store.start()
    stream_session = StreamSession()
    # Streaming backend resolution:
    #   1. explicit streaming_fn= wins (tests, custom backends)
    #   2. when allow_stream is on, try the real faster-whisper backend
    #      (#124). It imports cleanly only if `transcribe-anything[stream]`
    #      is installed; otherwise we fall back to the canned protocol-
    #      validation backend with a warning so the operator notices.
    #   3. otherwise the canned backend (safe because allow_stream
    #      defaults to off, so the WS handler refuses connections).
    if streaming_fn is not None:
        active_streaming_fn = streaming_fn
    elif config.allow_stream:
        try:
            # Probe the heavy deps so we fail loud at startup, not
            # at first-connect.
            from transcribe_anything.stream_backend import (
                _lazy_imports,
                faster_whisper_streaming_fn,
            )

            _lazy_imports()
            active_streaming_fn = faster_whisper_streaming_fn
        except ImportError as exc:
            import logging as _logging

            _logging.getLogger("transcribe_anything.server").warning(
                "allow_stream is on but the faster-whisper backend is not available (%s). "
                "Falling back to the canned protocol-validation backend. Install "
                "`transcribe-anything[stream]` for real transcription.",
                exc,
            )
            active_streaming_fn = _canned_streaming_fn
    else:
        active_streaming_fn = _canned_streaming_fn

    warmup: Optional[WarmupRunner] = None
    if config.prefetch == "eager":
        warmup = WarmupRunner(config, transcribe_fn or _default_transcribe_fn)
        warmup.start()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            store.stop()

    app = FastAPI(
        title="transcribe-anything daemon",
        version="1",
        description="Persistent FastAPI daemon for transcribe-anything (issue #107).",
        lifespan=lifespan,
    )

    # Stash for tests / introspection.
    app.state.config = config
    app.state.store = store
    app.state.warmup = warmup
    app.state.job_root = job_root

    def _auth_dep(
        authorization: Optional[str] = Header(default=None),
        x_transcribe_token: Optional[str] = Header(default=None, alias="X-Transcribe-Token"),
    ) -> None:
        if _check_auth(config, authorization, x_transcribe_token):
            return
        raise HTTPException(status_code=401, detail="missing or invalid auth token")

    @app.get("/healthz")
    def healthz() -> Response:
        if warmup is not None and not warmup.done:
            return JSONResponse({"status": "warming"}, status_code=503)
        if warmup is not None and warmup.error:
            return JSONResponse({"status": "warmup_failed", "error": warmup.error}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")
    def readyz() -> Response:
        if config.prefetch == "none" and not is_model_cached(config.model):
            return JSONResponse(
                {
                    "status": "not_ready",
                    "detail": (f"prefetch=none and model {config.model!r} is not cached locally. " "Pre-warm the HuggingFace cache or restart with --prefetch lazy/eager."),
                },
                status_code=503,
            )
        if warmup is not None and not warmup.done:
            return JSONResponse({"status": "warming"}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.get("/metrics")
    def metrics(_: None = Depends(_auth_dep)) -> Response:
        snap = store.snapshot_metrics()
        lines: list = [
            "# HELP transcribe_anything_jobs_total Lifetime job count by terminal status.",
            "# TYPE transcribe_anything_jobs_total counter",
        ]
        for status, n in snap["counts_lifetime"].items():
            lines.append(f'transcribe_anything_jobs_total{{status="{status}"}} {n}')
        lines.extend(
            [
                "# HELP transcribe_anything_jobs_in_flight Currently-running jobs.",
                "# TYPE transcribe_anything_jobs_in_flight gauge",
                f"transcribe_anything_jobs_in_flight {snap['in_flight']}",
                "# HELP transcribe_anything_queue_depth Jobs currently queued.",
                "# TYPE transcribe_anything_queue_depth gauge",
                f"transcribe_anything_queue_depth {snap['queued_now']}",
                "# HELP transcribe_anything_queue_capacity Configured --max-queue.",
                "# TYPE transcribe_anything_queue_capacity gauge",
                f"transcribe_anything_queue_capacity {snap['queue_capacity']}",
            ]
        )
        body = "\n".join(lines) + "\n"
        return Response(content=body, media_type="text/plain; version=0.0.4")

    @app.get("/v1/capabilities")
    def capabilities(_: None = Depends(_auth_dep)) -> dict:
        warmup_state: Optional[dict] = None
        if warmup is not None:
            warmup_state = {"done": warmup.done, "error": warmup.error}
        return {
            "device": config.device,
            "model_default": config.model,
            "allow_client_model": config.allow_client_model,
            "allow_embed": config.allow_embed,
            "prefetch": config.prefetch,
            "max_batch_size": config.max_batch_size,
            "max_queue": config.max_queue,
            "max_upload_size_bytes": config.max_upload_size_bytes,
            "warmup": warmup_state,
            "hf_token_configured": bool(config.hf_token),
        }

    @app.post("/v1/transcribe", status_code=202)
    async def submit(
        request: Request,
        _: None = Depends(_auth_dep),
    ) -> dict:
        if config.prefetch == "none" and not is_model_cached(config.model):
            raise HTTPException(
                status_code=503,
                detail=(f"prefetch=none and model {config.model!r} is not cached locally. " "Pre-warm the HuggingFace cache or restart with --prefetch lazy/eager."),
            )

        content_type = (request.headers.get("content-type") or "").lower()
        artifact_dir = Path(tempfile.mkdtemp(prefix="ta-job-", dir=str(job_root)))

        try:
            if "multipart/form-data" in content_type:
                form = await request.form()
                upload = form.get("file")
                # Duck-type: starlette's UploadFile is a different class than
                # fastapi.UploadFile across versions; rely on the attributes
                # we actually use (.file, .filename) instead of isinstance.
                if upload is None or not hasattr(upload, "file") or not hasattr(upload, "filename"):
                    raise HTTPException(status_code=400, detail="multipart submission requires a 'file' field")
                opts_raw = form.get("options")
                options: dict = {}
                if isinstance(opts_raw, str) and opts_raw.strip():
                    try:
                        options = json.loads(opts_raw)
                    except json.JSONDecodeError as exc:
                        raise HTTPException(status_code=400, detail=f"options JSON invalid: {exc}") from exc
                input_path = _save_upload(upload, artifact_dir, config.max_upload_size_bytes)
            else:
                try:
                    body = await request.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail=f"request body must be JSON: {exc}") from exc
                if not isinstance(body, dict):
                    raise HTTPException(status_code=400, detail="request body must be a JSON object")
                url = body.get("url")
                if not url or not isinstance(url, str):
                    raise HTTPException(status_code=400, detail="JSON submission requires a 'url' field")
                options = {k: v for k, v in body.items() if k != "url"}
                input_path = url

            try:
                normalized = validate_request_options(options, config)
            except SettingsViolation as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            job_request = {"input": input_path, **normalized}
            try:
                job = store.submit(job_request, str(artifact_dir))
            except QueueFull as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
        except HTTPException:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            raise

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "status_url": f"/v1/jobs/{job.job_id}",
        }

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(_auth_dep)) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job.status == JobStatus.COMPLETED and not job.artifacts:
            job.artifacts = store.list_artifacts(job)
        return job.to_public_dict()

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, _: None = Depends(_auth_dep)) -> None:
        if not store.delete(job_id):
            raise HTTPException(status_code=404, detail="unknown job")
        return None

    @app.get("/v1/jobs/{job_id}/artifacts/{filename}")
    def get_artifact(job_id: str, filename: str, _: None = Depends(_auth_dep)) -> FileResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if "/" in filename or "\\" in filename or filename.startswith(".."):
            raise HTTPException(status_code=400, detail="invalid filename")
        path = Path(job.artifact_dir) / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(str(path), filename=filename)

    @app.get("/v1/jobs/{job_id}/artifacts.zip")
    def get_artifacts_zip(job_id: str, _: None = Depends(_auth_dep)) -> StreamingResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        artifact_dir = Path(job.artifact_dir)
        files = sorted(p for p in artifact_dir.iterdir() if p.is_file()) if artifact_dir.is_dir() else []
        if not files:
            raise HTTPException(status_code=404, detail="no artifacts available for this job")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="job-{job_id}.zip"'},
        )

    # ---- realtime streaming (#122) ----
    # WebSocket handler implementing the wire protocol from the issue:
    #   client → server: text "hello" frame, then binary PCM frames,
    #                    optional "end_of_input" or "cancel" text frames.
    #   server → client: text "ready", then any number of "partial" /
    #                    "final" / "metrics" frames, then "done".
    # The backend is pluggable via streaming_fn; the in-tree fallback is a
    # canned scripted generator (safe because allow_stream defaults to off).
    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        if not config.allow_stream:
            await ws.send_json({"type": "error", "code": "stream_disabled", "message": "daemon was not started with --allow-stream"})
            await ws.close(code=WS_CLOSE_NOT_ALLOWED)
            return

        if config.requires_auth():
            auth = ws.headers.get("authorization") or ""
            x_tok = ws.headers.get("x-transcribe-token")
            if not _check_auth(config, auth, x_tok):
                await ws.send_json({"type": "error", "code": "unauthorized"})
                await ws.close(code=WS_CLOSE_UNAUTHORIZED)
                return

        if not stream_session.acquire():
            await ws.send_json({"type": "error", "code": "busy", "message": "another stream is in-flight"})
            await ws.close(code=WS_CLOSE_BUSY)
            return

        try:
            # Wait for the hello frame (text). Anything else → error.
            try:
                hello_raw = await ws.receive_text()
            except WebSocketDisconnect:
                return
            try:
                hello = json.loads(hello_raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "code": "bad_hello", "message": "first frame must be JSON"})
                return
            if not isinstance(hello, dict) or hello.get("type") != "hello":
                await ws.send_json({"type": "error", "code": "bad_hello", "message": "first frame must be {'type':'hello',...}"})
                return

            await ws.send_json({"type": "ready"})

            # Pull audio frames + control frames into an async queue so the
            # backend generator (which is sync) can iterate them without
            # blocking the event loop.
            audio_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
            input_done = asyncio.Event()

            async def _pump_inputs() -> None:
                deadline = stream_session.runtime_seconds
                while not input_done.is_set():
                    try:
                        msg = await ws.receive()
                    except WebSocketDisconnect:
                        stream_session.cancel()
                        input_done.set()
                        return
                    if "bytes" in msg and msg["bytes"] is not None:
                        await audio_queue.put(msg["bytes"])
                    elif "text" in msg and msg["text"] is not None:
                        try:
                            ctrl = json.loads(msg["text"])
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(ctrl, dict):
                            continue
                        if ctrl.get("type") == "end_of_input":
                            input_done.set()
                            return
                        if ctrl.get("type") == "cancel":
                            stream_session.cancel()
                            input_done.set()
                            return
                    # Hard cap on session duration to keep a misbehaving
                    # client from pinning the GPU forever.
                    runtime = stream_session.runtime_seconds
                    if runtime is not None and runtime > config.max_stream_duration_seconds:
                        await ws.send_json({"type": "error", "code": "duration_exceeded"})
                        await ws.close(code=WS_CLOSE_DURATION_EXCEEDED)
                        stream_session.cancel()
                        input_done.set()
                        return
                    _ = deadline  # currently unused; placeholder for adaptive backpressure

            pumper = asyncio.create_task(_pump_inputs())

            def _audio_iterable():
                """Non-blocking generator over the audio queue.

                Yields whatever PCM frames are currently buffered, then
                returns. The backend is expected to iterate this once per
                decode cycle — looping forever inside it would block the
                sync backend thread waiting on the async pumper.
                """
                while True:
                    if stream_session.is_cancelled:
                        return
                    try:
                        chunk = audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    yield chunk

            # Drive the (sync) streaming backend in a thread so it doesn't
            # block the event loop. Forward each emitted event over the
            # socket as a text frame.
            loop = asyncio.get_running_loop()
            queue_out: asyncio.Queue = asyncio.Queue()

            def _run_backend() -> None:
                try:
                    for event in active_streaming_fn(_audio_iterable(), session=stream_session, config=config, hello=hello):
                        asyncio.run_coroutine_threadsafe(queue_out.put(event), loop)
                except StreamCancelled:
                    pass
                except Exception as exc:  # pylint: disable=broad-except
                    redacted = _redact_secrets(str(exc), config.hf_token)
                    asyncio.run_coroutine_threadsafe(queue_out.put({"type": "error", "code": "backend_failure", "message": redacted}), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue_out.put({"type": "done"}), loop)

            backend_thread = asyncio.get_event_loop().run_in_executor(None, _run_backend)

            try:
                while True:
                    event = await queue_out.get()
                    try:
                        await ws.send_json(event)
                    except (WebSocketDisconnect, RuntimeError):
                        stream_session.cancel()
                        break
                    if event.get("type") == "done":
                        break
            finally:
                input_done.set()
                pumper.cancel()
                try:
                    await backend_thread
                except Exception:  # pylint: disable=broad-except
                    pass

            try:
                await ws.close()
            except RuntimeError:
                pass
        except Exception as exc:  # pylint: disable=broad-except
            redacted = _redact_secrets(str(exc), config.hf_token)
            try:
                await ws.send_json({"type": "error", "code": "internal", "message": redacted})
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                await ws.close(code=WS_CLOSE_INTERNAL)
            except RuntimeError:
                pass
        finally:
            stream_session.release()

    # Stash for tests / introspection.
    app.state.stream_session = stream_session

    return app


def _save_upload(upload, dest_dir: Path, max_bytes: int) -> str:
    """Stream an UploadFile to ``dest_dir``, enforcing ``max_bytes``."""
    filename = getattr(upload, "filename", None) or "upload.bin"
    safe = Path(filename).name
    out = dest_dir / "_input" / safe
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out, "wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                fh.close()
                try:
                    out.unlink()
                except OSError:
                    pass
                raise HTTPException(status_code=413, detail=f"upload exceeds max-upload-size {max_bytes} bytes")
            fh.write(chunk)
    return str(out)


def run_server(config: ServerConfig) -> None:
    """Start uvicorn on the FastAPI app. Called by the iso-env runner."""
    import uvicorn  # type: ignore

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
