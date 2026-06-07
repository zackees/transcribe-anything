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

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: F401
from fastapi.responses import FileResponse, JSONResponse, Response

# Re-export pure-logic surface so test modules and external callers keep
# importing from ``server_app`` after the file split.
from transcribe_anything.server_config import (  # noqa: F401
    DEFAULT_HOST,
    DEFAULT_PORT,
    Job,
    JobStatus,
    JobStore,
    QueueFull,
    ServerConfig,
    SettingsViolation,
    WarmupRunner,
    _default_transcribe_fn,
    _redact_secrets,
    check_auth as _check_auth,
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
) -> FastAPI:
    """Build the FastAPI app. Pure factory — easy to unit-test."""
    config.validate()

    if job_root is None:
        job_root = Path(config.job_root) if config.job_root else Path(tempfile.mkdtemp(prefix="ta-jobs-"))
    job_root.mkdir(parents=True, exist_ok=True)

    store = JobStore(config, transcribe_fn=transcribe_fn)
    store.start()

    warmup: Optional[WarmupRunner] = None
    if config.prefetch == "eager":
        warmup = WarmupRunner(config, transcribe_fn or _default_transcribe_fn)
        warmup.start()

    app = FastAPI(
        title="transcribe-anything daemon",
        version="1",
        description="Persistent FastAPI daemon for transcribe-anything (issue #107).",
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
                    "detail": (
                        f"prefetch=none and model {config.model!r} is not cached locally. "
                        "Pre-warm the HuggingFace cache or restart with --prefetch lazy/eager."
                    ),
                },
                status_code=503,
            )
        if warmup is not None and not warmup.done:
            return JSONResponse({"status": "warming"}, status_code=503)
        return JSONResponse({"status": "ready"})

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
                detail=(
                    f"prefetch=none and model {config.model!r} is not cached locally. "
                    "Pre-warm the HuggingFace cache or restart with --prefetch lazy/eager."
                ),
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

    @app.on_event("shutdown")
    def _shutdown() -> None:
        store.stop()

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
