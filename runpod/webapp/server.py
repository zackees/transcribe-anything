"""Self-hosted transcribe-anything webapp.

Single-process FastAPI service that:
  1. Accepts an input URL via POST /api/jobs
  2. Optionally pre-fetches the audio through a residential-IP resolver host
     (configurable via env) so YouTube / Spotify / paywalled URLs work
  3. Submits the resulting audio URL to a RunPod Serverless transcribe-anything
     endpoint
  4. Polls RunPod and updates an in-memory job state
  5. Returns the full transcript + diarization on GET /api/jobs/{job_id}

Serves the static frontend at /. Intended to run behind a Tailscale-only
listener; no app-level auth.

Environment variables (loaded from os.environ; also auto-loads $HOME/.vgh.env):
  RUNPOD_API_KEY                required
  RUNPOD_ENDPOINT_ID            required
  HF_TOKEN                      optional — passed to RunPod for diarization
  TRANSCRIBE_RESOLVER_HOST      required if --resolve is used
                                (SSH target, e.g. user@host)
  TRANSCRIBE_RESOLVER_SCRIPT    required if --resolve is used
                                (absolute path to resolve_and_host.py on host)
  TRANSCRIBE_BIND_HOST          default: 127.0.0.1  (override to Tailscale IP)
  TRANSCRIBE_BIND_PORT          default: 8050
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


def _load_vgh_env() -> None:
    """Best-effort loader for ~/.vgh.env (shell-style KEY=value)."""
    path = Path.home() / ".vgh.env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_vgh_env()


def _env(name: str, required: bool = False, default: str = "") -> str:
    val = os.environ.get(name, default)
    if required and not val:
        sys.stderr.write(f"FATAL: required env var {name} not set\n")
        sys.exit(2)
    return val


RUNPOD_API_KEY = _env("RUNPOD_API_KEY", required=True)
RUNPOD_ENDPOINT_ID = _env("RUNPOD_ENDPOINT_ID", required=True)
HF_TOKEN = _env("HF_TOKEN")
RESOLVER_HOST = _env("TRANSCRIBE_RESOLVER_HOST")
RESOLVER_SCRIPT = _env("TRANSCRIBE_RESOLVER_SCRIPT")
BIND_HOST = _env("TRANSCRIBE_BIND_HOST", default="127.0.0.1")
BIND_PORT = int(_env("TRANSCRIBE_BIND_PORT", default="8050"))

RUNPOD_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUNPOD_HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}


# --- Job state -------------------------------------------------------------

# In-memory job store. Keys: job_id (uuid). Values: dict of job state.
# Trade-off: jobs are lost on process restart. Acceptable for v1; swap for
# SQLite or Redis if persistence becomes important.
JOBS: dict[str, dict[str, Any]] = {}


def _mask(s: Any) -> Any:
    """Strip hf_ / rpa_ tokens from any string before exposing in API responses."""
    if isinstance(s, str):
        s = re.sub(r"hf_[A-Za-z0-9]{8,}", "hf_<REDACTED>", s)
        s = re.sub(r"rpa_[A-Za-z0-9]{8,}", "rpa_<REDACTED>", s)
    elif isinstance(s, dict):
        return {k: _mask(v) for k, v in s.items()}
    elif isinstance(s, list):
        return [_mask(x) for x in s]
    return s


# --- App -------------------------------------------------------------------

app = FastAPI(title="transcribe-anything", version="0.1.0")


class SubmitRequest(BaseModel):
    url: str
    use_resolver: bool = False
    model: str = "large-v3"


@app.post("/api/jobs")
async def submit(req: SubmitRequest) -> dict[str, str]:
    if not req.url.strip():
        raise HTTPException(400, "url is empty")
    if req.use_resolver and (not RESOLVER_HOST or not RESOLVER_SCRIPT):
        raise HTTPException(
            400,
            "use_resolver=true but TRANSCRIBE_RESOLVER_HOST/SCRIPT not configured",
        )
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "submitted",
        "input_url": req.url,
        "use_resolver": req.use_resolver,
        "model": req.model,
        "created_at": time.time(),
        "progress_message": "queued",
        "runpod_id": None,
        "result": None,
        "error": None,
    }
    asyncio.create_task(_process_job(job_id))
    return {"job_id": job_id, "status": "submitted"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return _mask(job)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "endpoint_id": RUNPOD_ENDPOINT_ID,
        "resolver_configured": bool(RESOLVER_HOST and RESOLVER_SCRIPT),
        "hf_token_configured": bool(HF_TOKEN),
        "jobs_in_memory": len(JOBS),
    }


# --- Pipeline --------------------------------------------------------------

async def _run_resolver(input_url: str) -> str:
    """SSH to RESOLVER_HOST and run RESOLVER_SCRIPT, returning the resolved URL."""
    cmd = [
        "ssh", "-o", "ConnectTimeout=10",
        RESOLVER_HOST, "python3", RESOLVER_SCRIPT, input_url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"resolver exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
        )
    last_line = stdout.decode(errors="replace").strip().splitlines()[-1] if stdout.strip() else ""
    if not last_line.startswith("http"):
        raise RuntimeError(f"resolver did not return a URL; got: {last_line!r}")
    return last_line


async def _runpod_submit(audio_url: str, model: str) -> str:
    payload = {
        "input": {
            "url_or_file": audio_url,
            "device": "insane",
            "model": model,
            "task": "transcribe",
            "flash": False,
            "batch_size": 8,
            "hf_token": HF_TOKEN or None,
        }
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{RUNPOD_BASE}/run", headers=RUNPOD_HEADERS, json=payload)
        r.raise_for_status()
        return r.json()["id"]


async def _runpod_status(runpod_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{RUNPOD_BASE}/status/{runpod_id}", headers=RUNPOD_HEADERS)
        r.raise_for_status()
        return r.json()


async def _process_job(job_id: str) -> None:
    job = JOBS[job_id]
    try:
        audio_url = job["input_url"]
        if job["use_resolver"]:
            job["status"] = "resolving"
            job["progress_message"] = "fetching audio through resolver host"
            audio_url = await _run_resolver(audio_url)
            job["resolved_url"] = audio_url

        job["status"] = "submitting"
        job["progress_message"] = "submitting to RunPod"
        runpod_id = await _runpod_submit(audio_url, job["model"])
        job["runpod_id"] = runpod_id

        job["status"] = "transcribing"
        job["progress_message"] = "RunPod is transcribing (this can take several minutes for long audio)"

        # Poll until terminal
        for _ in range(120):  # 120 * 15s = 30 min max
            await asyncio.sleep(15)
            d = await _runpod_status(runpod_id)
            s = d.get("status")
            if s == "COMPLETED":
                job["status"] = "completed"
                job["progress_message"] = "done"
                job["result"] = d.get("output") or {}
                job["completed_at"] = time.time()
                return
            if s in ("FAILED", "CANCELLED", "TIMED_OUT"):
                job["status"] = "failed"
                job["error"] = (d.get("error") or "")[:1500]
                job["progress_message"] = f"RunPod returned {s}"
                return
            # Update progress on intermediate states
            if s == "IN_PROGRESS":
                job["progress_message"] = "RunPod worker is processing"
            elif s == "IN_QUEUE":
                job["progress_message"] = "queued on RunPod"
        # Loop exited without terminal status — give up
        job["status"] = "failed"
        job["error"] = "exceeded 30 min poll window"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {str(e)[:500]}"


# --- Static frontend -------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


# --- CLI entrypoint --------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT, log_level="info")
