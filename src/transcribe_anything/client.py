"""
HTTP client used by the ``--remote`` CLI / Python-API path.

The client mirrors the local :func:`transcribe_anything.api.transcribe`
contract: the caller hands over a URL or file path and (eventually) gets
the same ``out.txt``/``out.srt``/``out.vtt``/``out.json`` artifact layout
in their ``output_dir``.

Uses ``httpx`` (synchronous client). ``httpx`` is in the base package
deps so client mode works out of the box.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx


class RemoteTranscriberError(Exception):
    """Raised when a remote daemon rejects or fails a transcription request."""


def _build_headers(token: Optional[str]) -> dict:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _collect_options(
    *,
    model: Optional[str],
    task: Optional[str],
    language: Optional[str],
    initial_prompt: Optional[str],
    align: bool,
    align_model: Optional[str],
    other_args: Optional[list[str]],
    embed: bool,
) -> dict:
    options: dict[str, Any] = {}
    if model is not None:
        options["model"] = model
    if task:
        options["task"] = task
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    if align:
        options["align"] = True
    if align_model:
        options["align_model"] = align_model
    if other_args:
        options["other_args"] = list(other_args)
    if embed:
        options["embed"] = True
    return options


def transcribe_remote(
    url_or_file: str,
    *,
    remote: str,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    align: bool = False,
    align_model: Optional[str] = None,
    other_args: Optional[list[str]] = None,
    embed: bool = False,
    poll_interval_seconds: float = 1.0,
    request_timeout_seconds: float = 30.0,
    job_timeout_seconds: float = 60 * 60 * 4,
) -> str:
    """Submit a transcription job to a remote daemon and download artifacts locally.

    Returns the absolute path to the output directory, matching the shape
    of :func:`transcribe_anything.api.transcribe`.
    """
    base_url = _normalize_base_url(remote)
    headers = _build_headers(token)
    options = _collect_options(
        model=model,
        task=task,
        language=language,
        initial_prompt=initial_prompt,
        align=align,
        align_model=align_model,
        other_args=other_args,
        embed=embed,
    )

    # Decide submission mode: URL or multipart file upload.
    is_url = url_or_file.startswith("http://") or url_or_file.startswith("https://") or url_or_file.startswith("ftp://")

    client = httpx.Client(timeout=request_timeout_seconds, headers=headers)
    try:
        if is_url:
            payload = {"url": url_or_file, **options}
            resp = client.post(f"{base_url}/v1/transcribe", json=payload)
        else:
            local = Path(url_or_file)
            if not local.is_file():
                raise RemoteTranscriberError(f"local file not found: {url_or_file}")
            with local.open("rb") as fh:
                files = {"file": (local.name, fh, "application/octet-stream")}
                data = {"options": json.dumps(options)} if options else {}
                resp = client.post(f"{base_url}/v1/transcribe", files=files, data=data)
        if resp.status_code >= 400:
            raise RemoteTranscriberError(
                f"daemon at {base_url} rejected submission: {resp.status_code} {resp.text}"
            )
        body = resp.json()
        job_id = body["job_id"]

        # Poll until terminal.
        deadline = time.time() + job_timeout_seconds
        last_status = None
        while True:
            if time.time() > deadline:
                raise RemoteTranscriberError(f"job {job_id} timed out after {job_timeout_seconds}s")
            jr = client.get(f"{base_url}/v1/jobs/{job_id}")
            if jr.status_code >= 400:
                raise RemoteTranscriberError(f"daemon returned {jr.status_code} fetching job status: {jr.text}")
            job = jr.json()
            status = job.get("status")
            if status != last_status:
                sys.stderr.write(f"remote job {job_id}: {status}\n")
                last_status = status
            if status == "completed":
                break
            if status == "failed":
                raise RemoteTranscriberError(f"daemon job {job_id} failed: {job.get('error')}")
            time.sleep(poll_interval_seconds)

        # Resolve output_dir similarly to api.transcribe's logic.
        if output_dir is None:
            base = Path(url_or_file).name
            stem = Path(base).stem if not is_url else (base or "remote")
            output_dir = f"text_{stem or 'remote'}"
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        artifacts = job.get("artifacts") or []
        if not artifacts:
            raise RemoteTranscriberError(f"daemon job {job_id} reported no artifacts")
        for name in artifacts:
            dest = out_path / name
            with client.stream("GET", f"{base_url}/v1/jobs/{job_id}/artifacts/{name}") as r:
                if r.status_code >= 400:
                    raise RemoteTranscriberError(
                        f"failed to download artifact {name}: {r.status_code} {r.text}"
                    )
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
        return str(out_path)
    finally:
        client.close()


def resolve_remote_and_token(
    *,
    remote_arg: Optional[str],
    token_arg: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Combine CLI flags + env vars to determine the active remote URL and token."""
    remote = remote_arg or os.environ.get("TRANSCRIBE_ANYTHING_REMOTE") or None
    token = token_arg or os.environ.get("TRANSCRIBE_ANYTHING_TOKEN") or None
    return remote, token
