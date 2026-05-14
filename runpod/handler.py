"""RunPod Serverless handler for transcribe-anything.

Wraps `transcribe_anything.transcribe()` and returns transcripts plus optional
speaker diarization JSON. Designed for `--device insane` on NVIDIA GPUs.

Cold-start pre-warm: the `--device insane` backend needs an isolated Python
env (created via iso-env). That env's installer requires a working CUDA stack,
so it cannot be built into the Docker image on a GPU-less builder. We
initialize it once per worker, at module import — that adds ~30-60 sec to the
first request on each cold worker, but subsequent requests on the same worker
are fast (within the idle-timeout window).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import runpod
from transcribe_anything import transcribe


def _prewarm_insane_env() -> None:
    """Pre-build the insane-mode isolated env on worker boot.

    No-op on subsequent imports (iso-env caches the env on disk). Skipped if
    the env var INSANE_PREWARM=0 is set (useful for CPU-only deployments).
    """
    if os.environ.get("INSANE_PREWARM") == "0":
        return
    try:
        subprocess.run(
            ["transcribe-anything-init-insane"],
            check=True,
            timeout=600,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        sys.stderr.write(
            f"WARNING: insane env pre-warm failed ({type(exc).__name__}): {exc}\n"
            "Insane-mode requests on this worker will likely fail; "
            "CPU mode will still work.\n"
        )


_prewarm_insane_env()


def _read_if_exists(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def _read_json_if_exists(path: Path) -> Any:
    text = _read_if_exists(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def handler(event: dict) -> dict:
    payload = event.get("input") or {}

    url_or_file = payload.get("url_or_file")
    if not url_or_file:
        return {"error": "Missing required field 'url_or_file'."}

    model = payload.get("model", "large-v3")
    device = payload.get("device", "insane")
    task = payload.get("task", "transcribe")
    language = payload.get("language")
    initial_prompt = payload.get("initial_prompt")

    hf_token = payload.get("hf_token") or os.environ.get("HF_TOKEN")

    other_args: list[str] = []
    if device == "insane":
        batch_size = payload.get("batch_size")
        if batch_size is not None:
            other_args += ["--batch-size", str(batch_size)]
        if payload.get("flash"):
            other_args += ["--flash", "True"]
        timestamp = payload.get("timestamp")
        if timestamp:
            other_args += ["--timestamp", str(timestamp)]
        for key in ("num_speakers", "min_speakers", "max_speakers"):
            val = payload.get(key)
            if val is not None:
                other_args += [f"--{key.replace('_', '-')}", str(val)]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        transcribe(
            url_or_file=url_or_file,
            output_dir=str(out_dir),
            model=model,
            task=task,
            language=language,
            device=device,
            hugging_face_token=hf_token,
            initial_prompt=initial_prompt,
            other_args=other_args or None,
        )

        return {
            "text": _read_if_exists(out_dir / "out.txt"),
            "srt": _read_if_exists(out_dir / "out.srt"),
            "vtt": _read_if_exists(out_dir / "out.vtt"),
            "json": _read_json_if_exists(out_dir / "out.json"),
            "speaker_json": _read_json_if_exists(out_dir / "speaker.json"),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
