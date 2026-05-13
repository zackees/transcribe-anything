"""Reference client for the RunPod transcribe-anything Serverless endpoint.

Usage:
    cp runpod/.env.example runpod/.env   # then fill in values
    set -a; source runpod/.env; set +a
    python runpod/client_example.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
"""

from __future__ import annotations

import json
import os
import sys

import requests


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.stderr.write(
            f"error: {name} is not set.\n"
            "Copy runpod/.env.example to runpod/.env, fill it in, then:\n"
            "  set -a; source runpod/.env; set +a\n"
        )
        sys.exit(2)
    return val


def transcribe_remote(
    url_or_file: str,
    *,
    model: str = "large-v3",
    flash: bool = True,
    batch_size: int = 8,
    hf_token: str | None = None,
    timeout_s: int = 1200,
) -> dict:
    api_key = _require_env("RUNPOD_API_KEY")
    endpoint_id = _require_env("RUNPOD_ENDPOINT_ID")

    payload = {
        "input": {
            "url_or_file": url_or_file,
            "model": model,
            "device": "insane",
            "task": "transcribe",
            "flash": flash,
            "batch_size": batch_size,
            "hf_token": hf_token,
        }
    }

    resp = requests.post(
        f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "COMPLETED":
        raise RuntimeError(f"RunPod job did not complete: {body}")
    return body["output"]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: client_example.py <url-or-file> [--hf-token <token>]")
        return 2
    url_or_file = sys.argv[1]
    hf_token = None
    if "--hf-token" in sys.argv:
        hf_token = sys.argv[sys.argv.index("--hf-token") + 1]
    result = transcribe_remote(url_or_file, hf_token=hf_token)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
