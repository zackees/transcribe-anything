"""Reference client for the RunPod transcribe-anything Serverless endpoint.

Usage:
    cp runpod/.env.example runpod/.env   # then fill in values
    set -a; source runpod/.env; set +a
    python runpod/client_example.py "https://example.com/podcast.mp3"

For URLs that RunPod can't fetch directly (YouTube, Spotify, listennotes,
Apple Podcasts pages, paywalled hosts), use --resolve to route through a
residential-IP host that pre-downloads and mirrors the audio:

    python runpod/client_example.py --resolve "https://lnns.co/<shortcode>"

The --resolve path needs these env vars:
    TRANSCRIBE_RESOLVER_HOST    SSH target, e.g. user@host
    TRANSCRIBE_RESOLVER_SCRIPT  Absolute path to resolve_and_host.py on the host
"""

from __future__ import annotations

import json
import os
import subprocess
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


def resolve_via_remote(url: str) -> str:
    """Run the resolver on a residential-IP host and return the hosted audio URL.

    Stdout of the remote script is the public URL. Stderr from the script
    streams through to our stderr for visibility.
    """
    host = _require_env("TRANSCRIBE_RESOLVER_HOST")
    script = _require_env("TRANSCRIBE_RESOLVER_SCRIPT")
    sys.stderr.write(f"[resolve] running {script} on {host} for: {url}\n")
    cmd = ["ssh", "-o", "ConnectTimeout=10", host, "python3", script, url]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    # Pass through resolver stderr so user can see its progress
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"resolver exited {proc.returncode}")
    resolved = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not resolved.startswith("http"):
        raise RuntimeError(f"resolver did not return a URL; got: {resolved!r}")
    sys.stderr.write(f"[resolve] resolved to: {resolved}\n")
    return resolved


def transcribe_remote(
    url_or_file: str,
    *,
    model: str = "large-v3",
    flash: bool = False,
    batch_size: int = 8,
    hf_token: str | None = None,
    poll_interval_s: int = 15,
    max_wait_s: int = 1800,
) -> dict:
    """Submit a transcription job via /run and poll /status until terminal.

    Uses async submit + polling instead of /runsync because runsync's 90 sec
    sync timeout is too short for any non-trivial audio (cold start + model
    download alone often exceed it). Polling handles cold starts and long
    audio gracefully up to max_wait_s.
    """
    import time

    api_key = _require_env("RUNPOD_API_KEY")
    endpoint_id = _require_env("RUNPOD_ENDPOINT_ID")
    headers = {"Authorization": f"Bearer {api_key}"}

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

    submit = requests.post(
        f"https://api.runpod.ai/v2/{endpoint_id}/run",
        headers=headers, json=payload, timeout=30,
    )
    submit.raise_for_status()
    job_id = submit.json()["id"]
    sys.stderr.write(f"[runpod] job_id: {job_id}\n")

    deadline = time.time() + max_wait_s
    last_status = None
    while time.time() < deadline:
        time.sleep(poll_interval_s)
        elapsed = int(max_wait_s - (deadline - time.time()))
        status = requests.get(
            f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}",
            headers=headers, timeout=30,
        )
        try:
            d = status.json()
        except Exception:
            continue
        s = d.get("status")
        if s != last_status:
            sys.stderr.write(f"[runpod] [{elapsed:4d}s] {s}\n")
            last_status = s
        if s == "COMPLETED":
            return d.get("output") or {}
        if s in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"RunPod job ended in {s}: {d.get('error', '')[:500]}")
    raise TimeoutError(f"RunPod job did not finish within {max_wait_s} sec")


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage: client_example.py [--resolve] <url-or-file> [--hf-token <token>]\n"
            "\n"
            "  --resolve         Route URL through TRANSCRIBE_RESOLVER_HOST first\n"
            "                    (use for YouTube, Spotify, lnns.co, etc.)\n"
            "  --hf-token TOKEN  Override env HF_TOKEN for this call\n"
        )
        return 2

    resolve = False
    hf_token: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--resolve":
            resolve = True
        elif a == "--hf-token":
            hf_token = args[i + 1]
            i += 1
        else:
            positional.append(a)
        i += 1
    if not positional:
        sys.stderr.write("error: missing URL or file argument\n")
        return 2
    url_or_file = positional[0]

    if resolve:
        url_or_file = resolve_via_remote(url_or_file)

    result = transcribe_remote(url_or_file, hf_token=hf_token)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
