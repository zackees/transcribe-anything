# RunPod Integration

Run [`transcribe-anything`](../README.md) on a [RunPod](https://docs.runpod.io/overview) Serverless GPU worker
so machines without an NVIDIA GPU can use the `--device insane` backend — the
only mode that produces `speaker.json` (diarization) and unlocks Flash
Attention 2 for top-tier throughput.

This directory is an **experiment scaffold**. Nothing here is deployed yet.

## Why this exists

| Backend | Hardware needed | Diarization (`speaker.json`) |
|---|---|---|
| `--device cpu` | Any | ❌ |
| `--device mlx` | Apple Silicon | ❌ |
| `--device insane` | NVIDIA CUDA, 8–24 GB VRAM | ✅ |

RunPod Serverless gives you per-second-billed `insane`-mode transcription
callable from anywhere via HTTPS.

## Files

| File | Purpose |
|---|---|
| `handler.py` | RunPod Serverless entrypoint. Wraps `transcribe_anything.transcribe()`. |
| `Dockerfile` | CUDA + transcribe-anything + RunPod SDK. Build and push to a registry, then reference from the Serverless endpoint. |
| `requirements.txt` | Pinned Python deps for the worker. |
| `client_example.py` | Reference caller that hits the Serverless endpoint. |
| `test_local.py` | Run the handler locally (CPU fallback, no RunPod runtime) for fast iteration. |
| `.env.example` | Template for the env vars the client needs. |
| `LICENSE` | MIT, matching upstream `transcribe-anything`. |

## Deploy

1. Build & push the image:
   ```bash
   docker build -f runpod/Dockerfile -t <registry>/transcribe-anything-runpod:latest .
   docker push <registry>/transcribe-anything-runpod:latest
   ```
2. In the RunPod console → Serverless → New Endpoint:
   - Container image: the image you just pushed
   - GPU type: 24 GB (RTX 4090 / L4 / A5000 fits `large-v3` with `--flash`)
   - Active workers: 0, Max workers: 2 (scale-to-zero)
   - Env var: `HF_TOKEN=<your-huggingface-token>` if you want diarization
3. Note the endpoint URL: `https://api.runpod.ai/v2/<endpoint-id>/runsync`

## Run the client

```bash
cp runpod/.env.example runpod/.env
# Edit runpod/.env and fill in RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID
set -a; source runpod/.env; set +a
python runpod/client_example.py "https://www.youtube.com/watch?v=..."
```

`runpod/.env` is gitignored — never commit it.

## Input / output contract

**Input** (Serverless `event.input`):
```json
{
  "url_or_file": "https://...",
  "model": "large-v3",
  "device": "insane",
  "task": "transcribe",
  "language": null,
  "hf_token": null,
  "initial_prompt": null,
  "batch_size": 8,
  "flash": true,
  "timestamp": "chunk"
}
```

**Output**:
```json
{
  "text": "full transcript ...",
  "srt": "1\n00:00:00,000 --> ...",
  "vtt": "WEBVTT\n\n...",
  "json": [ /* whisper segments */ ],
  "speaker_json": [ /* present only when hf_token supplied */ ]
}
```

## Cost notes

24 GB RunPod Serverless ≈ $0.0002–0.0005/sec. A 1-hour podcast on `large-v3` +
`--flash` typically transcribes in 3–8 minutes → **~$0.04–0.24 per hour of
audio**. Scale-to-zero means $0 when idle.

## Known limitations

**YouTube URLs do not work** from RunPod (or any datacenter-hosted) workers.
YouTube aggressively rate-limits and blocks requests from datacenter IPs, so
yt-dlp returns non-zero exit codes when invoked from the worker. Use direct
audio URLs instead:

- Podcast MP3 CDN links (Anchor, Libsyn, Megaphone, etc.) — work
- archive.org direct downloads — work
- Most non-YouTube media sites supported by yt-dlp — generally work
- YouTube — does not work; pre-download locally first and pass the file URL, or
  use a residential-IP proxy

This is a YouTube infrastructure limitation, not a bug in transcribe-anything
or this integration.

## License

MIT — see [LICENSE](./LICENSE). This directory is derivative work of
[zackees/transcribe-anything](https://github.com/zackees/transcribe-anything),
also MIT-licensed.
