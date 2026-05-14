# transcribe-anything on RunPod Serverless

Self-hostable transcription pipeline built on top of [`zackees/transcribe-anything`](https://github.com/zackees/transcribe-anything), with three ways to use it:

| Surface | Where it runs | Best for |
|---|---|---|
| **Webapp** (`webapp/`) | Tailscale-only on your VPS, paste-a-URL UI | Daily use, friends/team in the mesh |
| **CLI client** (`client_example.py`) | Anywhere with Python + the env vars | Scripting, cron, programmatic callers |
| **Direct RunPod API** | Any language | Lovable, Vercel, mobile, anything HTTP |

All three hit the same RunPod Serverless endpoint that runs `--device insane` (CUDA + pyannote diarization) and returns text + srt + vtt + json + **speaker.json**.

## Why this fork exists

Vanilla transcribe-anything wants a local NVIDIA GPU. Many of us don't have one. RunPod Serverless gives us per-second-billed GPU minutes that scale to zero — the perfect target. Getting there required four upstream patches (all documented in [`upstream-prs/`](upstream-prs/)), a residential-IP audio resolver to work around YouTube/Spotify datacenter blocks, and a small self-hosted webapp for daily use.

## Architecture

```
                Browser (any node on Tailscale)
                          │ HTTP
                          ▼
        ┌─────────────────────────────────────┐
        │  VPS :8050   FastAPI webapp         │
        │   ├─ static UI                      │
        │   └─ POST /api/jobs → background    │
        └──────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────────────────────┐
        │ if URL needs resolving (YouTube,    │
        │ Spotify, listennotes, paywalled):   │
        │                                     │
        ▼                                     │
   ┌─────────────────┐                        │
   │ TRIGGY          │ residential IP fetches │
   │ resolve_and_    │ via yt-dlp / custom    │
   │ host.py         │ scrapers, then scp's   │
   └──────────┬──────┘                        │
              │ scp                            │
              ▼                                │
   ┌─────────────────┐                        │
   │ VPS audio host  │ stable public URL      │
   │ /var/www/voice- │ at voice.vgh-usa.com/  │
   │  audio/         │ audio/transcribe-*.mp3 │
   └──────────┬──────┘                        │
              │◄────────────────────────────────
              │ url_or_file
              ▼
   ┌─────────────────────────────────────────┐
   │ RunPod Serverless GPU worker            │
   │  ├─ whisper-large-v3 inference          │
   │  ├─ pyannote/speaker-diarization-3.1    │
   │  └─ returns text/srt/vtt/json/speaker_json │
   └─────────────────────────────────────────┘
```

## Components

### `runpod/handler.py`
RunPod Serverless entrypoint. Wraps `transcribe_anything.transcribe()` and returns the standard 5-field output. Pre-warms the insane backend's iso-env at module import (see [Dockerfile](Dockerfile) note).

### `runpod/Dockerfile`
CUDA 12.8 + transcribe-anything + RunPod SDK. RunPod builds this directly from the GitHub fork on push — no Docker registry round-trip needed.

### `runpod/client_example.py`
Reference Python client. Submits via `/run`, polls `/status` until terminal. Supports `--resolve` to route the input URL through a residential-IP host first.

```bash
source runpod/load_secrets.sh     # one-time per shell
python3 runpod/client_example.py --resolve "https://lnns.co/<shortcode>"
```

### `runpod/resolve_and_host.py`
Residential-IP audio resolver. Runs on TRIGGY (or any non-datacenter host). Classifies the URL → fetches via yt-dlp / custom scraper / direct curl → mirrors to a public audio host via scp → prints the public URL on stdout. Reads its config from env / `~/.vgh.env`.

### `runpod/webapp/`
Single-process FastAPI + static-HTML webapp. Frontend uses Tailwind via CDN — no build step. Backend keeps in-memory job state and polls RunPod on behalf of the browser. Deployed to VPS as a systemd service.

### `runpod/upstream-prs/`
Three ready-to-submit PR descriptions for fixes we made in this fork. Targets `zackees/transcribe-anything` and `Vaibhavs10/insanely-fast-whisper`.

## Quick start

### 1. Deploy your RunPod Serverless endpoint

In the [RunPod console](https://www.console.runpod.io/serverless):

- **New Endpoint** → `Custom deployment` → `Deploy from GitHub`
- Repo: your fork of this repo, branch `runpod-integration`
- Dockerfile path: `runpod/Dockerfile`
- Build context: `.`
- GPU: 24 GB recommended (e.g. RTX 4090, L4, A5000)
- Active workers: **0** (scale-to-zero)
- Max workers: 2-3
- Idle timeout: 5 sec
- Execution timeout: 7200 sec (2 hr)
- Container disk: 30 GB
- Env vars (Secrets for HF_TOKEN):
  - `HF_TOKEN` = your HuggingFace token (must have read access to public gated repos; accept the pyannote/segmentation-3.0 + pyannote/speaker-diarization-3.1 agreements)
  - `HF_HOME` = `/root/.cache/huggingface`

After Create, note the **Endpoint ID** (`abc123...`) — that's `RUNPOD_ENDPOINT_ID`.

### 2. Get your credentials

- RunPod API key — https://www.console.runpod.io/user/settings → API Keys
- HuggingFace token — https://huggingface.co/settings/tokens (simple `Read` type)

### 3. Test the CLI

```bash
cp runpod/.env.example runpod/.env
# Fill in RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, HF_TOKEN
set -a; source runpod/.env; set +a

# A short direct-audio URL works without any resolver
python3 runpod/client_example.py "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"
```

Expected: ~30 sec end-to-end on a warm worker, JSON output with all 5 fields populated including `speaker_json`.

### 4. (Optional) Set up the resolver for blocked URLs

YouTube, Spotify share links, listennotes pages, and paywalled hosts don't work directly from RunPod's datacenter IPs. The resolver fixes this by fetching from a residential-IP host first.

On your residential host (Linux/macOS, anywhere with yt-dlp installable):

```bash
# Get the repo
git clone -b runpod-integration https://github.com/<your-user>/transcribe-anything
cd transcribe-anything

# Install yt-dlp via pipx for a current version
pipx install yt-dlp

# Configure where to upload audio (any host you can scp to)
cat >> ~/.vgh.env <<EOF
AUDIO_HOST_USER=root
AUDIO_HOST=your-audio-host.example.com
AUDIO_HOST_DIR=/var/www/audio
AUDIO_PUBLIC_PREFIX=https://audio.example.com
EOF

# Test
python3 runpod/resolve_and_host.py "https://lnns.co/<shortcode>"
# Should print: https://audio.example.com/transcribe-<uuid>.mp3
```

Then on your client host (e.g. MACKY), add to `~/.vgh.env`:

```
TRANSCRIBE_RESOLVER_HOST=user@your-residential-host
TRANSCRIBE_RESOLVER_SCRIPT=/path/to/transcribe-anything/runpod/resolve_and_host.py
```

And use the `--resolve` flag:

```bash
python3 runpod/client_example.py --resolve "https://lnns.co/<shortcode>"
```

### 5. (Optional) Self-host the webapp

```bash
# On a server with python3 + ssh access to your resolver host
git clone -b runpod-integration https://github.com/<your-user>/transcribe-anything /opt/transcribe-webapp-repo
cp -r /opt/transcribe-webapp-repo/runpod/webapp /opt/transcribe-webapp

python3 -m venv /opt/transcribe-webapp/venv
/opt/transcribe-webapp/venv/bin/pip install -r /opt/transcribe-webapp/requirements.txt

# Configure
cat > /etc/transcribe-webapp.env <<EOF
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=...
HF_TOKEN=...
TRANSCRIBE_RESOLVER_HOST=user@your-residential-host
TRANSCRIBE_RESOLVER_SCRIPT=/abs/path/to/resolve_and_host.py
TRANSCRIBE_BIND_HOST=0.0.0.0  # or your private IP
TRANSCRIBE_BIND_PORT=8050
EOF
chmod 600 /etc/transcribe-webapp.env

# Install systemd unit
sudo cp /opt/transcribe-webapp/transcribe-webapp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now transcribe-webapp

# Open in browser at http://<your-host>:8050/
```

## Input / output contract

### Request (POST to RunPod `/run`)

```json
{
  "input": {
    "url_or_file": "https://...",
    "model": "large-v3",
    "device": "insane",
    "task": "transcribe",
    "language": null,
    "hf_token": null,
    "initial_prompt": null,
    "batch_size": 8,
    "flash": false,
    "timestamp": "chunk"
  }
}
```

**Note on `flash`**: default `false`. The upstream insane env doesn't ship `flash_attn`, so `flash: true` errors at runtime with `ImportError: FlashAttention2`. To opt back in, add `flash-attn` to `src/transcribe_anything/insanley_fast_whisper_reqs.py` deps and rebuild.

### Response

```json
{
  "text": "full transcript ...",
  "srt": "1\n00:00:00,000 --> ...",
  "vtt": "WEBVTT\n\n...",
  "json": {"chunks": [...], "speakers": [...], "text": "..."},
  "speaker_json": [
    {"speaker": "SPEAKER_00", "text": "...", "timestamp": [0, 11], "reason": "beginning"},
    {"speaker": "SPEAKER_01", "text": "...", "timestamp": [11, 24], "reason": "speaker-switch"}
  ]
}
```

## Known limitations

- **YouTube URLs do not work** from RunPod's datacenter IPs. Use `--resolve` (CLI) or check the "Route through resolver" box (webapp). This is a universal cloud-GPU limitation, not a bug in this integration.
- **First request per cold worker pays ~30-60 sec env-install cost.** The insane backend's iso-env can't be pre-built into the Docker image because it requires CUDA at install time. Once a worker is warm, subsequent requests are fast.
- **Spotify direct/Apple Podcasts URLs** aren't resolved by the included resolver (only listennotes + yt-dlp-supported sources). Custom resolvers for those would be ~50-line additions to `resolve_and_host.py`.

## Cost

- ~$0.04-0.24 per hour of audio (24 GB RunPod Serverless GPU)
- $0 when idle (scale-to-zero)
- Build/deploy session: typically $0.30-1.00 of test runs

## Contributing upstream

We hit three issues during deployment, each worth contributing back. See [`runpod/upstream-prs/`](upstream-prs/) for ready-to-submit PR descriptions:

1. **`setuptools<82` build constraint** missing from insane env (zackees/transcribe-anything)
2. **`--hf-token` value leaks via OSError** (zackees/transcribe-anything)
3. **`diarize_audio` crashes on empty pyannote segments** (Vaibhavs10/insanely-fast-whisper)

## License

MIT. See [LICENSE](LICENSE). Derivative of [zackees/transcribe-anything](https://github.com/zackees/transcribe-anything), also MIT.
