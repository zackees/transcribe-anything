# transcribe-anything

[![MacOS_Tests](https://github.com/zackees/transcribe-anything/actions/workflows/test_macos.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_macos.yml)
[![Win_Tests](https://github.com/zackees/transcribe-anything/actions/workflows/test_win.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_win.yml)
[![Ubuntu_Tests](https://github.com/zackees/transcribe-anything/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_ubuntu.yml)
[![WhisperX_Tests](https://github.com/zackees/transcribe-anything/actions/workflows/test_whisperx.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_whisperx.yml)
[![Lint](https://github.com/zackees/transcribe-anything/actions/workflows/lint.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/lint.yml)

![image](https://github.com/zackees/transcribe-anything/assets/6856673/94bdd1fe-3225-438a-ac1b-09c81f1d4108)


### Every Whisper variant under one app w/ FULL OPTIMIZATIONS! Easiest install ever! Mac/Linux/Win

Over 1200+⭐'s because this program just works! Works great for windows and mac. This whisper front-end app is the only one to generate a `speaker.json` file which partitions the conversation by who doing the speaking.

[![Star History Chart](https://api.star-history.com/svg?repos=zackees/transcribe-anything&type=Date)](https://star-history.com/#zackees/transcribe-anything&Date)

## Sponsored by [Recall.ai](https://www.recall.ai)

*Supporting open-source transcription innovation since September 2025.*

[<img src="https://github.com/user-attachments/assets/1f92659b-5fd4-43ab-aedc-6adf4d219905" alt="Recall.ai banner" width="300" />](https://www.recall.ai)

🚀 **Why 1,000+ companies choose Recall.ai**
- **Accurate speaker diarization** — real participant names, not just “Speaker 1/2”
- **Ultra-low 200 ms latency** for real-time transcripts & audio
- **99.9% uptime SLA** with enterprise-grade security (SOC 2, HIPAA, ISO 27001)
- **One API for all meeting platforms** — no platform-specific integrations required

### New in 4.0! 🎉

**Three new backends, phoneme-precise word-level timestamps for the fast path, and read-only installs.** This is the biggest release since the `--device insane` debut. If you ran 3.2 and squinted at end-of-audio timestamps, this is the one to upgrade to.

#### 🚀 `--device insane-flash` — guaranteed FlashAttention2 on CUDA

Same blazing `insanely-fast-whisper` model path as `--device insane`, but in a separate isolated environment with **pinned FlashAttention2 wheels** for Windows x86_64, Linux x86_64, and Linux aarch64 (Python 3.11, `torch==2.7.0+cu128`, `flash-attn==2.8.3`). It verifies `flash_attn` and the compiled CUDA extension *before* transcription starts and fails early with platform diagnostics when no controlled wheel is available. No more sad-path silent fallbacks.

```bash
transcribe-anything video.mp4 --device insane-flash --batch-size 8
```

#### 🎯 `--align` — phoneme-precise word-level timestamps on `--device insane` / `--device insane-flash`

The HF-pipeline timestamp drift on long audio is **over**. Add `--align` and the transcript gets a WhisperX wav2vec2 forced-alignment post-pass: every chunk grows a `words: [{word, start, end, score}]` array and segment timestamps tighten to first/last-word boundaries. `out.srt` and `out.vtt` inherit the tightened bounds for free. Reuses the WhisperX iso-env, so no new deps in the insane env. Best-effort by design — unsupported language, env build failure, or runner crash falls back to the original output with a stderr warning, never breaks transcription.

```bash
# Phoneme-precise timestamps on the fast path:
transcribe-anything video.mp4 --device insane --align
transcribe-anything video.mp4 --device insane-flash --align

# Force a specific wav2vec2 aligner for languages outside the 41 defaults:
transcribe-anything video.mp4 --device insane --align --align_model facebook/wav2vec2-large-960h-lv60-self
```

#### 🎙️ `--device whisperx` — alignment + diarization + word timing as a first-class backend

[WhisperX](https://github.com/m-bain/whisperX) is now bundled as a parallel, additive backend (it does **not** replace `--device insane`). Built-in VAD chunking, wav2vec2 forced alignment, and pyannote diarization — all from one CLI. Use it when you want word-level timestamps and per-speaker labels from a single backend invocation.

```bash
transcribe-anything video.mp4 --device whisperx --diarize --hf_token <hf_xxxx>
```

Supports `--compute_type`, `--min_speakers` / `--max_speakers`, `--align_model`, `--no_align`, `--highlight_words`, `--vad_method`, `--chunk_size`.

#### 🌐 `--device sensevoice` — multilingual non-autoregressive, ~5x faster at comparable WER

New isolated-env backend wrapping FunASR's `iic/SenseVoiceSmall` model. **Non-autoregressive** — ~5x faster than `whisper-large-v3` at comparable word-error rate. Multilingual out of the box (`auto`/`zh`/`en`/`yue`/`ja`/`ko`/`nospeech`), built-in `fsmn-vad`, emotion detection, and event-tag postprocessing. Speaker diarization via `cam++` is opt-in with `--diarize`. Models pull from ModelScope by default; pass `--hub hf` for HuggingFace.

```bash
transcribe-anything video.mp4 --device sensevoice
transcribe-anything video.mp4 --device sensevoice --diarize --language zh
```

#### 📦 Read-only installs work now

Backend iso-env venvs and the bundled `static_ffmpeg` binary moved from inside the package directory (`<site-packages>/transcribe_anything/venv/...`) to the user cache directory (`<user_cache_dir>/transcribe-anything/...`). That unblocks Nix-store installs, OS-package installs, multi-user shared installs, baked-into-container installs, and `pip install --target` with a read-only mount. Override the location with `TRANSCRIBE_ANYTHING_CACHE_DIR=/somewhere/writable`.

> **One-time cost on upgrade:** existing 3.2 installs have a venv cache at the *old* path. Those caches are orphaned by this move, so the first run of each backend after upgrade re-downloads its dependencies (~10 GB for `--device insane`). No data loss — just the install-time wheel fetch, once.

#### ❄️ Native Nix flake — community contribution

```bash
nix run github:zackees/transcribe-anything -- <url-or-file> --device insane
nix shell github:zackees/transcribe-anything
nix build .#transcribe-anything
```

A complete [uv2nix](https://github.com/pyproject-nix/uv2nix)-based Nix flake is now part of the repo, contributed by community member [@eeedean](https://github.com/eeedean) (#68 → #104). One-line install for any Linux / macOS / NixOS user with Nix flakes enabled, the wrapper script puts `ffmpeg` and `uv` on PATH automatically, and the dev shell (`nix develop`) gives you an editable install with `yt-dlp` pre-installed. Pairs perfectly with the read-only-install support above — backend iso-envs land in your user cache, not the immutable Nix store.

#### ☁️ Cloud / Serverless (no local GPU)

If you don't have a local NVIDIA GPU, community member [@victorkjung](https://github.com/victorkjung) maintains a turnkey [RunPod Serverless deployment](https://github.com/victorkjung/transcribe-anything/tree/runpod-integration/runpod). Per-second-billed GPU minutes that scale to zero. See the [Cloud / Serverless](#cloud--serverless-no-local-gpu) section below.

#### 🔒 Security fix

`--hf-token` no longer leaks into stderr or the `OSError("Failed to execute ...")` traceback when the insane backend's subprocess fails. If you ran 3.2 on RunPod, Modal, or any other serverless host that surfaces stdout/stderr or exception messages in job-status APIs, **rotate your HuggingFace token** — older runs may have logged it. Going forward, the token is masked in both the `Running:` banner and the failure traceback. The subprocess itself still receives the real token.

#### 🙏 Thanks

`--device insane-flash`, `--device whisperx`, and the long-form timestamp regression suite were driven by community feedback through the issue tracker. `--device sensevoice` is wired up against FunASR ([FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)). `--align` reuses the wav2vec2 forced-alignment work from [m-bain/whisperX](https://github.com/m-bain/whisperX). Special thanks to [@aj47](https://github.com/aj47) for the MLX backend, [@victorkjung](https://github.com/victorkjung) for the RunPod Serverless deployment fork, and everyone who filed issues and PRs since 3.2.

### New in 3.2!

**Turbo Mac acceleration using the new [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) backend.**

This is a communinity contribution by https://github.com/aj47. On behalf of all the mac users, thank you!

#### MLX Backend details

  * 4x faster than the `mps` whisper backend.
  * Supports multiple languages (`mps` only supports english).
  * Supports custom vocabulary via `--initial_prompt`.
  
#### Usage


```bash
# Mac accelerated back-end
transcribe-anything "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device mlx
```

Special thank

### New in 3.1!

Mac acceleration option using the new [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) backend. Enable with `--device mlx`. Now supports multiple languages, custom vocabulary via `--initial_prompt`, and both transcribe/translate tasks. 10x faster than Whisper CPP, 4x faster than previous MLX implementations!

**Model Storage:** MLX models are now stored in `~/.cache/whisper/mlx_models/` for consistency with other backends, instead of cluttering your current working directory.

**GPU Accelerated Dockerfile**

Recently added in 3.0.10 is a GPU accelerated [Dockerfile](Dockerfile).

If you are are doing translations at scale, check out the sister project: [https://github.com/zackees/transcribe-everything](https://github.com/zackees/transcribe-everything).

You can pull the docker image like so:

`docker pull niteris/transcribe-anything`

## About

Easiest whisper implementation to install and use. Just install with `pip install transcribe-anything`. All whisper backends are executed in an isolated environment. GPU acceleration is _automatic_, using the _blazingly_ fast [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) as the backend for `--device insane`. CUDA users can choose `--device insane-flash` for a separate FlashAttention2-backed insane environment with pinned wheel artifacts. WhisperX is also available with `--device whisperx` for alignment, diarization, and word highlighting; it is additive and does not replace `--device insane`. This is the only tool to optionally produces a `speaker.json` file, representing speaker-assigned text that has been de-chunkified.

Hardware acceleration on Windows/Linux `--device insane`

MacArm acceleration when using `--device mlx` (now with multi-language support and custom vocabulary)

Input a local file or youtube/rumble url and this tool will transcribe it using Whisper AI into subtitle files and raw text.

Uses whisper AI so this is state of the art translation service - completely free. 🤯🤯🤯

Your data stays private and is not uploaded to any service.

The new version now has state of the art speed in transcriptions, thanks to the new backend `--device insane`, as well as producing a `speaker.json` file.

```bash
pip install transcribe-anything

# Basic usage - CPU mode (works everywhere, slower)
transcribe-anything "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# GPU accelerated (Windows/Linux)
transcribe-anything "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device insane

# GPU accelerated with guaranteed FlashAttention2 where supported
transcribe-anything "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device insane-flash

# Mac Apple Silicon accelerated
transcribe-anything "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device mlx

# Advanced options (see Advanced Options section below for full details)
transcribe-anything video.mp4 --device mlx --batch_size 16 --verbose
transcribe-anything video.mp4 --device insane-flash --batch-size 8
```

_python api_

```python
from transcribe_anything import transcribe

transcribe(
    url_or_file="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_dir="output_dir",
    task="transcribe",
    model="large",
    device="cuda"
)

# Full function signiture:
def transcribe(
    url_or_file: str,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,              # tiny,small,medium,large
    task: Optional[str] = None,               # transcribe or translate
    language: Optional[str] = None,           # auto detected if none, "en" for english...
    device: Optional[str] = None,             # cuda,cpu,insane,insane-flash,mlx,whisperx
    embed: bool = False,                      # Produces a video.mp4 with the subtitles burned in.
    hugging_face_token: Optional[str] = None, # For diarization/speaker.json on supported backends
    other_args: Optional[list[str]] = None,   # Other args to be passed to to the whisper backend
    initial_prompt: Optional[str] = None,     # Custom prompt for better recognition of specific terms
) -> str:

```

#### Fastest Transcription - Use `insane` mode with model `large-v3` + `batching`

This is by far the fastest combination. Experimental, it produces text that tends to be lower quality:

- Higher chance for repeated text patterns.
- Timestamps in the vtt/srt files become unaligned.

It's unclear if this is due to batching or `large-v3` itself. More testing is needed. If you do this then please let us know the results by filing a bug in the issues page.

Large batch sizes require more significant amounts of Nvidia GPU Ram. For a 12 GB card, it's been experimentally shown that batch-size=8 will work on all videos from an internally tested data lake.

#### Insanely fast on `cuda` platforms

If you pass in `--device insane` on a cuda platform then this tool will use this state of the art version of whisper: https://github.com/Vaibhavs10/insanely-fast-whisper, which is MUCH faster and has a pipeline for speaker identification (diarization) using the `--hf_token` option.

Compatible with Python 3.10 and above. Backends use an isolated environment with pinned requirements and python version.

Use `--device insane-flash` when you specifically want the FlashAttention2 path. It uses a separate isolated environment (`venv/insanely_fast_whisper_flash`) with pinned `flash-attn==2.8.3` wheel artifacts for supported CUDA tuples, injects `--flash True`, and verifies FlashAttention before transcription starts. Normal `--device insane` remains the default SDPA path and does not require `flash-attn`.

`insane-flash` currently supports controlled wheel candidates for Windows x86_64, Linux x86_64, and Linux aarch64 on Python 3.11 with `torch==2.7.0+cu128`. macOS is unsupported for this CUDA backend; use `--device mlx` on Apple Silicon.

To prebuild the flash environment, run `transcribe-anything-init-insane-flash`.

#### Speaker.json

When diarization is enabled with backend-specific options, such as `--hf_token` for `--device insane` or `--diarize --hf_token` for `--device whisperx`, then the output json will contain speaker info labeled as `SPEAKER_00`, `SPEAKER_01` etc. For licensing agreement reasons, you must get your own hugging face token if you want to enable this feature. Also there is an additional step to agree to the user policies for the `pyannote.audio` located here: https://huggingface.co/pyannote/segmentation-3.0. If you don't do this then you'll see runtime exceptions from `pyannote` when the `--hf_token` is used.

What's special to this app is that we also generate a `speaker.json` which is a de-chunkified version of the output json speaker section.

### speaker.json example:

```json
[
  {
    "speaker": "SPEAKER_00",
    "timestamp": [0.0, 7.44],
    "text": "for that. But welcome, Zach Vorhees. Great to have you back on. Thank you, Matt. Craving me back onto your show. Man, we got a lot to talk about.",
    "reason": "beginning"
  },
  {
    "speaker": "SPEAKER_01",
    "timestamp": [7.44, 33.52],
    "text": "Oh, we do. 2023 was the year that OpenAI released, you know, chat GPT-4, which I think most people would say has surpassed average human intelligence, at least in test taking, perhaps not in, you know, reasoning and things like that. But it was a major year for AI. I think that most people are behind the curve on this. What's your take of what just happened in the last 12 months and what it means for the future of human cognition versus machine cognition?",
    "reason": "speaker-switch"
  },
  {
    "speaker": "SPEAKER_00",
    "timestamp": [33.52, 44.08],
    "text": "Yeah. Well, you know, at the beginning of 2023, we had a pretty weak AI system, which was a chat GPT 3.5 turbo was the best that we had. And then between the beginning of last",
    "reason": "speaker-switch"
  }
]
```

Note that `speaker.json` is generated by diarization-capable backends such as `--device insane` and `--device whisperx`; standard `--device cuda` and `--device cpu` do not produce it.

#### `cuda` vs `insane`

Insane mode eats up a lot of memory and it's common to get out of memory errors while transcribing. For example a 3060 12GB nividia card produced out of memory errors are common for big content. If you experience this then pass in `--batch-size 8` or smaller. Note that any arguments not recognized by `transcribe-anything` are passed onto the backend transcriber.

Also, please don't use `distil-whisper/distil-large-v2`, it produces extremely bad stuttering and it's not entirely clear why this is. I've had to switch it out of production environments because it's so bad. It's also non-deterministic so I think that somehow a fallback non-zero temperature is being used, which produces these stutterings.

`cuda` is the original AI model supplied by openai. It's more stable but MUCH slower. It also won't produce a `speaker.json` file which looks like this:

`--embed`. This app will optionally embed subtitles directly "burned" into an output video.

# Install

This front end app for whisper boasts the easiest install in the whisper ecosystem thanks to [isolated-environment](https://pypi.org/project/isolated-environment/). You can simply install it with pip, like this:

```bash
pip install transcribe-anything
```

# Docker

We have a GPU accelerated [Dockerfile](Dockerfile). The default image prebuilds both CUDA insane backends, but avoids duplicating their full dependency stacks: `--device insane` and `--device insane-flash` share one FlashAttention-capable backend environment inside Docker. If you want the smallest possible image and can tolerate first-run backend setup, use `PREBUILD_BACKENDS=none`.

```bash
# Prebuilt Docker image for --device insane and --device insane-flash
docker build -t transcribe-anything .

# Lean image; backend envs are built on first use
docker build --build-arg PREBUILD_BACKENDS=none -t transcribe-anything:lean .
```

If you have extremely large batches of data you'd like to convert all at once then consider using the sister project [transcribe-everything](https://github.com/zackees/transcribe-everything) which operates on entire remote paths hierarchies.

# GPU Acceleration

GPU acceleration will be automatically enabled for windows and linux. Mac users can use `--device mlx` for hardware acceleration on Apple Silicon. `--device insane` may also work on Mac M1+ but has been less tested.

Windows/Linux:

- Use `--device insane`

Mac:

- Use `--device mlx`

# Cloud / Serverless (no local GPU)

Don't have a local NVIDIA GPU? Community member [@victorkjung](https://github.com/victorkjung) maintains a turnkey [RunPod Serverless deployment](https://github.com/victorkjung/transcribe-anything/tree/runpod-integration/runpod) of `transcribe-anything` that runs `--device insane` on per-second-billed GPU minutes that scale to zero. It ships a Dockerfile, RunPod handler, reference Python client, and an optional self-hosted webapp. This is a third-party fork, not an official deployment — issues and support live in that repo.

# Advanced Options and Backend-Specific Arguments

## Quick Reference

| Backend | Device Flag | Key Arguments | Best For |
|---------|-------------|---------------|----------|
| **MLX** | `--device mlx` | `--batch_size`, `--verbose`, `--initial_prompt` | Mac Apple Silicon |
| **Insanely Fast** | `--device insane` | `--batch-size`, `--hf_token`, `--timestamp`, `--align` | Windows/Linux GPU |
| **Insane Flash** | `--device insane-flash` | `--batch-size`, `--hf_token`, `--timestamp`, `--align` | CUDA GPU with verified FlashAttention2 |
| **WhisperX** | `--device whisperx` | `--compute_type`, `--diarize`, `--batch_size`, `--align_model` | Alignment, diarization, word timing |
| **SenseVoice** | `--device sensevoice` | `--diarize`, `--language`, `--hub` | FunASR/SenseVoice; multilingual (zh/en/yue/ja/ko), built-in VAD + emotion |
| **CPU** | `--device cpu` | Standard whisper args | Universal compatibility |

> **Note:** Each backend has different capabilities. MLX is optimized for Apple Silicon with a focused feature set. Insanely Fast uses a transformer-based architecture with specific options. `insane-flash` is the same backend family with verified FlashAttention2 dependencies. Both insane backends accept an opt-in `--align` flag that runs a WhisperX wav2vec2 forced-alignment post-pass on the transcript, replacing the HF pipeline's segment-level timestamps with phoneme-precise word-level timing (per-word `{word, start, end, score}` data in `out.json`, tightened segment bounds in `out.srt` / `out.vtt`). It reuses the WhisperX iso-env, so no new deps land in the insane env. WhisperX is an additive backend for alignment and diarization, not a replacement for `--device insane`. SenseVoice wraps FunASR's `iic/SenseVoiceSmall` model — multilingual (zh/en/yue/ja/ko), non-autoregressive, with built-in VAD; diarization (cam++) is opt-in via `--diarize`. Models download from ModelScope by default; pass `--hub hf` to use HuggingFace instead. CPU backend supports the full range of standard OpenAI Whisper arguments.

## Custom Prompts and Vocabulary

Whisper supports custom prompts to improve transcription accuracy for domain-specific vocabulary, names, or technical terms. This is especially useful when transcribing content with:

- Technical terminology (AI, machine learning, programming terms)
- Proper names (people, companies, products)
- Medical or scientific terms
- Industry-specific jargon

### Using Custom Prompts

#### Command Line

```bash
# Direct prompt
transcribe-anything video.mp4 --initial_prompt "The speaker discusses artificial intelligence, machine learning, and neural networks."

# Load prompt from file
transcribe-anything video.mp4 --prompt_file my_custom_prompt.txt
```

#### Python API

```python
from transcribe_anything import transcribe

# Direct prompt
transcribe(
    url_or_file="video.mp4",
    initial_prompt="The speaker discusses AI, PyTorch, TensorFlow, and deep learning algorithms."
)

# Load prompt from file
with open("my_prompt.txt", "r") as f:
    prompt = f.read()

transcribe(
    url_or_file="video.mp4",
    initial_prompt=prompt
)
```

#### Best Practices

- Keep prompts concise but comprehensive for your domain
- Include variations of terms (e.g., "AI", "artificial intelligence")
- Focus on terms that Whisper commonly misrecognizes
- Test with and without prompts to measure improvement

## MLX Backend Arguments (--device mlx)

The MLX backend supports additional arguments for fine-tuning performance:

### Available Options

```bash
# Adjust batch size for better performance/memory trade-off
transcribe-anything video.mp4 --device mlx --batch_size 24

# Enable verbose output for debugging
transcribe-anything video.mp4 --device mlx --verbose

# Use custom prompt for better recognition of specific terms
transcribe-anything video.mp4 --device mlx --initial_prompt "The speaker discusses AI, machine learning, and neural networks."
```

### MLX-Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 12 | Batch size for processing. Higher values use more memory but may be faster |
| `--verbose` | flag | false | Enable verbose output for debugging |
| `--initial_prompt` | string | None | Custom vocabulary/context prompt for better recognition |

### Supported Models

The MLX backend supports these whisper models optimized for Apple Silicon:
- `tiny`, `small`, `base`, `medium`, `large`, `large-v2`, `large-v3`
- Distilled models: `distil-small.en`, `distil-medium.en`, `distil-large-v2`, `distil-large-v3`

> **Note:** The MLX backend uses the lightning-whisper-mlx library which has a focused feature set optimized for Apple Silicon. Advanced whisper options like `--temperature` and `--word_timestamps` are not currently supported by this backend.

## Insanely Fast Whisper Arguments (--device insane)

The insanely-fast-whisper backend supports these specific options:

### Performance Options

```bash
# Adjust batch size (critical for GPU memory management)
transcribe-anything video.mp4 --device insane --batch-size 8

# Use different model variants
transcribe-anything video.mp4 --device insane --model large-v3

# Enable Flash Attention 2 for faster processing
transcribe-anything video.mp4 --device insane --flash True
```

### Speaker Diarization Options

```bash
# Enable speaker diarization with HuggingFace token
transcribe-anything video.mp4 --device insane --hf_token your_token_here

# Specify exact number of speakers
transcribe-anything video.mp4 --device insane --hf_token your_token --num-speakers 3

# Set speaker range
transcribe-anything video.mp4 --device insane --hf_token your_token --min-speakers 2 --max-speakers 5
```

### Timestamp Options

```bash
# Choose timestamp granularity
transcribe-anything video.mp4 --device insane --timestamp chunk  # default
transcribe-anything video.mp4 --device insane --timestamp word   # word-level
```

### Insanely Fast Whisper Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | 24 | Batch size for processing. Critical for GPU memory management |
| `--flash` | bool | false | Upstream pass-through flag. Prefer `--device insane-flash` for guaranteed FlashAttention2 setup |
| `--timestamp` | choice | chunk | Timestamp granularity: "chunk" or "word" |
| `--hf_token` | string | None | HuggingFace token for speaker diarization |
| `--num-speakers` | int | None | Exact number of speakers (cannot use with min/max) |
| `--min-speakers` | int | None | Minimum number of speakers |
| `--max-speakers` | int | None | Maximum number of speakers |
| `--diarization_model` | string | pyannote/speaker-diarization | Diarization model to use |

> **Note:** The insanely-fast-whisper backend uses a different architecture than standard OpenAI Whisper. It does NOT support standard whisper arguments like `--temperature`, `--beam_size`, `--best_of`, etc. These are specific to the OpenAI implementation. Use `--device insane-flash` instead of manually adding `--flash True` when you need FlashAttention2 to be installed and verified.

## WhisperX Backend Arguments (--device whisperx)

WhisperX supports forced alignment, word highlighting, VAD, and speaker diarization. Use it when you want those WhisperX-specific features; keep using `--device insane` for the insanely-fast-whisper backend.

### Common Options

```bash
# GPU memory/performance tuning
transcribe-anything video.mp4 --device whisperx --batch_size 16 --compute_type float16

# Main language/task arguments still apply
transcribe-anything video.mp4 --device whisperx --language en --task transcribe

# Speaker diarization
transcribe-anything video.mp4 --device whisperx --diarize --hf_token your_token --min_speakers 2 --max_speakers 5

# Alignment and word highlighting
transcribe-anything video.mp4 --device whisperx --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --highlight_words

# VAD/chunking controls
transcribe-anything video.mp4 --device whisperx --vad_method silero --chunk_size 30
```

### WhisperX Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--compute_type` | choice | backend default | Compute precision such as `float16`, `float32`, or `int8` |
| `--batch_size` | int | backend default | Batch size for transcription; lower it for GPU memory pressure |
| `--diarize` | flag | false | Enable speaker diarization |
| `--min_speakers` | int | None | Minimum number of speakers for diarization |
| `--max_speakers` | int | None | Maximum number of speakers for diarization |
| `--align_model` | string | auto | Alignment model override |
| `--no_align` | flag | false | Skip forced alignment |
| `--highlight_words` | bool | false | Highlight words in subtitle output when alignment is enabled |
| `--vad_method` | choice | backend default | Voice activity detector method |
| `--chunk_size` | int | backend default | Audio chunk size used by the backend |
| `--hf_token` | string | None | HuggingFace token for diarization |
| `--language` | string | auto | Main language argument; use when known |
| `--task` | choice | transcribe | Main task argument: `transcribe` or `translate` |

## CPU Backend Arguments (--device cpu)

The CPU backend uses the standard OpenAI Whisper implementation and supports many additional arguments:

### Standard Whisper Options

```bash
# Language and task options (also available as main arguments)
transcribe-anything video.mp4 --device cpu --language es --task translate

# Generation parameters
transcribe-anything video.mp4 --device cpu --temperature 0.1 --best_of 5 --beam_size 5

# Quality thresholds
transcribe-anything video.mp4 --device cpu --compression_ratio_threshold 2.4 --logprob_threshold -1.0

# Output formatting
transcribe-anything video.mp4 --device cpu --word_timestamps --highlight_words True

# Audio processing
transcribe-anything video.mp4 --device cpu --threads 4 --clip_timestamps "0,30"
```

> **Note:** The CPU backend supports most standard OpenAI Whisper arguments. These are passed through automatically and documented in the [OpenAI Whisper repository](https://github.com/openai/whisper).

### Batch Size Recommendations

**MLX Backend (`--device mlx`):**
- Default: 12
- Recommended range: 8-24
- Higher values for more VRAM, lower for less

**Insanely Fast Whisper (`--device insane`):**
- Default: 24
- Recommended for 8GB GPU: 4-8
- Recommended for 12GB GPU: 8-12
- Recommended for 24GB GPU: 16-24
- Start low and increase if no OOM errors

**Insane Flash (`--device insane-flash`):**
- Start with `--batch-size 4` or `--batch-size 8` on 8-12GB GPUs
- Increase batch size only after verifying SRT timing and memory stability
- Requires a supported CUDA FlashAttention wheel tuple; unsupported hosts fail before transcription

**WhisperX Backend (`--device whisperx`):**
- Start with `--batch_size 8` on smaller GPUs
- Increase batch size only if GPU memory is stable
- Use `--compute_type int8` when memory is tight

# Usage Examples

## Basic Usage

```bash
# Basic transcription
transcribe-anything "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Local file
transcribe-anything video.mp4
```

## Backend-Specific Examples

### MLX Backend (Mac Apple Silicon)

```bash
# Basic MLX usage
transcribe-anything video.mp4 --device mlx

# MLX with custom batch size and verbose output
transcribe-anything video.mp4 --device mlx --batch_size 16 --verbose

# MLX with custom prompt for technical content
transcribe-anything lecture.mp4 --device mlx --initial_prompt "The speaker discusses machine learning, neural networks, PyTorch, and TensorFlow."

# MLX with multiple options (using main arguments for language/task)
transcribe-anything video.mp4 --device mlx --batch_size 20 --verbose --task translate --language es
```

### Insanely Fast Whisper (GPU)

```bash
# Basic insane mode
transcribe-anything video.mp4 --device insane

# Insane mode with custom batch size (important for GPU memory)
transcribe-anything video.mp4 --device insane --batch-size 8

# Insane Flash mode with verified FlashAttention2
transcribe-anything video.mp4 --device insane-flash --batch-size 8

# Insane mode with speaker diarization
transcribe-anything video.mp4 --device insane --hf_token your_huggingface_token

# Insane mode with word-level timestamps and speaker diarization
transcribe-anything video.mp4 --device insane --timestamp word --hf_token your_token --num-speakers 3

# High-performance setup with all optimizations
transcribe-anything video.mp4 --device insane-flash --batch-size 12 --timestamp word
```

### WhisperX Backend

```bash
# Basic WhisperX usage
transcribe-anything video.mp4 --device whisperx

# WhisperX with alignment, word highlighting, and known language
transcribe-anything video.mp4 --device whisperx --language en --highlight_words

# WhisperX with speaker diarization
transcribe-anything video.mp4 --device whisperx --diarize --hf_token your_huggingface_token
```

### CPU Backend (Universal)

```bash
# CPU mode (works everywhere, slower)
transcribe-anything video.mp4 --device cpu

# CPU with custom model and language
transcribe-anything video.mp4 --device cpu --model medium --language fr --task transcribe
```

## Daemon Mode (`transcribe-anything serve` / `--remote`)

For batch workflows or shared GPU hosts, the cold-start cost (iso-env build, `torch` import, HF model load, CUDA context init) dominates wall-clock time of every CLI invocation. **Daemon mode** pays those costs once at startup; subsequent transcriptions submit over HTTP to the long-running FastAPI server and download artifacts back into the same `text_<file>/` layout you get from a local run.

Two deployment shapes:

| Mode | Bind | Auth | Use case |
|---|---|---|---|
| **Local** (default) | `127.0.0.1:8765` | none | Developer laptop, local batch script |
| **Public** | `--host 0.0.0.0` (any non-loopback) | **required** — daemon refuses to start without `--auth-token` | Trusted LAN, reverse-proxied deployment |

### Start the daemon

```bash
# Local, no auth, lazy model load on first request
transcribe-anything serve

# Lock to a backend + model and pre-warm them at startup
transcribe-anything serve --device insane --model large-v3 --prefetch eager

# Bind to all interfaces (auth required) — read the token from an env var
TRANSCRIBE_ANYTHING_TOKEN=$(openssl rand -hex 32) \
  transcribe-anything serve --host 0.0.0.0 --auth-token-env TRANSCRIBE_ANYTHING_TOKEN
```

The daemon **locks the backend, HF token, and prefetch policy at startup**. Per-request overrides of those fields are rejected with `400 daemon-locked`. By default the `model` is also locked to the daemon-configured default — pass `--allow-client-model` if you want clients to request different Whisper variants.

`--prefetch` modes:

- `lazy` *(default)* — daemon boots immediately; the first request pays the model download.
- `eager` — boot blocks `/healthz` until a synthetic 1-second warmup transcription completes. First real request is fast. Good for hosted deployments.
- `none` — refuse requests until weights are already cached. Pre-bake the cache in your container build.

### Use the CLI as a client

```bash
# Talk to a local daemon
transcribe-anything video.mp4 --remote http://127.0.0.1:8765

# Talk to a public daemon with auth
transcribe-anything "https://youtu.be/..." \
  --remote https://transcribe.example.com \
  --token "$TRANSCRIBE_ANYTHING_TOKEN"

# Or via env vars (works with every other transcribe-anything flag)
export TRANSCRIBE_ANYTHING_REMOTE=http://127.0.0.1:8765
export TRANSCRIBE_ANYTHING_TOKEN=...
transcribe-anything video.mp4
```

When `--remote` is set, the CLI uploads local files (or forwards URLs for the daemon to download), polls for completion, and writes the standard `out.txt` / `out.srt` / `out.vtt` / `out.json` files into `output_dir` — identical to a local invocation. If `--remote` is unset, behavior is unchanged.

### Python API

```python
from transcribe_anything.client import transcribe_remote

transcribe_remote(
    "video.mp4",
    remote="http://127.0.0.1:8765",
    output_dir="text_video",
    model="large-v3",
    language="en",
)
```

### Self-hosted with Docker + TLS reverse proxy

`examples/serve-compose.yml` runs the daemon behind a Caddy reverse proxy with automatic Let's Encrypt TLS:

```bash
export TRANSCRIBE_ANYTHING_TOKEN=$(openssl rand -hex 32)
export TRANSCRIBE_PUBLIC_HOST=transcribe.example.com
docker compose -f examples/serve-compose.yml up -d

# Then from any client
transcribe-anything video.mp4 \
  --remote "https://${TRANSCRIBE_PUBLIC_HOST}" \
  --token  "${TRANSCRIBE_ANYTHING_TOKEN}"
```

The compose file isolates the daemon to an internal network and only publishes 80/443 from Caddy — the daemon's own `:8765` is never exposed to the host. Named volumes preserve the HuggingFace model cache and per-backend iso-envs across container rebuilds.

If you prefer nginx, `examples/nginx.serve.conf` is a drop-in fragment with the right timeouts (4h `proxy_read_timeout`), `client_max_body_size 2g` to match `--max-upload-size`, and `proxy_buffering off` so the future SSE progress endpoint flushes events in real time.

### Realtime streaming (`WS /v1/stream`)

Opt-in by installing the streaming extras and launching with `--allow-stream`:

```bash
pip install 'transcribe-anything[server,stream]'
transcribe-anything serve --no-iso-env --allow-stream --max-stream-duration 3600
```

`[stream]` pulls in `faster-whisper`, `silero-vad`, `ctranslate2`, and `onnxruntime`. The daemon auto-detects them at startup and switches `WS /v1/stream` from the protocol-validation fallback to the real backend.

Pipe a live mic into the daemon from the client side:

```bash
arecord -f S16_LE -c 1 -r 16000 -t raw - | \
  transcribe-anything --remote ws://127.0.0.1:8765 --stream-in --model small.en
```

`--stream-in` reads PCM16-LE mono audio from stdin, opens the WebSocket, and prints `[partial]` / `[final]` lines as the daemon emits them. http(s):// URLs on `--remote` are auto-rewritten to ws(s)://.

Wire protocol on `WS /v1/stream`:

| Direction | Frame | Payload |
|---|---|---|
| C → S | text (first) | `{"type":"hello","model":"small.en","language":"en","sample_rate":16000,"encoding":"pcm16le"}` |
| C → S | binary | raw PCM16-LE mono audio at the declared sample rate |
| C → S | text (optional) | `{"type":"end_of_input"}` for graceful EOF, `{"type":"cancel"}` for immediate abort |
| S → C | text | `{"type":"ready"}`, then `{"type":"partial"|"final","text":"…","rev":N}` events, then `{"type":"done"}` |

`rev` increments monotonically. A `final` locks its span; later `partial`s never overwrite locked text. Close codes use the WebSocket private range (4401 unauthorized, 4429 busy — one streaming session per daemon, 4408 session duration exceeded, 4403 disabled).

Auth: same `Authorization: Bearer <token>` (or `X-Transcribe-Token`) as `/v1/*`.

Without the `[stream]` extras, `--allow-stream` falls back to a canned scripted generator that emits a fixed transcript regardless of audio — useful for protocol validation in CI but not a real transcriber. A startup warning fires in that case so the operator notices.

### Operational notes

- **Non-goals for v1:** multi-tenant identity, persistent job queue across restarts, multi-GPU scheduling. TLS termination is documented above via the reverse-proxy recipe.
- **HF token redaction** — tokens supplied via `--hf-token` at daemon startup are stripped from every client-visible error and never appear in job-status responses (extends the redaction from PR #93 to the HTTP layer).
- **Queue + concurrency** — the GPU is single-tenant per backend, so the daemon serializes work onto one worker. `--max-queue` (default 8) bounds the queue; overflow returns `429`.
- **Endpoints** — `POST /v1/transcribe`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/artifacts/{filename}`, `GET /v1/jobs/{id}/artifacts.zip` (all-artifacts bundle download), `DELETE /v1/jobs/{id}`, `GET /v1/capabilities`, `GET /healthz`, `GET /readyz`, `GET /metrics` (Prometheus text format — auth-protected). Auth header is `Authorization: Bearer <token>` (or `X-Transcribe-Token: <token>`).
- **Webhooks** — opt-in with `transcribe-anything serve --allow-webhooks`. Clients pass a `webhook_url` field on `POST /v1/transcribe`; the daemon POSTs the terminal job manifest (same JSON as `GET /v1/jobs/{id}`) once the job reaches `completed` or `failed`. Fire-and-forget — a slow webhook receiver never delays the next GPU-bound job. No signing in v1; assume the receiver also validates the network path (mTLS / VPN / private network).
- **In-process mode** — `pip install 'transcribe-anything[server]'` + `transcribe-anything serve --no-iso-env` skips the iso-env build and runs the daemon directly in your venv. Faster for dev / containers that already have FastAPI installed.

## Troubleshooting Common Issues

### `zsh: no matches found: https://...`

If you see this on macOS (or any host where zsh is the default shell):

```
$ transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ
zsh: no matches found: https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

…that's the shell, not `transcribe-anything`. zsh treats `?` and `*` in unquoted arguments as filename globs, and by default refuses to run the command if the glob matches nothing (its `NOMATCH` option). The CLI never sees the URL — the shell stops first. Fix any of these ways:

```bash
# 1. Quote the URL (recommended — works in every shell).
transcribe-anything "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 2. Or prefix with `noglob` (zsh-specific, one-shot).
noglob transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ

# 3. Or disable nomatch globally for the session.
setopt +o nomatch

# 4. Or escape the special characters by hand.
transcribe-anything https://www.youtube.com/watch\?v=dQw4w9WgXcQ
```

Every URL example in this README uses option 1.

### Out of Memory Errors

If you encounter GPU out-of-memory errors:

```bash
# Reduce batch size for MLX
transcribe-anything video.mp4 --device mlx --batch_size 8

# Reduce batch size for insane mode
transcribe-anything video.mp4 --device insane --batch-size 4

# Reduce batch size for WhisperX
transcribe-anything video.mp4 --device whisperx --batch_size 4 --compute_type int8

# Use smaller model
transcribe-anything video.mp4 --device insane --model small --batch-size 8
```

### Poor Quality Transcriptions

For better quality:

```bash
# Use larger model
transcribe-anything video.mp4 --device insane --model large-v3

# Enable Flash Attention 2 for better performance
transcribe-anything video.mp4 --device insane --flash True

# Use custom prompt for domain-specific content (works with supported backends)
transcribe-anything video.mp4 --initial_prompt "Medical terminology: diagnosis, treatment, symptoms, patient care"

# For CPU backend, you can use standard whisper quality options
transcribe-anything video.mp4 --device cpu --compression_ratio_threshold 2.0 --logprob_threshold -0.5
```

### Performance Optimization

For faster processing:

```bash
# Increase batch size (if you have enough GPU memory)
transcribe-anything video.mp4 --device mlx --batch_size 24
transcribe-anything video.mp4 --device insane --batch-size 16
transcribe-anything video.mp4 --device whisperx --batch_size 16

# Enable Flash Attention 2 for insane mode (significant speedup)
transcribe-anything video.mp4 --device insane --flash True --batch-size 16

# Use smaller model for speed
transcribe-anything video.mp4 --device insane --model small

# Use distilled models for even faster processing
transcribe-anything video.mp4 --device insane --model distil-whisper/large-v2 --flash True
```

Will output:

```
Detecting language using up to the first 30 seconds. Use `--language` to specify the language
Detected language: English
[00:00.000 --> 00:27.000]  We're no strangers to love, you know the rules, and so do I
[00:27.000 --> 00:31.000]  I've built commitments while I'm thinking of
[00:31.000 --> 00:35.000]  You wouldn't get this from any other guy
[00:35.000 --> 00:40.000]  I just wanna tell you how I'm feeling
[00:40.000 --> 00:43.000]  Gotta make you understand
[00:43.000 --> 00:45.000]  Never gonna give you up
[00:45.000 --> 00:47.000]  Never gonna let you down
[00:47.000 --> 00:51.000]  Never gonna run around and desert you
[00:51.000 --> 00:53.000]  Never gonna make you cry
[00:53.000 --> 00:55.000]  Never gonna say goodbye
[00:55.000 --> 00:58.000]  Never gonna tell a lie
[00:58.000 --> 01:00.000]  And hurt you
[01:00.000 --> 01:04.000]  We've known each other for so long
[01:04.000 --> 01:09.000]  Your heart's been aching but you're too shy to say it
[01:09.000 --> 01:13.000]  Inside we both know what's been going on
[01:13.000 --> 01:17.000]  We know the game and we're gonna play it
[01:17.000 --> 01:22.000]  And if you ask me how I'm feeling
[01:22.000 --> 01:25.000]  Don't tell me you're too much to see
[01:25.000 --> 01:27.000]  Never gonna give you up
[01:27.000 --> 01:29.000]  Never gonna let you down
[01:29.000 --> 01:33.000]  Never gonna run around and desert you
[01:33.000 --> 01:35.000]  Never gonna make you cry
[01:35.000 --> 01:38.000]  Never gonna say goodbye
[01:38.000 --> 01:40.000]  Never gonna tell a lie
[01:40.000 --> 01:42.000]  And hurt you
[01:42.000 --> 01:44.000]  Never gonna give you up
[01:44.000 --> 01:46.000]  Never gonna let you down
[01:46.000 --> 01:50.000]  Never gonna run around and desert you
[01:50.000 --> 01:52.000]  Never gonna make you cry
[01:52.000 --> 01:54.000]  Never gonna say goodbye
[01:54.000 --> 01:57.000]  Never gonna tell a lie
[01:57.000 --> 01:59.000]  And hurt you
[02:08.000 --> 02:10.000]  Never gonna give
[02:12.000 --> 02:14.000]  Never gonna give
[02:16.000 --> 02:19.000]  We've known each other for so long
[02:19.000 --> 02:24.000]  Your heart's been aching but you're too shy to say it
[02:24.000 --> 02:28.000]  Inside we both know what's been going on
[02:28.000 --> 02:32.000]  We know the game and we're gonna play it
[02:32.000 --> 02:37.000]  I just wanna tell you how I'm feeling
[02:37.000 --> 02:40.000]  Gotta make you understand
[02:40.000 --> 02:42.000]  Never gonna give you up
[02:42.000 --> 02:44.000]  Never gonna let you down
[02:44.000 --> 02:48.000]  Never gonna run around and desert you
[02:48.000 --> 02:50.000]  Never gonna make you cry
[02:50.000 --> 02:53.000]  Never gonna say goodbye
[02:53.000 --> 02:55.000]  Never gonna tell a lie
[02:55.000 --> 02:57.000]  And hurt you
[02:57.000 --> 02:59.000]  Never gonna give you up
[02:59.000 --> 03:01.000]  Never gonna let you down
[03:01.000 --> 03:05.000]  Never gonna run around and desert you
[03:05.000 --> 03:08.000]  Never gonna make you cry
[03:08.000 --> 03:10.000]  Never gonna say goodbye
[03:10.000 --> 03:12.000]  Never gonna tell a lie
[03:12.000 --> 03:14.000]  And hurt you
[03:14.000 --> 03:16.000]  Never gonna give you up
[03:16.000 --> 03:23.000]  If you want, never gonna let you down Never gonna run around and desert you
[03:23.000 --> 03:28.000]  Never gonna make you hide Never gonna say goodbye
[03:28.000 --> 03:42.000]  Never gonna tell you I ain't ready
```

## Api

```python
from transcribe_anything import transcribe

transcribe(
    url_or_file="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_dir="output_dir",
)
```

## Develop

Works for Ubuntu/MacOS/Win32(in git-bash)
This will create a virtual environment

```bash
> cd transcribe_anything
> ./install.sh
# Enter the environment:
> source activate.sh
```

The environment is now active and the next step will only install to the local python. If the terminal
is closed then to get back into the environment `cd transcribe_anything` and execute `source activate.sh`

## Required: Install to current python environment

- `pip install transcribe-anything`
  - The command `transcribe_anything` will magically become available.
- `transcribe_anything <YOUTUBE_URL>`

# Tech Stack

- OpenAI whisper
- insanely-fast-whisper
- WhisperX
- yt-dlp: https://github.com/yt-dlp/yt-dlp
- static-ffmpeg
  - github: https://github.com/zackees/static_ffmpeg
  - pypi: https://pypi.org/project/static-ffmpeg/

# Testing

- Every commit is tested for standard linters and a batch of unit tests.

## Updated version 2.3.0

`transcribe-anything` now works much better across different configurations and is now much faster. Why? I switched the environment isolation that I was using from my own homespun version built on top of `venv` to the AMAZING `uv` system. The biggest improvement is the runtime speed and re-installs. UV is just insane at how fast it is for checking the environment. Also it turns out that `uv` has strict package dependency checking which found a minor bug where a certain version of one of the `pytorch` dependencies was being constantly re-installed because of a dependency conflict that pip was apparently perfectly happy to never warn about. This manifested as certain packages being constantly re-installed with the previous version. `uv` identified this as an error immediately and was fixed.

The real reason behind `transcribe-anything`'s surprising popularity comes from the fact that it just works. And the reason for this is that I can isolate environments for different configurations and install them lazily. If you have the same problem then consider my other tool: https://github.com/zackees/iso-env

# Versions

- 4.0.0 (planned): Adds an optional WhisperX backend, enabled with `--device whisperx`. This backend runs in its own isolated environment and does not replace `--device insane`.
  - WhisperX supports forced alignment, VAD, word-level timing, word highlighting, and optional speaker diarization with `--diarize --hf_token`.
  - Output is normalized to the existing `out.srt`, `out.vtt`, `out.txt`, and `out.json` contract, with `speaker.json` emitted when speaker labels are available.
  - CUDA timing comparison on `tests/localfile/video.wav`: standard `--device cuda` produced 2 coarse SRT cues, while WhisperX running on CUDA produced 4 phrase-level cues and 20 word segments. The normalized transcript text matched, and both final SRT timestamps landed within 1s of the 10.027s audio duration. Use WhisperX when finer subtitle timing or alignment metadata is more important than the fastest transcription path.
- 3.0.7: Insane whisperer mode no longer prints out the srt file during transcription completion.
- 3.0.6: MacOS MLX mode fixed/improved
  - PR: https://github.com/zackees/transcribe-anything/pull/39
  - Thank you https://github.com/aj47!
- 3.0.5: A temp wav file was not being cleaned up, now it is.
- 3.1.0: Upgraded Mac-arm backend to [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx), enable with `--device mlx`. Now supports multiple languages, custom vocabulary via `--initial_prompt`, and both transcribe/translate tasks. 10x faster than Whisper CPP!
- 3.0.0: Implemented new Mac-arm accelerated [whisper-mps](https://github.com/AtomGradient/whisper-mps) backend, enable with `--device mps` (now `--device mlx`). Only does english, but is quite fast.
- 2.3.0: Swapped out the environment isolator. Now based on `uv`, should fix the missing dll's on some windows systems.
- 2.7.39: Fix `--hf-token` usage for insanely fast whisper backend.
- 2.7.37: Fixed breakage due to numpy 2.0 being released.
- 2.7.36: Fixed some ffmpeg dependencies.
- 2.7.35: All `ffmpeg` commands are now `static_ffmpeg` commands. Fixes issue.
- 2.7.34: Various fixes.
- 2.7.33: Fixes linux
- 2.7.32: Fixes mac m1 and m2.
- 2.7.31: Adds a warning if using python 3.12, which isn't supported yet in the backend.
- 2.7.30: adds --query-gpu-json-path
- 2.7.29: Made to json -> srt more robust for `--device insane`, bad entries will be skipped but warn.
- 2.7.28: Fixes bad title fetching with weird characters.
- 2.7.27: `pytorch-audio` upgrades broke this package. Upgrade to latest version to resolve.
- 2.7.26: Add model option `distil-whisper/distil-large-v2`
- 2.7.25: Windows (Linux/MacOS) bug with `--device insane` and python 3.11 installing wrong `insanely-fast-whisper` version.
- 2.7.22: Fixes `transcribe-anything` on Linux.
- 2.7.21: Tested that Mac Arm can run `--device insane`. Added tests to ensure this.
- 2.7.20: Fixes wrong type being returned when speaker.json happens to be empty.
- 2.7.19: speaker.json is now in plain json format instead of json5 format
- 2.7.18: Fixes tests
- 2.7.17: Fixes speaker.json nesting.
- 2.7.16: Adds `--save_hf_token`
- 2.7.15: Fixes 2.7.14 breakage.
- 2.7.14: (Broken) Now generates `speaker.json` when diarization is enabled.
- 2.7.13: Default diarization model is now pyannote/speaker-diarization-3.1
- 2.7.12: Adds srt_swap for line breaks and improved isolated_environment usage.
- 2.7.11: `--device insane` now generates a \*.vtt translation file
- 2.7.10: Better support for namespaced models. Trims text output in output json. Output json is now formatted with indents. SRT file is now printed out for `--device insane`
- 2.7.9: All SRT translation errors fixed for `--device insane`. All tests pass.
- 2.7.8: During error of `--device insane`, write out the error.json file into the destination.
- 2.7.7: Better error messages during failure.
- 2.7.6: Improved generation of out.txt, removes linebreaks.
- 2.7.5: `--device insane` now generates better conforming srt files.
- 2.7.3: Various fixes for the `insane` mode backend.
- 2.7.0: Introduces an `insanely-fast-whisper`, enable by using `--device insane`
- 2.6.0: GPU acceleration now happens automatically on Windows thanks to `isolated-environment`. This will also prevent
  interference with different versions of torch for other AI tools.
- 2.5.0: `--model large` now aliases to `--model large-v3`. Use `--model large-legacy` to use original large model.
- 2.4.0: pytorch updated to 2.1.2, gpu install script updated to same + cuda version is now 121.
- 2.3.9: Fallback to `cpu` device if `gpu` device is not compatible.
- 2.3.8: Fix --models arg which
- 2.3.7: Critical fix: fixes dependency breakage with open-ai. Fixes windows use of embedded tool.
- 2.3.6: Fixes typo in readme for installation instructions.
- 2.3.5: Now has `--embed` to burn the subtitles into the video itself. Only works on local mp4 files at the moment.
- 2.3.4: Removed `out.mp3` and instead use a temporary wav file, as that is faster to process. --no-keep-audio has now been removed.
- 2.3.3: Fix case where there spaces in name (happens on windows)
- 2.3.2: Fix windows transcoding error
- 2.3.1: static-ffmpeg >= 2.5 now specified
- 2.3.0: Now uses the official version of whisper ai
- 2.2.1: "test\_" is now prepended to all the different output folder names.
- 2.2.0: Now explictly setting a language will put the file in a folder with that language name, allowing multi language passes without overwriting.
- 2.1.2: yt-dlp pinned to new minimum version. Fixes downloading issues from old lib. Adds audio normalization by default.
- 2.1.1: Updates keywords for easier pypi finding.
- 2.1.0: Unknown args are now assumed to be for whisper and passed to it as-is. Fixes https://github.com/zackees/transcribe-anything/issues/3
- 2.0.13: Now works with python 3.9
- 2.0.12: Adds --device to argument parameters. This will default to CUDA if available, else CPU.
- 2.0.11: Automatically deletes files in the out directory if they already exist.
- 2.0.10: fixes local file issue https://github.com/zackees/transcribe-anything/issues/2
- 2.0.9: fixes sanitization of path names for some youtube videos
- 2.0.8: fix `--output_dir` not being respected.
- 2.0.7: `install_cuda.sh` -> `install_cuda.py`
- 2.0.6: Fixes twitter video fetching. --keep-audio -> --no-keep-audio
- 2.0.5: Fix bad filename on trailing urls ending with /, adds --keep-audio
- 2.0.3: GPU support is now added. Run the `install_cuda.sh` script to enable.
- 2.0.2: Minor cleanup of file names (no more out.mp3.txt, it's now out.txt)
- 2.0.1: Fixes missing dependencies and adds whisper option.
- 2.0.0: New! Now a front end for Whisper ai!

## Notes:

- Insanely Fast whisper for GPU
  - https://github.com/Vaibhavs10/insanely-fast-whisper
- Fast Whisper for CPU
  - https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file
- A better whisper CLI that supports more options but has a manual install.
  - https://github.com/ochen1/insanely-fast-whisper-cli/blob/main/requirements.txt
- Subtitles translator:
  - https://github.com/TDHM/Subtitles-Translator
- Forum post on how to avoid stuttering
  - https://community.openai.com/t/how-to-avoid-hallucinations-in-whisper-transcriptions/125300/23
- More stable transcriptions:
  - https://github.com/jianfch/stable-ts?tab=readme-ov-file
