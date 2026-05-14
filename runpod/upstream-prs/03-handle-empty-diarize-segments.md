# Upstream PR #3 — Handle empty pyannote segments in `diarize_audio`

**Target repo**: https://github.com/Vaibhavs10/insanely-fast-whisper (**NOT** zackees/transcribe-anything — this fix is in a different package)
**File touched**: `src/insanely_fast_whisper/utils/diarize.py`
**Source**: not yet committed in our fork; patch below

## Title

`diarize.py: handle empty pyannote segments without IndexError`

## Body

When the diarization pipeline (`pyannote.audio`) returns zero speaker segments — e.g., audio that is silent, contains music only, or has no recognizable speech — `diarize_audio` crashes with:

```
File ".../insanely_fast_whisper/utils/diarize.py", line 82, in diarize_audio
    prev_segment = cur_segment = segments[0]
IndexError: list index out of range
```

This is correct pyannote behavior (no speech → no segments), but it cascades into an unhelpful traceback for the caller. We should treat "zero segments" the same way we'd treat any "no speakers detected" case: return an empty result and let the caller proceed.

I hit this in production transcribing music samples through `--flash-attn 0 --hf-token …` and had to debug ~40 minutes before realizing the issue was upstream of any auth/scope/network failure.

## Repro

```bash
# Any audio with no speech (e.g. a piano recording)
insanely-fast-whisper \
    --file-name piano.wav \
    --device-id 0 \
    --model-name openai/whisper-large-v3 \
    --task transcribe \
    --transcript-path out.json \
    --hf-token <valid-token-with-pyannote-access> \
    --batch-size 8
```

Result: transcription succeeds (Whisper produces empty/hallucinated text), but `diarize_audio` crashes on `segments[0]` before the result is written.

## Diff

```diff
--- a/src/insanely_fast_whisper/utils/diarize.py
+++ b/src/insanely_fast_whisper/utils/diarize.py
@@ -78,6 +78,16 @@ def diarize_audio(diarizer_inputs, diarization_pipeline, num_speakers, min_speak
     segments = []
     for segment, track, label in diarization.itertracks(yield_label=True):
         segments.append({"segment": {"start": segment.start, "end": segment.end},
                          "track": track,
                          "label": label})
+
+    # No speakers detected (e.g. audio is silent, music-only, or otherwise
+    # contains no recognizable speech). Return an empty list rather than
+    # IndexError-ing on segments[0] below.
+    if not segments:
+        return []
+
     prev_segment = cur_segment = segments[0]
```

If you'd like richer behavior — e.g., emit a `[{"speaker": "UNKNOWN", "text": "(no speech detected)"}]` placeholder — happy to expand. The minimal return-empty is the smallest change that prevents the crash without changing the existing happy-path output shape.

## How to open the PR

```bash
git clone https://github.com/Vaibhavs10/insanely-fast-whisper
cd insanely-fast-whisper
git checkout -b fix/empty-diarize-segments
# apply the diff above (or copy/paste the if-not-segments block)
git commit -am "diarize.py: handle empty pyannote segments without IndexError"
git push origin fix/empty-diarize-segments
gh pr create --base main \
  --title "diarize.py: handle empty pyannote segments without IndexError" \
  --body-file path/to/03-handle-empty-diarize-segments.md
```

## Note for our fork

Until this lands upstream, the workaround is to NOT pass `--hf-token` when transcribing audio that may have no speech. That bypasses the diarization path entirely and avoids the crash.

Alternatively, we could vendor a patched `insanely_fast_whisper` into our fork's `insanley_fast_whisper_reqs.py` deps list, but that adds maintenance overhead vs. a small upstream PR.
