# Upstream PR #1 — Apply `setuptools<82` build constraint to insane env

**Target repo**: https://github.com/zackees/transcribe-anything
**Source commit in our fork**: [`d78814f`](https://github.com/victorkjung/transcribe-anything/commit/d78814f)
**File touched**: `src/transcribe_anything/insanley_fast_whisper_reqs.py`

## Title (use as PR title)

`Apply setuptools<82 build constraint to the insane backend's iso-env`

## Body

The insane backend's iso-env fails to build `openai-whisper==20240930` because that package's `setup.py` imports `pkg_resources` without declaring it as a build dependency — and `pkg_resources` was removed from setuptools 82.

`whisper.py` (the CPU env builder) already has this fix:

```python
content_lines.append("[tool.uv]")
content_lines.append('build-constraint-dependencies = ["setuptools<82"]')
```

…but `insanley_fast_whisper_reqs.py` is missing it. As a result, every first invocation of `--device insane` on a fresh worker fails with a misleading `FileNotFoundError` on the cuda-detection tempfile (the iso-env tooling errors out before `cuda_available.py` can write its output).

This PR mirrors the same `setuptools<82` constraint over to the insane env's `get_environment()`. Verified end-to-end against a clean Docker build + a fresh worker spawn on a RunPod Serverless endpoint: `--device insane` now successfully builds its env, runs whisper-large-v3, and (with `--hf-token`) produces `speaker.json`.

### Repro before this patch

On any machine without a cached insane env:

```bash
pip install transcribe-anything
transcribe-anything <any-audio> --device insane
```

…hangs on env install → eventually FileNotFoundError on `/tmp/<tmpdir>/stdout.txt` because the openai-whisper build inside the iso-env crashed silently with `ModuleNotFoundError: No module named 'pkg_resources'`.

### After

Env install completes in ~60 sec on a cold worker (the existing 157 packages download dominates the time). Subsequent runs reuse the cached env and are fast.

## Diff

```diff
--- a/src/transcribe_anything/insanley_fast_whisper_reqs.py
+++ b/src/transcribe_anything/insanley_fast_whisper_reqs.py
@@ -84,6 +84,13 @@ def get_environment(has_nvidia: bool | None = None) -> IsoEnv:
         content_lines.append(f'  "{dep}",')
     content_lines.append("]")

+    # Constrain setuptools < 82 for build isolation because
+    # openai-whisper imports pkg_resources which was removed in setuptools 82.
+    # Mirrors the same fix already present in whisper.py for the CPU env.
+    content_lines.append("")
+    content_lines.append("[tool.uv]")
+    content_lines.append('build-constraint-dependencies = ["setuptools<82"]')
+
     if has_nvidia:
         content_lines.append("[tool.uv.sources]")
         content_lines.append("torch = [")
```

## How to open the PR

```bash
gh pr create \
  --repo zackees/transcribe-anything \
  --head victorkjung:runpod-integration \
  --base main \
  --title "Apply setuptools<82 build constraint to the insane backend's iso-env" \
  --body-file runpod/upstream-prs/01-setuptools-constraint-insane-env.md
```

Note: since our `runpod-integration` branch carries 9 unrelated commits, you may want to cherry-pick `d78814f` onto a fresh branch first:

```bash
git checkout -b upstream/insane-env-setuptools-constraint upstream/main
git cherry-pick d78814f
git push origin upstream/insane-env-setuptools-constraint
# then gh pr create --head victorkjung:upstream/insane-env-setuptools-constraint
```
