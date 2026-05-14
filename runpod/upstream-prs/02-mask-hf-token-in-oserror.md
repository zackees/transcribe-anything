# Upstream PR #2 — Mask `--hf-token` in subprocess OSError + log

**Target repo**: https://github.com/zackees/transcribe-anything
**Source commit in our fork**: [`8fcaf83`](https://github.com/victorkjung/transcribe-anything/commit/8fcaf83)
**File touched**: `src/transcribe_anything/insanely_fast_whisper.py`

## Title

`Mask --hf-token in OSError + stderr log to prevent secret leak`

## Body

`run_insanely_fast_whisper` calls `subprocess.list2cmdline(cmd_list)` to render the full insanely-fast-whisper command, prints it to stderr (the `Running: ...` line), and on non-zero exit raises `OSError(f"Failed to execute {cmd}")`. If `--hf-token <hf_xxxx>` is in `cmd_list`, the token leaks into:

- Process stderr (worker stdout in any serverless/Docker context)
- The `OSError` `args`, which propagates into:
  - Caller's error responses (RunPod Serverless's job status API surfaces the full traceback)
  - Any error-tracking / logging the caller has wired up

I personally hit this twice running transcribe-anything on RunPod Serverless before realizing the wrapper was the source — both times forcing a HuggingFace token rotation across multiple machines. The fix is local and tiny: regex-mask the token before any logging or raising.

## Diff

```diff
--- a/src/transcribe_anything/insanely_fast_whisper.py
+++ b/src/transcribe_anything/insanely_fast_whisper.py
@@ -8,6 +8,7 @@
 import json  # type: ignore
 import os
+import re
 import subprocess
 import sys
 import tempfile
@@ -266,7 +267,10 @@ def run_insanely_fast_whisper(
     cmd_list = [x.strip() for x in cmd_list if x.strip()]
     cmd = subprocess.list2cmdline(cmd_list)
-    sys.stderr.write(f"Running:\n  {cmd}\n")
+    # Mask --hf-token's value before any logging/error so the token doesn't
+    # leak into stdout, error responses, or downstream observability tools.
+    cmd_safe = re.sub(r"(--hf[-_]token)\s+\S+", r"\1 <REDACTED>", cmd)
+    sys.stderr.write(f"Running:\n  {cmd_safe}\n")
     proc = iso_env.open_proc(
         cmd_list,
         shell=False,
@@ -281,7 +285,7 @@ def run_insanely_fast_whisper(
             time.sleep(0.1)
             continue
         if rtn != 0:
-            msg = f"Failed to execute {cmd}\n "
+            msg = f"Failed to execute {cmd_safe}\n "
             raise OSError(msg)
         break
```

The regex matches both `--hf-token` and `--hf_token` variants.

The subprocess itself still receives the raw token via `cmd_list` (which is passed to `iso_env.open_proc` unchanged), so functionality is untouched. Only the displayed/raised string is sanitized.

## How to open the PR

```bash
git checkout -b upstream/mask-hf-token upstream/main
git cherry-pick 8fcaf83
git push origin upstream/mask-hf-token

gh pr create \
  --repo zackees/transcribe-anything \
  --head victorkjung:upstream/mask-hf-token \
  --base main \
  --title "Mask --hf-token in OSError + stderr log to prevent secret leak" \
  --body-file runpod/upstream-prs/02-mask-hf-token-in-oserror.md
```
