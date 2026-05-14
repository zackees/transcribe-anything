# Upstream PR drafts

Three issues hit while deploying transcribe-anything on RunPod Serverless, each worth contributing back upstream. The fixes are already in our `runpod-integration` branch; these markdown files are PR descriptions ready to paste into GitHub.

| # | Issue | Upstream repo | Fork commit |
|---|---|---|---|
| 1 | `setuptools<82` constraint missing from insane env's pyproject | `zackees/transcribe-anything` | [`d78814f`](https://github.com/victorkjung/transcribe-anything/commit/d78814f) |
| 2 | `--hf-token` leaks via `OSError` cmdline | `zackees/transcribe-anything` | [`8fcaf83`](https://github.com/victorkjung/transcribe-anything/commit/8fcaf83) |
| 3 | `diarize_audio` crashes on empty pyannote segments | `Vaibhavs10/insanely-fast-whisper` | not yet — see #3 doc |

## Recommended workflow

Don't open one giant PR from our `runpod-integration` branch — it has many unrelated commits. Cherry-pick each fix onto a clean branch:

```bash
# In your fork clone
for n in 01 02; do
  # Use the commit refs from each PR doc, then push, then create PR
done
```

Each PR doc has the exact `gh pr create` command ready to paste.

## Note: these are drafts, not submissions

I (the human) need to review and click submit on each PR. The diffs and commit refs are real; the body text is ready.
