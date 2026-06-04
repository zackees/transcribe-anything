"""
Host-side wrapper for the WhisperX forced-alignment post-pass.

Reuses the already-built WhisperX iso-env so the insane backends don't
have to install ``whisperx`` themselves. Public entry point is
:func:`apply_forced_alignment`, called from
``insanely_fast_whisper.run_insanely_fast_whisper`` when the user passes
``--align``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from transcribe_anything.util import has_nvidia_smi
from transcribe_anything.whisperx_reqs import get_environment as get_whisperx_environment

HERE = Path(__file__).parent
RUNNER = HERE / "insane_align_runner.py"


def _align_device() -> str:
    if has_nvidia_smi():
        return "cuda"
    if sys.platform == "darwin":
        return "mps"
    return "cpu"


def apply_forced_alignment(
    json_data: dict[str, Any],
    input_wav: Path,
    language: str,
    *,
    align_model: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run wav2vec2 forced alignment on ``json_data`` and return the enriched JSON.

    The runner runs inside the WhisperX iso-env so no new dependency
    lands in the insane env. The returned dict is the same shape as
    ``json_data`` plus:

    * ``chunks[i].words = [{word, start, end, score}, ...]`` (per-word data)
    * ``chunks[i].timestamp = [first_word_start, last_word_end]`` (tightened)
    * top-level ``aligned: true`` and ``align_language: <code>`` markers.

    On any failure (unsupported language, runner crash, missing whisperx
    env), returns the input ``json_data`` unchanged after writing a
    warning to stderr. Alignment is always best-effort.
    """
    if not RUNNER.exists():
        sys.stderr.write(f"insane_align: runner script missing at {RUNNER}; skipping alignment\n")
        return json_data

    chunks = json_data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        sys.stderr.write("insane_align: input JSON has no chunks; skipping alignment\n")
        return json_data

    try:
        iso_env = get_whisperx_environment()
    except Exception as exc:  # pragma: no cover - env build failures are rare and noisy
        sys.stderr.write(f"insane_align: failed to obtain WhisperX env: {exc}; skipping alignment\n")
        return json_data

    target_device = device or _align_device()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_json = tmp_dir / "in.json"
        out_json = tmp_dir / "out.json"
        in_json.write_text(json.dumps(json_data), encoding="utf-8")

        cmd_list: list[str] = [
            str(RUNNER),
            "--input-wav",
            str(input_wav),
            "--input-json",
            str(in_json),
            "--output-json",
            str(out_json),
            "--language",
            language or "",
            "--device",
            target_device,
        ]
        if align_model:
            cmd_list.extend(["--align-model", align_model])

        cmd_str = subprocess.list2cmdline(cmd_list)
        sys.stderr.write(f"insane_align: running alignment post-pass:\n  {cmd_str}\n")

        try:
            iso_env.run(
                cmd_list,
                shell=False,
                check=True,
                universal_newlines=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(
                f"insane_align: runner failed (exit {exc.returncode}); leaving timestamps unaligned\n"
            )
            return json_data
        except Exception as exc:  # pragma: no cover - belt-and-suspenders
            sys.stderr.write(f"insane_align: unexpected runner failure: {exc}; leaving timestamps unaligned\n")
            return json_data

        if not out_json.exists():
            sys.stderr.write("insane_align: runner produced no output; leaving timestamps unaligned\n")
            return json_data
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"insane_align: runner output not valid JSON: {exc}; leaving timestamps unaligned\n")
            return json_data
