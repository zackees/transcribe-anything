"""End-to-end check: --device cuda and --device insane agree on long-form timestamps.

Background
----------

The HF transformers Whisper pipeline (used by ``--device insane``) had a
chunked-decoding timestamp-offset bug that made the last segment of any
audio longer than 30 s under-report its end time by a multiple of 30 s
(transformers issues #34210 / #31942 / #34472, fixed in PRs #34537 and
#35750, landing fully in transformers v4.53.0).

This test bumps the bundled 10-s ``sample.mp3`` up to ~40 s by looping
it four times, runs both backends with ``--model tiny`` (small enough
that the test is hermetic on first run), and asserts that the last
segment's end timestamp from each backend agrees to within 1.0 s.

Before the pin bump (``transformers==4.46.3``), the insane backend
would report a last timestamp roughly 30 s short — well outside the
tolerance — and the test would fail loudly. After the bump
(``transformers==4.55.4``), both backends should land near the wav's
true duration (~40 s).

Gated on the presence of NVIDIA: the insane backend is GPU-only on
Linux/Windows, and there is no Mac path here. Skipped on hosts without
``nvidia-smi``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from transcribe_anything.api import transcribe
from transcribe_anything.util import has_nvidia_smi, is_mac

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
ASSETS = HERE.parent / "src" / "transcribe_anything" / "assets"
SAMPLE_MP3 = ASSETS / "sample.mp3"

CAN_RUN_TEST = has_nvidia_smi() and not is_mac() and SAMPLE_MP3.is_file()
RUN_INSANE_FLASH_TEST = os.environ.get("TRANSCRIBE_ANYTHING_TEST_INSANE_FLASH", "").strip().lower() in {"1", "true", "yes", "on"}

# Number of loops of the ~10 s sample.mp3 to concatenate. Four copies
# put us well past the 30-s chunk boundary that triggers the upstream bug.
LOOP_COUNT = 4

# Tolerance between the two backends' final-segment end timestamps.
# The legitimate per-segment-token vs per-chunk-token granularity
# difference is at most a few hundred ms; anything larger than 1 s
# means we're hitting the chunked-decoding rollover bug again.
LAST_TIMESTAMP_TOLERANCE_S = 1.0

# How far below / above the true audio duration the last timestamp may
# legitimately fall. The model can stop a beat early (no speech at the
# very end) or overshoot trailing silence by a small amount.
WAV_DURATION_LOWER_SLACK_S = 4.0
WAV_DURATION_UPPER_SLACK_S = 1.0


def _make_long_wav(workdir: Path) -> Path:
    """Use static_ffmpeg to loop sample.mp3 and render a 16 kHz mono WAV.

    Returns the path to the new wav. The wav is dropped in ``workdir``.
    """
    import static_ffmpeg  # type: ignore[import-untyped]

    static_ffmpeg.add_paths()
    out_wav = workdir / "long_sample.wav"
    # -stream_loop N replays the input (N+1) total times. LOOP_COUNT-1
    # for the desired total.
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        str(LOOP_COUNT - 1),
        "-i",
        str(SAMPLE_MP3),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out_wav.is_file(), f"ffmpeg returned 0 but no output wav: {result.stderr}"
    return out_wav


def _wav_duration_seconds(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _last_end_timestamp_from_whisper_json(path: Path) -> float:
    """openai-whisper writes ``out.json`` with a ``segments`` list of
    ``{start, end, text}`` dicts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data["segments"]
    assert segments, f"openai-whisper out.json had no segments: {data}"
    return float(segments[-1]["end"])


def _last_end_timestamp_from_insane_json(path: Path) -> float:
    """The insane backend's ``out.json`` is the HF pipeline format with a
    ``chunks`` list of ``{text, timestamp: [start, end]}`` dicts. The last
    chunk's end timestamp may be ``None`` if the model never emitted a
    closing timestamp token; in that case we fall back to the previous
    chunk's end."""
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    assert chunks, f"insane out.json had no chunks: {data}"
    for chunk in reversed(chunks):
        end = chunk["timestamp"][1]
        if end is not None:
            return float(end)
    raise AssertionError(f"no closed-end chunk in insane out.json: {chunks}")


def _last_end_timestamp_from_srt(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})", text)
    if not matches:
        raise AssertionError(f"No SRT timestamp end found in {path}:\n{text}")
    hours, minutes, seconds, millis = matches[-1]
    return (float(hours) * 3600.0) + (float(minutes) * 60.0) + float(seconds) + (float(millis) / 1000.0)


@unittest.skipUnless(CAN_RUN_TEST, "Requires NVIDIA (not on Mac) and bundled sample.mp3")
class LongformTimestampAgreementTester(unittest.TestCase):
    """Pin down that the two backends agree on the end of long audio."""

    def test_last_segment_timestamps_agree_within_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            long_wav = _make_long_wav(workdir)
            wav_duration = _wav_duration_seconds(long_wav)
            # Sanity: ensure we actually pushed past the 30-s chunk boundary.
            self.assertGreater(
                wav_duration,
                30.0,
                msg=f"Long wav is only {wav_duration:.2f}s — does not exercise multi-chunk path",
            )

            cuda_out = workdir / "out_cuda"
            insane_out = workdir / "out_insane"
            insane_flash_out = workdir / "out_insane_flash"
            runtimes: dict[str, float] = {}

            # Clean before each backend run in case a previous attempt
            # left stale artifacts.
            shutil.rmtree(cuda_out, ignore_errors=True)
            shutil.rmtree(insane_out, ignore_errors=True)
            shutil.rmtree(insane_flash_out, ignore_errors=True)

            # Use the public transcribe() API rather than the low-level
            # run_whisper / run_insanely_fast_whisper helpers, because
            # transcribe() handles the per-backend output-file renaming
            # (openai-whisper writes <input_stem>.json; transcribe()
            # renames everything to out.json/out.srt/etc.).
            start = time.perf_counter()
            transcribe(
                url_or_file=str(long_wav),
                output_dir=str(cuda_out),
                model="tiny",
                task="transcribe",
                language="en",
                device="cuda",
            )
            runtimes["cuda"] = time.perf_counter() - start

            start = time.perf_counter()
            transcribe(
                url_or_file=str(long_wav),
                output_dir=str(insane_out),
                model="tiny",
                task="transcribe",
                language="en",
                device="insane",
            )
            runtimes["insane"] = time.perf_counter() - start

            cuda_last = _last_end_timestamp_from_whisper_json(cuda_out / "out.json")
            insane_last = _last_end_timestamp_from_insane_json(insane_out / "out.json")
            cuda_srt_last = _last_end_timestamp_from_srt(cuda_out / "out.srt")
            insane_srt_last = _last_end_timestamp_from_srt(insane_out / "out.srt")

            # Each backend's last timestamp should bracket the true duration.
            self.assertGreaterEqual(
                cuda_last,
                wav_duration - WAV_DURATION_LOWER_SLACK_S,
                msg=f"cuda last={cuda_last:.2f}s far below wav_duration={wav_duration:.2f}s",
            )
            self.assertLessEqual(
                cuda_last,
                wav_duration + WAV_DURATION_UPPER_SLACK_S,
                msg=f"cuda last={cuda_last:.2f}s overshot wav_duration={wav_duration:.2f}s",
            )
            self.assertGreaterEqual(
                insane_last,
                wav_duration - WAV_DURATION_LOWER_SLACK_S,
                msg=(
                    f"insane last={insane_last:.2f}s far below wav_duration={wav_duration:.2f}s "
                    "— probable upstream chunked-decoding timestamp-rollover bug "
                    "(transformers PR #35750). Check the transformers pin."
                ),
            )
            self.assertLessEqual(
                insane_last,
                wav_duration + WAV_DURATION_UPPER_SLACK_S,
                msg=f"insane last={insane_last:.2f}s overshot wav_duration={wav_duration:.2f}s",
            )

            # And the two backends should agree on the end-of-audio.
            self.assertAlmostEqual(
                cuda_last,
                insane_last,
                delta=LAST_TIMESTAMP_TOLERANCE_S,
                msg=(
                    f"cuda last={cuda_last:.2f}s vs insane last={insane_last:.2f}s differ "
                    f"by more than {LAST_TIMESTAMP_TOLERANCE_S}s. Indicates the chunked "
                    "decoding offset bug is back; verify transformers>=4.53.0 is actually installed."
                ),
            )
            self.assertAlmostEqual(cuda_last, cuda_srt_last, delta=LAST_TIMESTAMP_TOLERANCE_S)
            self.assertAlmostEqual(insane_last, insane_srt_last, delta=LAST_TIMESTAMP_TOLERANCE_S)

            if RUN_INSANE_FLASH_TEST:
                start = time.perf_counter()
                transcribe(
                    url_or_file=str(long_wav),
                    output_dir=str(insane_flash_out),
                    model="tiny",
                    task="transcribe",
                    language="en",
                    device="insane-flash",
                )
                runtimes["insane-flash"] = time.perf_counter() - start

                insane_flash_last = _last_end_timestamp_from_insane_json(insane_flash_out / "out.json")
                insane_flash_srt_last = _last_end_timestamp_from_srt(insane_flash_out / "out.srt")

                self.assertGreaterEqual(
                    insane_flash_last,
                    wav_duration - WAV_DURATION_LOWER_SLACK_S,
                    msg=f"insane-flash last={insane_flash_last:.2f}s far below wav_duration={wav_duration:.2f}s",
                )
                self.assertLessEqual(
                    insane_flash_last,
                    wav_duration + WAV_DURATION_UPPER_SLACK_S,
                    msg=f"insane-flash last={insane_flash_last:.2f}s overshot wav_duration={wav_duration:.2f}s",
                )
                self.assertAlmostEqual(
                    cuda_last,
                    insane_flash_last,
                    delta=LAST_TIMESTAMP_TOLERANCE_S,
                    msg=(f"cuda last={cuda_last:.2f}s vs insane-flash last={insane_flash_last:.2f}s " f"differ by more than {LAST_TIMESTAMP_TOLERANCE_S}s"),
                )
                self.assertAlmostEqual(
                    insane_last,
                    insane_flash_last,
                    delta=LAST_TIMESTAMP_TOLERANCE_S,
                    msg=(f"insane last={insane_last:.2f}s vs insane-flash last={insane_flash_last:.2f}s " f"differ by more than {LAST_TIMESTAMP_TOLERANCE_S}s"),
                )
                self.assertAlmostEqual(insane_flash_last, insane_flash_srt_last, delta=LAST_TIMESTAMP_TOLERANCE_S)

            print("Long-form backend runtimes:", {name: round(value, 2) for name, value in runtimes.items()})


if __name__ == "__main__":
    unittest.main()
