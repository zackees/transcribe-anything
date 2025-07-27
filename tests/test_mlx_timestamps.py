"""Tests for MLX timestamp conversion logic.

This suite contains:
- Unit tests for the internal _json_to_srt() conversion using synthetic MLX-style
  segment data (list format: [start_seek, end_seek, text]).
- A dictionary format regression test (old format path).
- An optional integration test (Mac ARM only) that runs the MLX backend on the
  existing test audio and compares the last subtitle end time to the true WAV
  duration.

The fix for Issue #1 has been applied - the correct 0.01 seconds per frame
conversion factor is now used for MLX list-format segments.

Issue Reference: #1 (MLX Mode Timestamp Accuracy)
"""

from __future__ import annotations

import re
import shutil
import sys
import wave
from pathlib import Path

import pytest

from transcribe_anything.util import is_mac_arm
from transcribe_anything.whisper_mac import _json_to_srt, run_whisper_mac_mlx

HERE = Path(__file__).parent
LOCALFILE_DIR = HERE / "localfile"
TEST_WAV = LOCALFILE_DIR / "video.wav"
REFERENCE_SRT = LOCALFILE_DIR / "video.srt"
INTEGRATION_OUT = LOCALFILE_DIR / "mlx_timestamp_out"


def test_mlx_list_segment_frame_scaling() -> None:
    """Unit test for list-format MLX segments ensuring correct frame->sec scaling.

    Synthetic MLX JSON (frames):
      Frame indices: 0-50 (0.5s), 50-150 (1.0s), 150-200 (0.5s)
    Expected times with correct 0.01 factor:
      00:00:00,000 --> 00:00:00,500
      00:00:00,500 --> 00:00:01,500
      00:00:01,500 --> 00:00:02,000
    """
    json_data = {
        "segments": [
            [0, 50, "First"],
            [50, 150, "Second"],
            [150, 200, "Third"],
        ],
        "text": "First Second Third",
    }
    srt = _json_to_srt(json_data)  # type: ignore[arg-type]

    # Validate expected timestamps (these will not appear until the factor is fixed)
    assert "00:00:00,000 --> 00:00:00,500" in srt
    assert "00:00:00,500 --> 00:00:01,500" in srt
    assert "00:00:01,500 --> 00:00:02,000" in srt


def test_old_dict_segment_format_pass_through() -> None:
    """Regression test: dict segment format should NOT apply frame scaling.

    Ensures backward-compatible path remains correct.
    """
    json_data = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
        ],
        "text": "Hello world",
    }
    srt = _json_to_srt(json_data)  # type: ignore[arg-type]
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "Hello world" in srt


def test_bug_demonstration_old_vs_new_factor() -> None:
    """Demonstrate the difference between old (0.02) and new (0.01) conversion factors.

    This test shows what timestamps would look like with the old bug vs the fix.
    """
    # Simulate a 10-second audio file with segments at frame indices
    # that should span the full duration when converted correctly
    json_data = {
        "segments": [
            [0, 500, "First half"],  # 0-5 seconds with 0.01 factor
            [500, 1000, "Second half"],  # 5-10 seconds with 0.01 factor
        ],
        "text": "First half Second half",
    }

    srt = _json_to_srt(json_data)  # type: ignore[arg-type]

    # With correct 0.01 factor, should get:
    assert "00:00:00,000 --> 00:00:05,000" in srt  # 0*0.01 to 500*0.01
    assert "00:00:05,000 --> 00:00:10,000" in srt  # 500*0.01 to 1000*0.01

    # With old 0.02 factor, would have gotten (demonstrating the bug):
    # 00:00:00,000 --> 00:00:10,000 (0*0.02 to 500*0.02)
    # 00:00:10,000 --> 00:00:20,000 (500*0.02 to 1000*0.02)
    # This would make a 10-second audio appear to have 20-second subtitles!


@pytest.mark.slow
@pytest.mark.skipif(not is_mac_arm(), reason="MLX backend only supported / meaningful on Mac ARM")
def test_integration_mlx_end_time_close_to_wav_duration() -> None:
    """Integration test (Mac ARM only) comparing SRT end time with WAV duration.

    Uses existing test audio (video.wav). This test downloads a model ("small")
    so it is marked as slow. A tolerance band is used because Whisper segment end
    may undershoot or overshoot slightly. Bug causes approx 2x stretch → ratio
    outside band.
    """
    out_dir = INTEGRATION_OUT
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run transcription (will populate out.srt)
    run_whisper_mac_mlx(
        input_wav=TEST_WAV,
        model="small",
        output_dir=out_dir,
        language="en",
        task="transcribe",
    )

    # Compute actual WAV duration
    with wave.open(str(TEST_WAV), "rb") as w:
        wav_duration = w.getnframes() / w.getframerate()

    srt_path = out_dir / "out.srt"
    assert srt_path.exists(), "SRT file not generated"
    srt_content = srt_path.read_text(encoding="utf-8")

    # Find last subtitle timing line
    matches = re.findall(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})", srt_content)
    assert matches, "No timestamp lines found in SRT"
    last = matches[-1]
    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, last)
    end_secs = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0

    ratio = end_secs / wav_duration if wav_duration > 0 else 0.0
    # With correct scaling expect ratio roughly near 1 (allow generous band).
    # Current bug ~2.0 → outside 1.3 upper bound.
    assert 0.7 <= ratio <= 1.3, f"End time ratio outside tolerance: {ratio:.2f} (end={end_secs:.2f}, dur={wav_duration:.2f})"


@pytest.mark.slow
@pytest.mark.skipif(not is_mac_arm(), reason="MLX backend only supported / meaningful on Mac ARM")
def test_integration_mlx_vs_reference_srt() -> None:
    """Integration test comparing MLX output against reference SRT file.

    This test runs MLX transcription and compares the timing accuracy against
    a known-good reference SRT file (video.srt) for the same audio.
    """
    # Load reference SRT for comparison
    assert REFERENCE_SRT.exists(), f"Reference SRT not found: {REFERENCE_SRT}"
    reference_content = REFERENCE_SRT.read_text(encoding="utf-8")

    # Parse reference timestamps
    ref_matches = re.findall(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})", reference_content)
    assert ref_matches, "No timestamps found in reference SRT"

    # Get reference end time
    ref_last = ref_matches[-1]
    ref_h1, ref_m1, ref_s1, ref_ms1, ref_h2, ref_m2, ref_s2, ref_ms2 = map(int, ref_last)
    ref_end_secs = ref_h2 * 3600 + ref_m2 * 60 + ref_s2 + ref_ms2 / 1000.0

    # Run MLX transcription
    out_dir = INTEGRATION_OUT
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_whisper_mac_mlx(
        input_wav=TEST_WAV,
        model="small",
        output_dir=out_dir,
        language="en",
        task="transcribe",
    )

    # Parse MLX output
    srt_path = out_dir / "out.srt"
    assert srt_path.exists(), "MLX SRT file not generated"
    mlx_content = srt_path.read_text(encoding="utf-8")

    mlx_matches = re.findall(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})", mlx_content)
    assert mlx_matches, "No timestamps found in MLX SRT"

    # Get MLX end time
    mlx_last = mlx_matches[-1]
    mlx_h1, mlx_m1, mlx_s1, mlx_ms1, mlx_h2, mlx_m2, mlx_s2, mlx_ms2 = map(int, mlx_last)
    mlx_end_secs = mlx_h2 * 3600 + mlx_m2 * 60 + mlx_s2 + mlx_ms2 / 1000.0

    # Compare end times - should be reasonably close
    # Reference ends at 9.460s, so MLX should be in similar range
    time_diff = abs(mlx_end_secs - ref_end_secs)
    tolerance = 2.0  # Allow 2 second difference (different models may segment differently)

    assert time_diff <= tolerance, f"MLX end time differs too much from reference: " f"MLX={mlx_end_secs:.3f}s, Reference={ref_end_secs:.3f}s, " f"Diff={time_diff:.3f}s (tolerance={tolerance}s)"

    # Additional check: MLX end time should be reasonable relative to audio duration
    with wave.open(str(TEST_WAV), "rb") as w:
        wav_duration = w.getnframes() / w.getframerate()

    mlx_ratio = mlx_end_secs / wav_duration
    assert 0.7 <= mlx_ratio <= 1.1, f"MLX end time ratio unreasonable: {mlx_ratio:.3f} " f"(MLX={mlx_end_secs:.3f}s, WAV={wav_duration:.3f}s)"


def test_reference_srt_structure_validation() -> None:
    """Validate that our reference SRT has the expected structure and timing."""
    assert REFERENCE_SRT.exists(), f"Reference SRT not found: {REFERENCE_SRT}"
    content = REFERENCE_SRT.read_text(encoding="utf-8")

    # Parse timestamps
    matches = re.findall(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})", content)
    assert len(matches) >= 1, "Reference SRT should have at least one timestamp"

    # Check that timestamps are reasonable for our 10-second audio
    last = matches[-1]
    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, last)
    end_secs = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0

    # Reference should end before the audio duration (9.460s for 10.027s audio)
    assert 8.0 <= end_secs <= 11.0, f"Reference end time seems unreasonable: {end_secs:.3f}s"

    # Check that timestamps are in ascending order
    prev_end = 0.0
    for match in matches:
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match)
        start_secs = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end_secs = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0

        assert start_secs >= prev_end, f"Timestamps not in order: {start_secs} should be >= {prev_end}"
        assert end_secs > start_secs, f"End time should be after start time: {end_secs} > {start_secs}"
        prev_end = end_secs


if __name__ == "__main__":  # pragma: no cover
    # Allow running this file directly for debugging.
    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-vv"]))
