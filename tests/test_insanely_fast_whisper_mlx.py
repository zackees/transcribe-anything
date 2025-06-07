"""
Tests transcribe_anything with lightning-whisper-mlx (MLX backend)
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import shutil
import unittest
from pathlib import Path

from transcribe_anything.util import is_mac_arm
from transcribe_anything.whisper_mac import run_whisper_mac_mlx, run_whisper_mac_english

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TESTS_DATA_DIR = LOCALFILE_DIR / "text_video_mlx" / "en"
TEST_WAV = LOCALFILE_DIR / "video.wav"

CAN_RUN_TEST = is_mac_arm()


class MacOsWhisperMLXTester(unittest.TestCase):
    """Tester for transcribe anything with lightning-whisper-mlx (MLX backend)."""

    @unittest.skipUnless(CAN_RUN_TEST, "Not mac")
    def test_local_file_english(self) -> None:
        """Check that the command works on a local file with English."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        run_whisper_mac_mlx(
            input_wav=TEST_WAV,
            model="small",
            output_dir=TESTS_DATA_DIR,
            language="en",
            task="transcribe"
        )

        # Verify output files were created
        self.assertTrue((TESTS_DATA_DIR / "out.txt").exists())
        self.assertTrue((TESTS_DATA_DIR / "out.srt").exists())
        self.assertTrue((TESTS_DATA_DIR / "out.json").exists())
        self.assertTrue((TESTS_DATA_DIR / "out.vtt").exists())

    @unittest.skipUnless(CAN_RUN_TEST, "Not mac")
    def test_local_file_with_initial_prompt(self) -> None:
        """Check that the command works with initial_prompt (now supported)."""
        test_dir = LOCALFILE_DIR / "text_video_mlx_prompt"
        shutil.rmtree(test_dir, ignore_errors=True)

        # This should work with initial_prompt support
        run_whisper_mac_mlx(
            input_wav=TEST_WAV,
            model="small",
            output_dir=test_dir,
            language="en",
            task="transcribe",
            other_args=["--initial_prompt", "test vocabulary terms"]
        )

        # Verify output files were created
        self.assertTrue((test_dir / "out.txt").exists())
        self.assertTrue((test_dir / "out.srt").exists())
        self.assertTrue((test_dir / "out.json").exists())

    @unittest.skipUnless(CAN_RUN_TEST, "Not mac")
    def test_backward_compatibility(self) -> None:
        """Check that the old function still works for backward compatibility."""
        test_dir = LOCALFILE_DIR / "text_video_compat"
        shutil.rmtree(test_dir, ignore_errors=True)

        # Test the old function name
        run_whisper_mac_english(
            input_wav=TEST_WAV,
            model="small",
            output_dir=test_dir,
        )

        # Verify output files were created
        self.assertTrue((test_dir / "out.txt").exists())
        self.assertTrue((test_dir / "out.srt").exists())
        self.assertTrue((test_dir / "out.json").exists())

    @unittest.skipUnless(CAN_RUN_TEST, "Not mac")
    def test_multilingual_support(self) -> None:
        """Check that multilingual support works (auto-detect)."""
        test_dir = LOCALFILE_DIR / "text_video_multilingual"
        shutil.rmtree(test_dir, ignore_errors=True)

        # Test with auto-detection (no language specified)
        run_whisper_mac_mlx(
            input_wav=TEST_WAV,
            model="small",
            output_dir=test_dir,
            language=None,  # Auto-detect
            task="transcribe"
        )

        # Verify output files were created
        self.assertTrue((test_dir / "out.txt").exists())
        self.assertTrue((test_dir / "out.srt").exists())
        self.assertTrue((test_dir / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
