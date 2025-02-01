"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import shutil
import unittest
from pathlib import Path

from transcribe_anything.util import is_mac_arm
from transcribe_anything.whisper_mac import run_whisper_mac_english

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TESTS_DATA_DIR = LOCALFILE_DIR / "text_video_insane" / "en"
TEST_WAV = LOCALFILE_DIR / "video.wav"

CAN_RUN_TEST = is_mac_arm()


class MacOsWhisperMpsTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipUnless(CAN_RUN_TEST, "Not mac")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        run_whisper_mac_english(
            input_wav=TEST_WAV,
            model="small",
            output_dir=TESTS_DATA_DIR,
        )


if __name__ == "__main__":
    unittest.main()
