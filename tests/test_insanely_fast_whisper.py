"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import unittest
import shutil
from pathlib import Path

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TESTS_DATA_DIR = LOCALFILE_DIR / "text_video" / "en"
TEST_WAV = LOCALFILE_DIR  / "video.wav"


class InsanelFastWhisperTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skip("DISABLED FOR NOW - WORK IN PROGRESS")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        #run_insanely_fast_whisper(
        #    input_wav=TEST_WAV,
        #    #device="cuda",
        #    #device="cpu",
        #    device="cuda:0",
        #    model="small",
        #    output_dir=TESTS_DATA_DIR,
        #    task="transcribe",
        #    language="en",
        #    other_args=None,
        #)



if __name__ == "__main__":
    unittest.main()
