# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501


"""
Tests transcribe_anything
"""


import os
import shutil
import unittest
from pathlib import Path

from dotenv import load_dotenv

from transcribe_anything.insanely_fast_whisper import (
    run_insanely_fast_whisper,
)
from transcribe_anything.util import has_nvidia_smi, is_mac_arm

load_dotenv()  # take environment variables from .env.

HF_TOKEN = os.getenv("HF_TOKEN")


HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TESTS_DATA_DIR = LOCALFILE_DIR / "text_video_insane" / "en"
TEST_WAV = LOCALFILE_DIR / "video.wav"
PROJECT_ROOT = HERE.parent

CAN_RUN_TEST = (has_nvidia_smi() or is_mac_arm()) and HF_TOKEN is not None


class InsanelFastWhisperDiarizationTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipUnless(CAN_RUN_TEST, "No GPU, or HF_TOKEN not set")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        run_insanely_fast_whisper(
            input_wav=TEST_WAV,
            model="small",
            output_dir=TESTS_DATA_DIR,
            task="transcribe",
            language="en",
            hugging_face_token=HF_TOKEN,
        )
        expected_file = TESTS_DATA_DIR / "speaker.json"
        self.assertTrue(expected_file.exists())


if __name__ == "__main__":
    unittest.main()
