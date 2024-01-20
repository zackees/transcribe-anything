"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import shutil
import unittest
from pathlib import Path

from transcribe_anything.insanely_fast_whisper import (
    CudaInfo,
    get_cuda_info,
    has_nvidia_smi,
    run_insanely_fast_whisper,
)
from transcribe_anything.util import is_mac_arm

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TESTS_DATA_DIR = LOCALFILE_DIR / "text_video_insane" / "en"
TEST_WAV = LOCALFILE_DIR / "video.wav"

CAN_RUN_TEST = has_nvidia_smi() or is_mac_arm()


class InsanelFastWhisperTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipUnless(CAN_RUN_TEST, "No GPU detected")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        run_insanely_fast_whisper(
            input_wav=TEST_WAV,
            model="small",
            output_dir=TESTS_DATA_DIR,
            task="transcribe",
            language="en",
        )

    @unittest.skipUnless(CAN_RUN_TEST, "No GPU detected")
    def test_cuda_info(self) -> None:
        """Check that the command works on a local file."""
        cuda_info0 = get_cuda_info()
        out = cuda_info0.to_json_str()
        cuda_info1 = CudaInfo.from_json_str(out)
        print(out)
        self.assertEqual(cuda_info0, cuda_info1)


if __name__ == "__main__":
    unittest.main()
