"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import unittest
import shutil
from pathlib import Path
import tempfile
from transcribe_anything.insanely_fast_whisper import (
    srt_wrap_to_string,
    has_nvidia_smi,
)


HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TEST_SRT = LOCALFILE_DIR / "long.srt"


class InsanelFastWhisperTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipUnless(has_nvidia_smi(), "No GPU detected")
    def test_srt_wrap(self) -> None:
        """Check that the command works on a local file."""
        with tempfile.TemporaryDirectory() as tempdir:
            td = Path(tempdir)
            target = td / "long.srt"
            shutil.copy(TEST_SRT, target)
            wrapped_srt = srt_wrap_to_string(target)
            print(wrapped_srt)
            print()


if __name__ == "__main__":
    unittest.main()
