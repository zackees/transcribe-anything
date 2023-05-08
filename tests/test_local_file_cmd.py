"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access

import os
import subprocess
import unittest
import shutil

HERE = os.path.abspath(os.path.dirname(__file__))
LOCALFILE_DIR = os.path.join(HERE, "localfile")
TESTS_DATA_DIR = os.path.join(LOCALFILE_DIR, "text_video", "en")


class TranscribeAnythingTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        subprocess.check_output(
            ["transcribe_anything", "video.mp4", "--language", "en", "--model", "tiny"],
            cwd=LOCALFILE_DIR,
        )


if __name__ == "__main__":
    unittest.main()
