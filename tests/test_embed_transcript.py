"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801

import os
import unittest
import shutil

from transcribe_anything.api import transcribe

HERE = os.path.abspath(os.path.dirname(__file__))
LOCALFILE_DIR = os.path.join(HERE, "localfile")
TESTS_DATA_DIR = os.path.join(LOCALFILE_DIR, "text_video", "en")


class TranscribeAnythingApiEmbedTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        vidfile = os.path.join(LOCALFILE_DIR, "video.mp4")
        prev_dir = os.getcwd()
        os.chdir(LOCALFILE_DIR)
        transcribe(url_or_file=vidfile, language="en", model="tiny", embed=True)
        os.chdir(prev_dir)
        expected_paths = [
            TESTS_DATA_DIR,
            os.path.join(TESTS_DATA_DIR, "out.txt"),
            os.path.join(TESTS_DATA_DIR, "out.srt"),
            os.path.join(TESTS_DATA_DIR, "out.vtt"),
        ]
        for expected_path in expected_paths:
            self.assertTrue(
                os.path.exists(expected_path),
                f"expected path {expected_path} not found",
            )


if __name__ == "__main__":
    unittest.main()
