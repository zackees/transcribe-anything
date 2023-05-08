"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801

import os
import unittest
import shutil

from transcribe_anything.api import transcribe

HERE = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA_DIR = os.path.join(HERE, "test_data", "en")
LOCALFILE_DIR = os.path.join(HERE, "localfile")


class TranscribeAnythingApiTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        expected_base_dir = os.path.join(LOCALFILE_DIR, "text_video", "en")
        shutil.rmtree(expected_base_dir, ignore_errors=True)
        vidfile = os.path.join(LOCALFILE_DIR, "video.mp4")
        prev_dir = os.getcwd()
        os.chdir(LOCALFILE_DIR)
        transcribe(url_or_file=vidfile, language="en", model="tiny")
        os.chdir(prev_dir)
        expected_paths = [
            expected_base_dir,
            # os.path.join(TESTS_DATA_DIR, "out.mp3"),
            os.path.join(expected_base_dir, "out.txt"),
            os.path.join(expected_base_dir, "out.srt"),
            os.path.join(expected_base_dir, "out.vtt"),
        ]
        for expected_path in expected_paths:
            self.assertTrue(
                os.path.exists(expected_path),
                f"expected path {expected_path} not found",
            )


if __name__ == "__main__":
    unittest.main()
