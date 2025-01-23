"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import shutil
import unittest

from transcribe_anything.api import transcribe

HERE = os.path.abspath(os.path.dirname(__file__))
LOCALFILE_DIR = os.path.join(HERE, "localfile")
TESTS_DATA_DIR = os.path.join(LOCALFILE_DIR, "text_video", "en")

_IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


class TranscribeAnythingApiTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipIf(_IS_GITHUB_ACTIONS, "Skipping test on GitHub Actions")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        vidfile = os.path.join(LOCALFILE_DIR, "video.mp4")
        prev_dir = os.getcwd()
        os.chdir(LOCALFILE_DIR)
        transcribe(url_or_file=vidfile, language="en", model="tiny")
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

    @unittest.skipIf(_IS_GITHUB_ACTIONS, "Skipping test on GitHub Actions")
    def test_fetch_command_installed(self) -> None:
        """Check that the command works on a live short video."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        transcribe(
            url_or_file="https://www.youtube.com/watch?v=DWtpNPZ4tb4",
            language="en",
            model="tiny",
            output_dir=TESTS_DATA_DIR,
        )
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
