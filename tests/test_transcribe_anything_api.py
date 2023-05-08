"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access

import os
import unittest
import shutil

from transcribe_anything.api import transcribe

HERE = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA_DIR = os.path.join(HERE, "test_data")
LOCALFILE_DIR = os.path.join(HERE, "localfile")


class TranscribeAnythingApiTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        vidfile = os.path.join(LOCALFILE_DIR, "video.mp4")
        transcribe(url_or_file=vidfile, language="en", model="tiny")


if __name__ == "__main__":
    unittest.main()
