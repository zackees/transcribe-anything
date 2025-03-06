"""
Tests whisper ai cmd
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transcribe_anything import transcribe_anything

HERE = os.path.abspath(os.path.dirname(__file__))
LOCALFILE_DIR = os.path.join(HERE, "localfile")
VIDEO_MP4 = os.path.join(LOCALFILE_DIR, "video.mp4")


class ApiTester(unittest.TestCase):
    """Tester for whisper ai."""

    def test_api(self) -> None:
        """Check that the command is installed by the setup process."""
        with TemporaryDirectory() as tempdir:
            print(f"Running in {tempdir}")
            transcribe_anything(url_or_file=VIDEO_MP4, output_dir=tempdir, task="transcribe", model="tiny", device="cpu")
            output_has_files = len(list(Path(tempdir).iterdir())) > 0
            self.assertTrue(output_has_files)


if __name__ == "__main__":
    unittest.main()
