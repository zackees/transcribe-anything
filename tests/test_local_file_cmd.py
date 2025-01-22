"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access

import os
import shutil
import subprocess
import unittest

HERE = os.path.abspath(os.path.dirname(__file__))
LOCALFILE_DIR = os.path.join(HERE, "localfile")
TESTS_DATA_DIR = os.path.join(LOCALFILE_DIR, "text_video", "en")


class TranscribeAnythingTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        cmd_list: list[str] = [
            "transcribe_anything",
            "video.mp4",
            "--language",
            "en",
            "--model",
            "tiny",
        ]
        try:
            cmd_str = subprocess.list2cmdline(cmd_list)
            print(f"Running in {LOCALFILE_DIR}: {cmd_str}")
            subprocess.run(cmd_list, cwd=LOCALFILE_DIR, check=True)
        except subprocess.CalledProcessError as e:  # pylint: disable=R0801
            print(e.output)
            raise e


if __name__ == "__main__":
    unittest.main()
