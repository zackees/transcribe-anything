"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access
# flake8: noqa E501

import os
import shutil
import subprocess
import unittest

HERE = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA_DIR = os.path.join(HERE, "test_data")
LOCALFILE_DIR = os.path.join(HERE, "localfile")
_IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TranscribeAnythingTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_fetch_command_help(self) -> None:
        """Check that the command is installed by the setup process."""
        subprocess.check_output(["transcribe_anything", "-h"])

    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        subprocess.check_output(
            ["transcribe_anything", "video.mp4", "--language", "en", "--model", "tiny"],
            cwd=LOCALFILE_DIR,
        )

    def test_initial_prompt(self) -> None:
        """Check that the command works on a local file."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        subprocess.check_output(
            [
                "transcribe_anything",
                "video.mp4",
                "--language",
                "en",
                "--model",
                "tiny",
                "--initial_prompt",
                '"nervous"',
            ],
            cwd=LOCALFILE_DIR,
        )
        # Expect that when the language is explicitly set that the output file ends up in
        # that language folder.
        expected_output = os.path.join(LOCALFILE_DIR, "text_video", "en", "out.txt")
        self.assertTrue(
            os.path.exists(expected_output),
            f"Path {expected_output} doesn't exist, instead it was {os.listdir(LOCALFILE_DIR)}",  # pylint: disable=line-too-long
        )

    @unittest.skipIf(_IS_GITHUB_ACTIONS, "Skipping test on GitHub Actions")
    def test_fetch_command_installed(self) -> None:
        """Check that the command works on a live short video."""
        shutil.rmtree(TESTS_DATA_DIR, ignore_errors=True)
        cmd = "transcribe_anything https://www.youtube.com/watch?v=DWtpNPZ4tb4" f" --model tiny --output_dir {TESTS_DATA_DIR}"
        rtn_val = subprocess.call(cmd, shell=True)
        self.assertEqual(rtn_val, 0, "command failed")
        expected_paths = [
            TESTS_DATA_DIR,
            # os.path.join(TESTS_DATA_DIR, "out.mp3"),
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
