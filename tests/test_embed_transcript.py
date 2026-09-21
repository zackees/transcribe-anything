"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import shutil
import unittest
from unittest.mock import Mock, patch

from transcribe_anything.api import _embed_subtitles, fix_subtitles_path, transcribe

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

    @patch("transcribe_anything.api.subprocess.run")
    def test_embed_does_not_invoke_a_shell(self, run_mock: Mock) -> None:
        """Treat shell metacharacters in an input filename as literal text."""
        input_path = "video&whoami&.mp4"
        srt_path = "subtitle.srt"
        output_path = "out.mp4"

        _embed_subtitles(input_path, srt_path, output_path)

        run_mock.assert_called_once_with(
            [
                "static_ffmpeg",
                "-y",
                "-i",
                input_path,
                "-i",
                srt_path,
                "-vf",
                f"subtitles={fix_subtitles_path(srt_path)}",
                output_path,
            ],
            universal_newlines=True,
            check=True,
            capture_output=True,
            shell=False,
        )


if __name__ == "__main__":
    unittest.main()
