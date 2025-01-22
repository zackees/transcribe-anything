"""
Tests transcribe_anything
"""

import os
import platform
import unittest

from transcribe_anything.api import fix_subtitles_path

HERE = os.path.abspath(os.path.dirname(__file__))

_IS_WINDOWS = platform.system() == "Windows"


class FixSubtitlesTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipIf(not _IS_WINDOWS, "Test only works on Windows")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        path = r"C:\Users\Toshiba\Pictures\test vcp\shopi.mp4"
        fixed_path = fix_subtitles_path(path)
        expected_path = r"C\\:/\Users/\Toshiba/\Pictures/\test vcp/\shopi.mp4"
        self.assertEqual(fixed_path, expected_path)


if __name__ == "__main__":
    unittest.main()
