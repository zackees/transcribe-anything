"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import os
import unittest

from transcribe_anything.api import get_video_name_from_url

HERE = os.path.abspath(os.path.dirname(__file__))

_IS_GITHUB = os.environ.get("GITHUB_ACTIONS") == "true"


class BadVideoTitleTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipIf(_IS_GITHUB, "Skipping test on GitHub Actions")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        name = get_video_name_from_url("https://www.youtube.com/watch?v=3JZ_D3ELwOQ")
        self.assertEqual(name, "Flexin' On Ya (2014)")


if __name__ == "__main__":
    unittest.main()
