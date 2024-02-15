"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import json
import os
import sys
import tempfile
import unittest

IS_MACOS = sys.platform == "darwin"

HERE = os.path.abspath(os.path.dirname(__file__))


class BadVideoTitleTester(unittest.TestCase):
    """Tester for transcribe anything."""

    @unittest.skipIf(IS_MACOS, "Skipping test on MacOS")
    def test_local_file(self) -> None:
        """Check that the command works on a local file."""
        prevdir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                print(f"Using temporary directory {tmpdir}")
                os.chdir(tmpdir)
                os.system("transcribe_anything --query-gpu-json-path info.json")
                self.assertTrue(os.path.exists("info.json"))
                # parsing test
                with open("info.json", encoding="utf-8", mode="rt") as fd:
                    info = json.load(fd)
                self.assertTrue(info is not None)
            finally:
                os.chdir(prevdir)


if __name__ == "__main__":
    unittest.main()
