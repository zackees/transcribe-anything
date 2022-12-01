"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access

import os
import subprocess
import unittest

HERE = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA_DIR = os.path.join(HERE, "test_data")


class TranscribeAnythingTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_fetch_command_help(self) -> None:
        """Check that the command is installed by the setup process."""
        subprocess.check_output(['transcribe_anything', '-h'])

    def test_fetch_command_installed(self) -> None:
        """Check that the command works on a live short video."""
        cmd_list = [
            'transcribe_anything',
            'https://www.youtube.com/watch?v=8Wg8f2g_GQY',
            '--model', 'small',
            '--output_dirname', TESTS_DATA_DIR,
        ]
        rtn_val = subprocess.call(cmd_list, shell=True)
        self.assertEqual(rtn_val, 0, "command failed")


if __name__ == '__main__':
    unittest.main()
