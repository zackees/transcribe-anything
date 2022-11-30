"""
Tests transcribe_anything
"""

# pylint: disable=no-self-use,protected-access

import subprocess
import unittest


class TranscribeAnythingTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_fetch_command_help(self) -> None:
        """Check that the command is installed by the setup process."""
        subprocess.check_output(['transcribe_anything', '-h'])

    def test_fetch_command_installed(self) -> None:
        """Check that the command works on a live short video."""
        subprocess.check_output([
            'transcribe_anything',
            'https://www.youtube.com/watch?v=8Wg8f2g_GQY',
        ])


if __name__ == '__main__':
    unittest.main()
