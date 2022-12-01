"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access

import unittest
import subprocess

class WhisperTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_whisper_cmd(self) -> None:
        """Check that the command is installed by the setup process."""
        subprocess.check_output("whisper", shell=True)


if __name__ == '__main__':
    unittest.main()
