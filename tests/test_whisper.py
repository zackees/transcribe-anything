"""
Tests whisper ai cmd
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access

import unittest
import subprocess

from transcribe_anything.whisper import get_environment


class WhisperTester(unittest.TestCase):
    """Tester for whisper ai."""

    def test_whisper_cmd(self) -> None:
        """Check that the command is installed by the setup process."""
        env = get_environment().environment()
        subprocess.check_output(
            "whisper --help",
            shell=True,
            env=env,
        )


if __name__ == "__main__":
    unittest.main()
