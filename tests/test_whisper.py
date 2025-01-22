"""
Tests whisper ai cmd
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801

import subprocess
import unittest

from transcribe_anything.whisper import get_environment


class WhisperTester(unittest.TestCase):
    """Tester for whisper ai."""

    def test_whisper_cmd(self) -> None:
        """Check that the command is installed by the setup process."""
        env = get_environment()
        try:
            env.run(["whisper", "--help"])
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise e


if __name__ == "__main__":
    unittest.main()
