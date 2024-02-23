"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import unittest

from transcribe_anything.insanley_fast_whisper_reqs import get_environment


class InsanelFastWhisperTesterDeps(unittest.TestCase):
    """Tester for transcribe anything."""

    def test(self) -> None:
        """Check that the command works on a local file."""
        get_environment()


if __name__ == "__main__":
    unittest.main()
