"""
Test the parse speaker module.
"""

import json  # type: ignore
import os
import unittest
from pathlib import Path
from pprint import pprint

from transcribe_anything.generate_speaker_json import generate_speaker_json

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TEST_SRT = LOCALFILE_DIR / "speaker-test.json"


class SpeakerJsonTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_generate_speaker_json(self) -> None:
        """Check that the command works on a local file."""
        json_str = TEST_SRT.read_text()
        data = json.loads(json_str)
        segmented = generate_speaker_json(data)
        pprint(segmented)
        print()


if __name__ == "__main__":
    unittest.main()
