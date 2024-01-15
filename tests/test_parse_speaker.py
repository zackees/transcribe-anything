"""
Test the parse speaker module.
"""


import os
import unittest
from pathlib import Path
from pprint import pprint

import json5 as json  # type: ignore

from transcribe_anything.generate_speaker_json import generate_speaker_json

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
LOCALFILE_DIR = HERE / "localfile"
TEST_SRT = LOCALFILE_DIR / "speaker-test.json"


class ParseSpeakerTest(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_srt_wrap(self) -> None:
        """Check that the command works on a local file."""
        json_str = TEST_SRT.read_text()
        data = json.loads(json_str)
        segmented = generate_speaker_json(data)
        pprint(segmented)
        print()


if __name__ == "__main__":
    unittest.main()
