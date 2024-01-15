"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import json
import unittest
from pathlib import Path

from transcribe_anything.insanely_fast_whisper import (
    convert_json_to_srt,
    convert_json_to_text,
)

HERE = Path(__file__).parent
LOCALFILE_DIR = HERE / "localfile"
PROBLEM_JSON = LOCALFILE_DIR / "problem.json"

EXAMPLE_JSON = """
{
  "speakers": [],
  "chunks": [
    {
      "timestamp": [
        0,
        3.24
      ],
      "text": " Oh wow, I'm so nervous."
    },
    {
      "timestamp": [
        3.24,
        6.56
      ],
      "text": " Gosh, these lights are so bright."
    },
    {
      "timestamp": [
        6.56,
        8.52
      ],
      "text": " Is this mic on?"
    },
    {
      "timestamp": [
        8.52,
        9.52
      ],
      "text": " Is there even a mic?"
    }
  ],
  "text": " Oh wow, I'm so nervous. Gosh, these lights are so bright. Is this mic on? Is there even a mic?"
}
"""

EXPECTED_SRT_FILE = """
1
00:00:00,000 --> 00:00:03,240
Oh wow, I'm so nervous.

2
00:00:03,240 --> 00:00:06,559
Gosh, these lights are so bright.

3
00:00:06,559 --> 00:00:08,519
Is this mic on?

4
00:00:08,519 --> 00:00:09,519
Is there even a mic?
""".strip()

EXPECTED_TXT_FILE = """
Oh wow, I'm so nervous. Gosh, these lights are so bright. Is this mic on? Is there even a mic?
""".strip()


class JsonToSrtTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_json_to_srt(self) -> None:
        """Check that the command works on a local file."""
        data = json.loads(EXAMPLE_JSON)
        out = convert_json_to_srt(data, 9.0)
        self.assertIn(EXPECTED_SRT_FILE, out)

    def test_json_to_txt(self) -> None:
        """Check that the command works on a local file."""
        data = json.loads(EXAMPLE_JSON)
        out = convert_json_to_text(data)
        self.assertIn(EXPECTED_TXT_FILE, out)


if __name__ == "__main__":
    unittest.main()
