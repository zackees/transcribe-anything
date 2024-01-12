



"""
Tests transcribe_anything
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import unittest
import json

from transcribe_anything.insanely_fast_whisper import convert_json_to_srt, convert_json_to_text

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

class JsonToSrtTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_json_to_srt(self) -> None:
        """Check that the command works on a local file."""
        data = json.loads(EXAMPLE_JSON)
        out = convert_json_to_srt(data)
        print(out)
        print()

    def test_json_to_txt(self) -> None:
        """Check that the command works on a local file."""
        data = json.loads(EXAMPLE_JSON)
        out = convert_json_to_text(data)
        print(out)
        print()


if __name__ == "__main__":
    unittest.main()
