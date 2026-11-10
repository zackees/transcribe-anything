"""Robustness tests for generate_speaker_json with malformed chunks.

The module under test is loaded directly from its file so these tests
stay hermetic (importing the ``transcribe_anything`` package pulls in
heavy optional backends such as whisper/torch that are unrelated here).
"""

import importlib.util
import unittest
import warnings
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "transcribe_anything"
    / "generate_speaker_json.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_speaker_json_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_speaker_json = _load_module().generate_speaker_json


class MalformedSpeakerJsonTester(unittest.TestCase):
    """Malformed diarization chunks must be skipped, not crash the run."""

    def test_non_numeric_timestamp_is_skipped(self) -> None:
        """A chunk whose timestamp cannot be parsed is warned about and skipped."""
        data = {
            "speakers": [
                {"speaker": "A", "timestamp": ["bad", 1.0], "text": "hi"},
                {"speaker": "A", "timestamp": [1.0, 2.0], "text": "there"},
            ]
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = generate_speaker_json(data)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker"], "A")
        self.assertEqual(out[0]["text"], "there")

    def test_none_timestamp_is_skipped(self) -> None:
        """A chunk with a null timestamp is warned about and skipped."""
        data = {
            "speakers": [
                {"speaker": "B", "timestamp": None, "text": "oops"},
                {"speaker": "B", "timestamp": [2.0, 3.0], "text": "ok"},
            ]
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = generate_speaker_json(data)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "ok")


if __name__ == "__main__":
    unittest.main()
