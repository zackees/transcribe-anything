"""
Tests for initial_prompt functionality
"""

import os
import tempfile
import unittest
from pathlib import Path

from transcribe_anything.api import transcribe


class TestInitialPrompt(unittest.TestCase):
    """Test initial prompt functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent
        self.localfile_dir = self.test_dir / "localfile"
        self.test_wav = self.localfile_dir / "video.wav"

    def test_initial_prompt_parameter(self):
        """Test that initial_prompt parameter is accepted and processed."""
        if not self.test_wav.exists():
            self.skipTest(f"Test file {self.test_wav} not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = transcribe(
                url_or_file=str(self.test_wav),
                output_dir=tmpdir,
                model="tiny",  # Use smallest model for speed
                initial_prompt="The speaker discusses technology and artificial intelligence."
            )
            
            # Check that output files were created
            self.assertTrue(os.path.exists(output_dir))
            srt_file = os.path.join(output_dir, "out.srt")
            self.assertTrue(os.path.exists(srt_file))

    def test_initial_prompt_with_other_args(self):
        """Test that initial_prompt works alongside other_args."""
        if not self.test_wav.exists():
            self.skipTest(f"Test file {self.test_wav} not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = transcribe(
                url_or_file=str(self.test_wav),
                output_dir=tmpdir,
                model="tiny",
                initial_prompt="Technology discussion with AI terms.",
                other_args=["--word_timestamps", "True"]
            )
            
            # Check that output files were created
            self.assertTrue(os.path.exists(output_dir))
            srt_file = os.path.join(output_dir, "out.srt")
            self.assertTrue(os.path.exists(srt_file))

    def test_no_initial_prompt(self):
        """Test that transcription works without initial_prompt (backward compatibility)."""
        if not self.test_wav.exists():
            self.skipTest(f"Test file {self.test_wav} not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = transcribe(
                url_or_file=str(self.test_wav),
                output_dir=tmpdir,
                model="tiny"
            )
            
            # Check that output files were created
            self.assertTrue(os.path.exists(output_dir))
            srt_file = os.path.join(output_dir, "out.srt")
            self.assertTrue(os.path.exists(srt_file))


if __name__ == "__main__":
    unittest.main()
