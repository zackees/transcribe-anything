"""
Tests that fetch_audio() surfaces ffmpeg failures on local files.
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801

import os
import subprocess
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcribe_anything.audio import fetch_audio


class FetchAudioLocalFfmpegTester(unittest.TestCase):
    """fetch_audio local-file branch must raise CalledProcessError when ffmpeg fails."""

    def test_local_ffmpeg_failure_raises_called_process_error(self) -> None:
        """A failed local ffmpeg run must raise CalledProcessError, not FileNotFoundError."""

        def fake_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=cmd,
                    output=b"",
                    stderr=b"Output file does not contain any stream\n",
                )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout=b"",
                stderr=b"Output file does not contain any stream\n",
            )

        with TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "silent.mp4")
            with open(src_path, "wb") as handle:
                handle.write(b"not-a-real-video")
            out_wav = os.path.join(tmpdir, "out.wav")

            with patch("transcribe_anything.audio.shutil.which", return_value="/usr/bin/static_ffmpeg"):
                with patch("transcribe_anything.audio.subprocess.run", side_effect=fake_run):
                    with self.assertRaises(subprocess.CalledProcessError) as ctx:
                        fetch_audio(src_path, out_wav)

        self.assertEqual(ctx.exception.returncode, 1)
        self.assertIn(b"does not contain any stream", ctx.exception.stderr)

    def test_local_ffmpeg_failure_non_utf8_stderr_still_raises_called_process_error(
        self,
    ) -> None:
        """Non-UTF-8 ffmpeg output must still raise CalledProcessError, not UnicodeDecodeError."""

        def fake_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                output=b"\xff\xfe",
                stderr=b"\xff Output file does not contain any stream\n",
            )

        with TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "silent.mp4")
            with open(src_path, "wb") as handle:
                handle.write(b"not-a-real-video")
            out_wav = os.path.join(tmpdir, "out.wav")

            with patch(
                "transcribe_anything.audio.shutil.which",
                return_value="/usr/bin/static_ffmpeg",
            ):
                with patch(
                    "transcribe_anything.audio.subprocess.run", side_effect=fake_run
                ):
                    with self.assertRaises(subprocess.CalledProcessError) as ctx:
                        fetch_audio(src_path, out_wav)

        self.assertEqual(ctx.exception.returncode, 1)
        self.assertIn(b"does not contain any stream", ctx.exception.stderr)


if __name__ == "__main__":
    unittest.main()
