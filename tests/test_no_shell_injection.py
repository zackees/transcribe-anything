"""
Regression tests: URLs must never be interpolated into a shell string.

``ytldp_download`` and ``get_video_name_from_url`` both take a caller-supplied
URL. When that URL was pasted into a shell command string, any shell
metacharacter in it (`&`, `;`, `|`, backticks, `$(...)`) was interpreted by the
shell instead of being handed to yt-dlp. Beyond the security impact this also
broke ordinary URLs -- share links carrying `?a=1&b=2` query strings split at
the `&`.

These tests pin the fix: argv lists, ``shell=False``, and the URL delivered to
yt-dlp byte-for-byte.
"""

import subprocess
import tempfile
import unittest
from unittest.mock import patch

from transcribe_anything.api import get_video_name_from_url
from transcribe_anything.ytldp_download import ytdlp_download

# A URL shaped like a real share link (`&` query string) that also carries
# characters a shell would act on. No payload -- we only assert it survives
# transport intact.
HOSTILE_URL = "https://example.com/watch?a=1&b=2;echo marker`id`$(id)"


class NoShellInjectionTester(unittest.TestCase):
    """Verify subprocess calls that carry a URL do not go through a shell."""

    def test_ytdlp_download_does_not_use_shell(self) -> None:
        """ytdlp_download must pass argv, not a shell string."""
        with patch("transcribe_anything.ytldp_download.subprocess.run") as mock_run, tempfile.TemporaryDirectory() as tmpdir:
            # The mocked run downloads nothing, so the post-download
            # "exactly one file" assertion fires. We only care about how the
            # command was built, which has already happened by then.
            with self.assertRaises(AssertionError):
                ytdlp_download(HOSTILE_URL, tmpdir)

            self.assertTrue(mock_run.called, "subprocess.run was never invoked")
            args, kwargs = mock_run.call_args
            cmd = args[0]

            self.assertIsInstance(cmd, list, f"command must be an argv list, got {type(cmd).__name__}")
            self.assertFalse(kwargs.get("shell", False), "shell=True re-introduces the injection")
            # The URL must arrive as exactly one argv element, unmodified.
            self.assertIn(HOSTILE_URL, cmd)
            # And the -o template must not carry literal quote characters --
            # with shell=False there is no shell to strip them.
            self.assertIn("out.%(ext)s", cmd)

    def test_get_video_name_from_url_does_not_use_shell(self) -> None:
        """get_video_name_from_url must pass argv, not a shell string."""
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="A Title\n", stderr="")
        with patch("transcribe_anything.api.subprocess.run", return_value=completed) as mock_run:
            get_video_name_from_url(HOSTILE_URL)

            self.assertTrue(mock_run.called, "subprocess.run was never invoked")
            args, kwargs = mock_run.call_args
            cmd = args[0]

            self.assertIsInstance(cmd, list, f"command must be an argv list, got {type(cmd).__name__}")
            self.assertFalse(kwargs.get("shell", False), "shell=True re-introduces the injection")
            self.assertIn(HOSTILE_URL, cmd)


if __name__ == "__main__":
    unittest.main()
