"""Diagnostic-quality tests for parse_whisper_options().

This module locks down two UX guarantees:

1. When the underlying ``whisper --help`` subprocess exits non-zero (issue #52
   on M1: silent ``CalledProcessError: returned non-zero exit status 2``), the
   caller must get an actionable ``RuntimeError`` whose message includes the
   subprocess stderr so users have something to act on.
2. ``parse_whisper_options`` must NOT capture stderr — uv prints download /
   build progress to stderr on first run, and capturing it makes the call
   appear to hang silently (issue #40 on Ubuntu 24).
"""

from __future__ import annotations

import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from transcribe_anything import parse_whisper_options as pwo_module


def _fake_env(returncode: int, stdout: str = "", stderr: str = "") -> mock.MagicMock:
    env = mock.MagicMock()
    env.run.return_value = SimpleNamespace(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    return env


class ParseWhisperOptionsDiagnosticsTester(unittest.TestCase):
    def test_nonzero_exit_raises_runtime_error_with_stderr(self) -> None:
        """Issue #52: surface whisper/uv stderr instead of opaque exit-2."""
        stderr_text = "error: unrecognized arguments: --some-flag-that-broke"
        fake_env = _fake_env(returncode=2, stdout="", stderr=stderr_text)
        with mock.patch.object(pwo_module, "get_environment", return_value=fake_env):
            with self.assertRaises(RuntimeError) as ctx:
                pwo_module.parse_whisper_options()
        msg = str(ctx.exception)
        self.assertIn("whisper --help", msg)
        self.assertIn("2", msg)  # the exit code
        self.assertIn(stderr_text, msg)

    def test_stderr_not_captured_so_progress_is_visible(self) -> None:
        """Issue #40: env.run must NOT pass stderr=PIPE / capture_output=True.

        Without this, uv's first-run download progress is hidden and the
        process looks frozen for several minutes.
        """
        fake_env = _fake_env(returncode=0, stdout="", stderr="")
        with mock.patch.object(pwo_module, "get_environment", return_value=fake_env):
            pwo_module.parse_whisper_options()
        self.assertEqual(fake_env.run.call_count, 1)
        kwargs = fake_env.run.call_args.kwargs
        self.assertNotEqual(
            kwargs.get("stderr"),
            subprocess.PIPE,
            msg="parse_whisper_options must not redirect stderr to a PIPE; "
            "uv first-run progress would be hidden (issue #40).",
        )
        self.assertFalse(
            kwargs.get("capture_output", False),
            msg="capture_output=True implies stderr=PIPE; do not use it here.",
        )

    def test_happy_path_parses_options(self) -> None:
        sample_stdout = (
            "usage: whisper [-h] [--model {tiny,base}] [--language LANG]\n"
            "  [--task {transcribe,translate}]\n"
        )
        fake_env = _fake_env(returncode=0, stdout=sample_stdout, stderr="")
        with mock.patch.object(pwo_module, "get_environment", return_value=fake_env):
            data = pwo_module.parse_whisper_options()
        # The parser keys options by their long name; values come from the
        # {choice,choice} block when present.
        self.assertIn("model", data)
        self.assertEqual(data["model"], ["tiny", "base"])


if __name__ == "__main__":
    unittest.main()
