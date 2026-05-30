"""Regression test for the perceived-hang UX in issue #40.

When the whisper venv hasn't been built yet, ``parse_whisper_options()`` will
trigger an iso-env install whose subprocess `uv venv` + `uv pip compile` run
with `capture_output=True` upstream — i.e. silent for several minutes. To
the user this looks like a hang and they Ctrl+C, producing the symptom
reported in #40 ("nothing shown and stuck").

This test pins the contract: when the venv directory does not exist,
``parse_whisper_options()`` writes an obvious "first-run install in
progress" banner to stderr BEFORE calling ``env.run`` so the user knows
to wait.
"""

from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest import mock

from transcribe_anything import parse_whisper_options as pwo_module


def _fake_env_run_ok(stdout: str = "") -> mock.MagicMock:
    env = mock.MagicMock()
    env.run.return_value = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    return env


class FirstRunBannerTester(unittest.TestCase):
    def test_banner_shown_when_venv_missing(self) -> None:
        fake_env = _fake_env_run_ok(stdout="")
        captured = io.StringIO()
        with mock.patch.object(pwo_module, "get_environment", return_value=fake_env):
            with mock.patch.object(pwo_module, "_whisper_venv_exists", return_value=False):
                with mock.patch("sys.stderr", captured):
                    pwo_module.parse_whisper_options()
        msg = captured.getvalue()
        # The exact wording can evolve; pin only the bits a confused user
        # needs to see.
        self.assertIn("first", msg.lower())
        self.assertIn("install", msg.lower())
        self.assertIn("minutes", msg.lower())

    def test_banner_not_shown_when_venv_exists(self) -> None:
        fake_env = _fake_env_run_ok(stdout="")
        captured = io.StringIO()
        with mock.patch.object(pwo_module, "get_environment", return_value=fake_env):
            with mock.patch.object(pwo_module, "_whisper_venv_exists", return_value=True):
                with mock.patch("sys.stderr", captured):
                    pwo_module.parse_whisper_options()
        msg = captured.getvalue()
        self.assertNotIn("first", msg.lower())
        self.assertNotIn("install", msg.lower())

    def test_banner_emitted_before_env_run(self) -> None:
        """Order matters: banner must precede env.run, not follow it,
        otherwise the user sits in silence for the whole install."""
        events: list[str] = []

        captured_stderr = io.StringIO()
        original_write = captured_stderr.write

        def recording_write(s: str) -> int:
            if "first" in s.lower():
                events.append("banner")
            return original_write(s)

        captured_stderr.write = recording_write  # type: ignore[method-assign]

        fake_env = mock.MagicMock()

        def fake_run(*_args, **_kwargs):
            events.append("env.run")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        fake_env.run.side_effect = fake_run

        with mock.patch.object(pwo_module, "get_environment", return_value=fake_env):
            with mock.patch.object(pwo_module, "_whisper_venv_exists", return_value=False):
                with mock.patch("sys.stderr", captured_stderr):
                    pwo_module.parse_whisper_options()

        self.assertEqual(events, ["banner", "env.run"])


if __name__ == "__main__":
    unittest.main()
