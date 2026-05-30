"""Regression tests for issue #52.

PyTorch dropped macOS x86_64 wheels in 2.3+, so transcribe-anything's
whisper backend cannot install on x86_64 Macs — either real Intel Macs or
Apple Silicon hosts where Python is running under Rosetta translation
(common when the user has Intel Homebrew at /usr/local in addition to or
instead of arm64 Homebrew at /opt/homebrew).

Without an early check the failure surfaces deep inside uv as a torch
resolution error that's hard to interpret. The helper under test detects
both cases and returns an actionable message.

All inputs to ``detect_macos_x86_unsupported`` are injectable so this test
file runs on any host (Windows in CI here).
"""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from transcribe_anything.util import (
    _check_rosetta_translated,
    detect_macos_x86_unsupported,
)


class DetectMacosX86UnsupportedTester(unittest.TestCase):
    def test_returns_none_on_linux(self) -> None:
        msg = detect_macos_x86_unsupported(
            sys_platform="linux",
            machine="x86_64",
            rosetta_check=lambda: 1,  # would say Rosetta but we're not on Mac
        )
        self.assertIsNone(msg)

    def test_returns_none_on_windows(self) -> None:
        msg = detect_macos_x86_unsupported(
            sys_platform="win32",
            machine="AMD64",
            rosetta_check=lambda: None,
        )
        self.assertIsNone(msg)

    def test_returns_none_on_mac_arm64(self) -> None:
        msg = detect_macos_x86_unsupported(
            sys_platform="darwin",
            machine="arm64",
            rosetta_check=lambda: 0,
        )
        self.assertIsNone(msg)

    def test_mac_x86_under_rosetta_returns_rosetta_specific_message(self) -> None:
        msg = detect_macos_x86_unsupported(
            sys_platform="darwin",
            machine="x86_64",
            rosetta_check=lambda: 1,
        )
        assert msg is not None
        self.assertIn("Rosetta", msg)
        self.assertIn("arm64", msg)  # the suggested fix
        self.assertIn("52", msg)  # link to the issue for context

    def test_mac_x86_native_returns_intel_message(self) -> None:
        msg = detect_macos_x86_unsupported(
            sys_platform="darwin",
            machine="x86_64",
            rosetta_check=lambda: 0,
        )
        assert msg is not None
        self.assertIn("Intel", msg)
        self.assertNotIn("Rosetta", msg)

    def test_mac_x86_sysctl_unavailable_falls_back_to_intel_message(self) -> None:
        msg = detect_macos_x86_unsupported(
            sys_platform="darwin",
            machine="x86_64",
            rosetta_check=lambda: None,
        )
        assert msg is not None
        self.assertIn("Intel", msg)


class CheckRosettaTranslatedTester(unittest.TestCase):
    """Sanity checks on the subprocess-backed Rosetta probe.

    These don't actually invoke sysctl on the host — they patch subprocess.run.
    """

    def _stub_sysctl(self, returncode: int, stdout: str) -> mock.MagicMock:
        completed = mock.MagicMock()
        completed.returncode = returncode
        completed.stdout = stdout
        return completed

    def test_returns_1_when_sysctl_reports_translated(self) -> None:
        with mock.patch(
            "transcribe_anything.util.subprocess.run",
            return_value=self._stub_sysctl(0, "1\n"),
        ):
            self.assertEqual(_check_rosetta_translated(), 1)

    def test_returns_0_when_sysctl_reports_native(self) -> None:
        with mock.patch(
            "transcribe_anything.util.subprocess.run",
            return_value=self._stub_sysctl(0, "0\n"),
        ):
            self.assertEqual(_check_rosetta_translated(), 0)

    def test_returns_none_when_sysctl_missing(self) -> None:
        with mock.patch(
            "transcribe_anything.util.subprocess.run",
            side_effect=FileNotFoundError("sysctl: command not found"),
        ):
            self.assertIsNone(_check_rosetta_translated())

    def test_returns_none_on_sysctl_nonzero_exit(self) -> None:
        with mock.patch(
            "transcribe_anything.util.subprocess.run",
            return_value=self._stub_sysctl(1, ""),
        ):
            self.assertIsNone(_check_rosetta_translated())

    def test_returns_none_on_unparseable_output(self) -> None:
        with mock.patch(
            "transcribe_anything.util.subprocess.run",
            return_value=self._stub_sysctl(0, "garbage"),
        ):
            self.assertIsNone(_check_rosetta_translated())

    def test_returns_none_on_timeout(self) -> None:
        with mock.patch(
            "transcribe_anything.util.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["sysctl"], timeout=2),
        ):
            self.assertIsNone(_check_rosetta_translated())


if __name__ == "__main__":
    unittest.main()
