"""Regression test for issue #35.

PyTorch 2.7.0+cu128 dlopens libcusparseLt.so.0 at import time on Linux. That
library is NOT bundled with the torch wheel; it ships in the
nvidia-cusparselt-cu12 pip package. Without that dependency, the
insanely_fast_whisper venv imports torch, fails with::

    ImportError: libcusparseLt.so.0: cannot open shared object file

and silently disables CUDA — which is exactly what the user in #35 hit even
though their `cuda_available` probe reported True.

This test pins the dep at requirements-generation time so it can't be removed
without a deliberate edit.
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from transcribe_anything.insanley_fast_whisper_reqs import _get_reqs_generic


def _has_cusparselt(deps: list[str]) -> bool:
    return any(d.startswith("nvidia-cusparselt-cu12") for d in deps)


class CusparseltDepTester(unittest.TestCase):
    def test_linux_with_nvidia_includes_cusparselt(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            deps = _get_reqs_generic(has_nvidia=True)
        self.assertTrue(
            _has_cusparselt(deps),
            msg=(
                "On Linux + nvidia, the insanely_fast_whisper venv must include "
                "nvidia-cusparselt-cu12 so torch 2.7.0+cu128 can find "
                "libcusparseLt.so.0 at import time (issue #35). Current deps: "
                f"{deps}"
            ),
        )

    def test_linux_without_nvidia_omits_cusparselt(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            deps = _get_reqs_generic(has_nvidia=False)
        self.assertFalse(
            _has_cusparselt(deps),
            msg="No NVIDIA → no need for the CUDA-only cusparselt wheel.",
        )

    def test_darwin_with_nvidia_omits_cusparselt(self) -> None:
        # has_nvidia=True is degenerate on darwin but exercise the platform gate.
        with mock.patch.object(sys, "platform", "darwin"):
            deps = _get_reqs_generic(has_nvidia=True)
        self.assertFalse(
            _has_cusparselt(deps),
            msg="cusparselt is Linux-specific; macOS must not pull it in.",
        )


if __name__ == "__main__":
    unittest.main()
