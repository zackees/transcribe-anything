"""Regression test: pin transformers >= 4.53.0 in the insane backend venv.

Three upstream Whisper bugs that affect ``--device insane``'s timestamps
are fixed in successive transformers releases:

* WhisperTokenizer chunk-offset decode — PR #34537, first in v4.47.0
* Mac MPS float64 regression — PR #35295, first in v4.48.0
* Long-form sequential decoding 30-s rollover — PR #35750, first in v4.53.0

Anything below 4.53.0 is missing at least one of these. This test locks
the floor so a future contributor cannot silently regress below the
line where the upstream fixes live.
"""

from __future__ import annotations

import re
import unittest

from transcribe_anything.insanley_fast_whisper_reqs import _get_reqs_generic


TRANSFORMERS_LINE = re.compile(r"^transformers==(?P<v>\d+\.\d+\.\d+)$")
MIN_VERSION = (4, 53, 0)


def _find_transformers_pin(deps: list[str]) -> tuple[int, int, int] | None:
    for dep in deps:
        m = TRANSFORMERS_LINE.match(dep.strip())
        if m:
            parts = m.group("v").split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
    return None


class TransformersFloorTester(unittest.TestCase):
    def test_pin_is_exact_equals_style(self) -> None:
        """The file convention is == point pins; a future loose pin (>= / ~=)
        would defeat the floor guarantee silently."""
        deps = _get_reqs_generic(has_nvidia=True)
        pinned = [d for d in deps if d.strip().startswith("transformers")]
        self.assertEqual(
            len(pinned),
            1,
            msg=f"Expected exactly one transformers pin, got: {pinned}",
        )
        self.assertRegex(
            pinned[0].strip(),
            r"^transformers==\d+\.\d+\.\d+$",
            msg=f"Expected strict '==' pin matching file convention; got {pinned[0]!r}",
        )

    def test_pin_floor_at_least_4_53_0_with_nvidia(self) -> None:
        deps = _get_reqs_generic(has_nvidia=True)
        version = _find_transformers_pin(deps)
        assert version is not None, f"No transformers pin found in deps: {deps}"
        self.assertGreaterEqual(
            version,
            MIN_VERSION,
            msg=(
                f"transformers pin {version} is below {MIN_VERSION}, which is the "
                "minimum that contains the Whisper long-form chunked-decoding "
                "timestamp-offset fix (PR #35750 → v4.53.0). Anything older still "
                "exhibits the segment-rollover-every-30s bug on --device insane."
            ),
        )

    def test_pin_floor_at_least_4_53_0_without_nvidia(self) -> None:
        # Same floor regardless of GPU; the bug lives in the transformers
        # library and is platform-agnostic.
        deps = _get_reqs_generic(has_nvidia=False)
        version = _find_transformers_pin(deps)
        assert version is not None, f"No transformers pin found in deps: {deps}"
        self.assertGreaterEqual(version, MIN_VERSION)


if __name__ == "__main__":
    unittest.main()
