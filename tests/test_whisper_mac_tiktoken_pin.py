"""Regression test for issues #66 and #67.

lightning-whisper-mlx==0.0.10 hard-pins tiktoken==0.3.3. If our generated
pyproject.toml adds any conflicting tiktoken constraint (e.g. tiktoken>=0.5.0),
uv refuses to solve the dependency graph and the MLX backend fails before it
ever runs.

This test pins down the constraint at the pyproject-string level so a future
edit can't quietly re-introduce a conflicting tiktoken requirement.
"""

from __future__ import annotations

import re
import unittest

from transcribe_anything.whisper_mac import _make_pyproject_toml_content

# Matches any tiktoken requirement line, whether pinned, lower-bounded,
# upper-bounded, or anything in between.
TIKTOKEN_REQ = re.compile(r'"\s*tiktoken\s*(?P<op>[<>=!~][^"]*)?\s*"')


class WhisperMacPyprojectTester(unittest.TestCase):
    def test_no_conflicting_tiktoken_pin(self) -> None:
        content = _make_pyproject_toml_content()
        matches = TIKTOKEN_REQ.findall(content)
        for op in matches:
            # An empty operator string means "tiktoken" with no version
            # constraint, which is fine — uv will resolve it via the transitive
            # lightning-whisper-mlx pin to 0.3.3.
            self.assertEqual(
                op,
                "",
                msg=(
                    f"whisper_mac.py pyproject pins tiktoken with '{op}', but "
                    "lightning-whisper-mlx==0.0.10 hard-pins tiktoken==0.3.3. "
                    "Any non-empty constraint here breaks dependency resolution "
                    "(issues #66, #67)."
                ),
            )


if __name__ == "__main__":
    unittest.main()
