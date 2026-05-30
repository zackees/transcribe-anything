"""Regression tests for issue #69 (speechbrain k2_fsa ImportError).

speechbrain>=1.0's ``integrations/k2_fsa/__init__.py`` does ``import k2`` at
module load. The real ``k2`` package has no installable wheels on Windows
(and is hard to build), so on Windows + diarization the import raises and
crashes the entire insane backend even though pyannote doesn't actually
need k2 features.

This module locks two contracts:

1. transcribe-anything ships an empty ``k2`` stub package under
   ``transcribe_anything._k2_stub/k2/`` whose import succeeds with no
   side effects.
2. ``run_insanely_fast_whisper`` prepends the stub root to ``PYTHONPATH``
   in the subprocess env so the venv's Python finds it on
   ``import k2`` — without shadowing a real ``k2`` already on
   ``sys.path`` (which is why we put the stub at the END, not the front).
"""

from __future__ import annotations

import importlib.util
import os
import unittest

from transcribe_anything.insanely_fast_whisper import (
    _k2_stub_root,
    _prepare_subprocess_env,
)


class K2StubExistsTester(unittest.TestCase):
    def test_stub_root_is_real_directory_with_k2_package(self) -> None:
        root = _k2_stub_root()
        self.assertTrue(root.is_dir(), msg=f"{root} should be a directory")
        self.assertTrue(
            (root / "k2" / "__init__.py").is_file(),
            msg=f"Expected k2/__init__.py under {root}",
        )

    def test_stub_k2_module_loads_without_side_effects(self) -> None:
        # Loading the stub directly from its file confirms `import k2` would
        # succeed in any subprocess that has the stub root on its sys.path.
        init_path = _k2_stub_root() / "k2" / "__init__.py"
        spec = importlib.util.spec_from_file_location("transcribe_anything_k2_stub", init_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Empty docstring-only module: no surprise attributes.
        public = [a for a in dir(module) if not a.startswith("_")]
        self.assertEqual(public, [], msg=f"Stub should be empty; found {public}")


class PrepareSubprocessEnvTester(unittest.TestCase):
    def test_pythonpath_includes_stub_root_when_unset(self) -> None:
        env = _prepare_subprocess_env({})
        stub = str(_k2_stub_root())
        self.assertIn("PYTHONPATH", env)
        self.assertIn(stub, env["PYTHONPATH"].split(os.pathsep))

    def test_pythonpath_preserves_existing_entries_first(self) -> None:
        existing = os.path.join("some", "existing", "path")
        env = _prepare_subprocess_env({"PYTHONPATH": existing})
        parts = env["PYTHONPATH"].split(os.pathsep)
        stub = str(_k2_stub_root())
        # Existing path stays first so a real k2 installed there wins.
        self.assertEqual(parts[0], existing)
        self.assertIn(stub, parts)

    def test_does_not_overwrite_unrelated_env_vars(self) -> None:
        env = _prepare_subprocess_env({"FOO": "bar", "PATH": "/usr/bin"})
        self.assertEqual(env["FOO"], "bar")
        self.assertEqual(env["PATH"], "/usr/bin")


if __name__ == "__main__":
    unittest.main()
