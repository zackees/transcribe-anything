"""
Tests for the PyTorch/CUDA version upgrade to 2.7.0+cu128 (Blackwell sm_120 support).

Validates that:
- Generated pyproject.toml content contains correct versions for both venvs
- NVIDIA and non-NVIDIA code paths produce correct output
- CUDA version references are consistent across the codebase
- PyTorch wheel index URLs are reachable
- Dockerfile base image matches the expected versions
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Project root and source paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "transcribe_anything"


# ---------------------------------------------------------------------------
# Helper: capture the pyproject.toml string that gets passed to PyProjectToml
# ---------------------------------------------------------------------------


def _capture_whisper_pyproject(has_nvidia: bool) -> str:
    """Generate the whisper venv pyproject.toml content and return it as a string."""
    captured = {}

    class FakePyProjectToml:
        def __init__(self, content: str):
            captured["content"] = content

    class FakeIsoEnvArgs:
        def __init__(self, *a, **kw):
            pass

    class FakeIsoEnv:
        def __init__(self, *a, **kw):
            pass

    with (
        patch("transcribe_anything.whisper.PyProjectToml", FakePyProjectToml),
        patch("transcribe_anything.whisper.IsoEnvArgs", FakeIsoEnvArgs),
        patch("transcribe_anything.whisper.IsoEnv", FakeIsoEnv),
        patch("transcribe_anything.whisper.has_nvidia_smi", return_value=has_nvidia),
    ):
        from transcribe_anything.whisper import get_environment

        get_environment()

    return captured["content"]


def _capture_insane_pyproject(has_nvidia: bool) -> str:
    """Generate the insanely_fast_whisper venv pyproject.toml content and return it."""
    captured = {}

    class FakePyProjectToml:
        def __init__(self, content: str):
            captured["content"] = content

    class FakeIsoEnvArgs:
        def __init__(self, *a, **kw):
            pass

    class FakeIsoEnv:
        def __init__(self, *a, **kw):
            pass

    with (
        patch("transcribe_anything.insanley_fast_whisper_reqs.PyProjectToml", FakePyProjectToml),
        patch("transcribe_anything.insanley_fast_whisper_reqs.IsoEnvArgs", FakeIsoEnvArgs),
        patch("transcribe_anything.insanley_fast_whisper_reqs.IsoEnv", FakeIsoEnv),
    ):
        from transcribe_anything.insanley_fast_whisper_reqs import get_environment

        get_environment(has_nvidia=has_nvidia)

    return captured["content"]


# ===========================================================================
# 1. Whisper venv pyproject.toml generation
# ===========================================================================


class TestWhisperPyprojectGeneration(unittest.TestCase):
    """Verify the whisper venv generates correct pyproject.toml for NVIDIA / non-NVIDIA."""

    def test_nvidia_pyproject_has_cu128_torch(self):
        content = _capture_whisper_pyproject(has_nvidia=True)
        self.assertIn("torch==2.7.0+cu128", content)

    def test_nvidia_pyproject_has_cu128_index(self):
        content = _capture_whisper_pyproject(has_nvidia=True)
        self.assertIn("https://download.pytorch.org/whl/cu128", content)

    def test_nvidia_pyproject_has_cu128_source_name(self):
        content = _capture_whisper_pyproject(has_nvidia=True)
        self.assertIn("pytorch-cu128", content)

    def test_nvidia_pyproject_no_old_cu121(self):
        content = _capture_whisper_pyproject(has_nvidia=True)
        self.assertNotIn("cu121", content)
        self.assertNotIn("cu126", content)

    def test_no_nvidia_pyproject_has_no_cuda_torch(self):
        content = _capture_whisper_pyproject(has_nvidia=False)
        self.assertNotIn("+cu", content)
        self.assertNotIn("pytorch-cu", content)

    def test_no_nvidia_pyproject_has_openai_whisper(self):
        content = _capture_whisper_pyproject(has_nvidia=False)
        self.assertIn("openai-whisper", content)

    def test_nvidia_pyproject_is_valid_toml_structure(self):
        content = _capture_whisper_pyproject(has_nvidia=True)
        self.assertIn("[build-system]", content)
        self.assertIn("[project]", content)
        self.assertIn("[tool.uv.sources]", content)
        self.assertIn("[[tool.uv.index]]", content)


# ===========================================================================
# 2. Insanely-fast-whisper venv pyproject.toml generation
# ===========================================================================


class TestInsanePyprojectGeneration(unittest.TestCase):
    """Verify insanely_fast_whisper venv generates correct pyproject.toml."""

    def test_nvidia_pyproject_has_cu128_torch(self):
        content = _capture_insane_pyproject(has_nvidia=True)
        self.assertIn("torch==2.7.0+cu128", content)

    def test_nvidia_pyproject_has_cu128_torchaudio(self):
        content = _capture_insane_pyproject(has_nvidia=True)
        self.assertIn("torchaudio==2.7.0+cu128", content)

    def test_nvidia_pyproject_has_cu128_index(self):
        content = _capture_insane_pyproject(has_nvidia=True)
        self.assertIn("https://download.pytorch.org/whl/cu128", content)

    def test_nvidia_pyproject_has_cu128_source_name(self):
        content = _capture_insane_pyproject(has_nvidia=True)
        self.assertIn("pytorch-cu128", content)

    def test_nvidia_pyproject_torch_and_torchaudio_both_use_cu128_index(self):
        content = _capture_insane_pyproject(has_nvidia=True)
        # Both torch and torchaudio should reference the cu128 index
        torch_src = content.count("{ index = 'pytorch-cu128' }")
        self.assertEqual(torch_src, 2, "Expected torch AND torchaudio to use pytorch-cu128 index")

    def test_nvidia_pyproject_no_old_versions(self):
        content = _capture_insane_pyproject(has_nvidia=True)
        self.assertNotIn("cu121", content)
        self.assertNotIn("cu126", content)
        self.assertNotIn("2.6.0", content)
        self.assertNotIn("2.2.1", content)

    def test_no_nvidia_pyproject_has_cpu_torch(self):
        content = _capture_insane_pyproject(has_nvidia=False)
        self.assertIn("torch==2.7.0", content)
        self.assertNotIn("+cu", content)

    def test_no_nvidia_pyproject_has_cpu_torchaudio(self):
        content = _capture_insane_pyproject(has_nvidia=False)
        self.assertIn("torchaudio==2.7.0", content)

    def test_no_nvidia_pyproject_no_cuda_index(self):
        content = _capture_insane_pyproject(has_nvidia=False)
        self.assertNotIn("pytorch-cu", content)
        self.assertNotIn("tool.uv.sources", content)

    def test_nvidia_pyproject_is_valid_toml_structure(self):
        content = _capture_insane_pyproject(has_nvidia=True)
        self.assertIn("[build-system]", content)
        self.assertIn("[project]", content)
        self.assertIn("[tool.uv.sources]", content)
        self.assertIn("[[tool.uv.index]]", content)


# ===========================================================================
# 3. Module-level constants are correct
# ===========================================================================


class TestModuleConstants(unittest.TestCase):
    """Verify the version constants in both modules are set correctly."""

    def test_whisper_tensor_version(self):
        from transcribe_anything.whisper import TENSOR_VERSION

        self.assertEqual(TENSOR_VERSION, "2.7.0")

    def test_whisper_cuda_version(self):
        from transcribe_anything.whisper import CUDA_VERSION

        self.assertEqual(CUDA_VERSION, "cu128")

    def test_whisper_extra_index_url(self):
        from transcribe_anything.whisper import EXTRA_INDEX_URL

        self.assertEqual(EXTRA_INDEX_URL, "https://download.pytorch.org/whl/cu128")

    def test_insane_tensor_version(self):
        from transcribe_anything.insanley_fast_whisper_reqs import TENSOR_VERSION

        self.assertEqual(TENSOR_VERSION, "2.7.0")

    def test_insane_cuda_version(self):
        from transcribe_anything.insanley_fast_whisper_reqs import CUDA_VERSION

        self.assertEqual(CUDA_VERSION, "cu128")

    def test_insane_tensor_cuda_version(self):
        from transcribe_anything.insanley_fast_whisper_reqs import TENSOR_CUDA_VERSION

        self.assertEqual(TENSOR_CUDA_VERSION, "2.7.0+cu128")

    def test_insane_extra_index_url(self):
        from transcribe_anything.insanley_fast_whisper_reqs import EXTRA_INDEX_URL

        self.assertEqual(EXTRA_INDEX_URL, "https://download.pytorch.org/whl/cu128")


# ===========================================================================
# 4. Cross-codebase CUDA version consistency
# ===========================================================================


class TestCudaVersionConsistency(unittest.TestCase):
    """Ensure all non-archived source files reference the same CUDA version."""

    EXPECTED_CUDA_SHORT = "12.8"
    EXPECTED_CU_TAG = "cu128"

    # Files that should have been updated (skip archive/)
    SOURCE_FILES = [
        SRC_DIR / "whisper.py",
        SRC_DIR / "insanley_fast_whisper_reqs.py",
        SRC_DIR / "insanely_fast_whisper.py",
        SRC_DIR / "cuda_available.py",
        SRC_DIR / "util.py",
    ]

    def test_no_stale_cu121_references(self):
        """No active source file should still reference cu121."""
        for path in self.SOURCE_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("cu121", text, f"{path.name} still contains 'cu121'")

    def test_no_stale_cu126_references(self):
        """No active source file should still reference cu126."""
        for path in self.SOURCE_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("cu126", text, f"{path.name} still contains 'cu126'")

    def test_no_stale_cuda_12_6_diagnostic_strings(self):
        """Diagnostic/error strings should reference 12.8, not 12.6."""
        for path in self.SOURCE_FILES:
            text = path.read_text(encoding="utf-8")
            # Look for hardcoded "12.6" in string literals (not comments)
            # We check for patterns like '12.6' or "12.6" that indicate hardcoded versions
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Check for "12.6" in string literals
                if '"12.6"' in line or "'12.6'" in line or "12.6+" in line:
                    # Allow the comment lines
                    if not stripped.startswith("#"):
                        self.fail(f"{path.name}:{i} still contains hardcoded '12.6': {stripped}")

    def test_no_stale_pytorch_2_2_1_references(self):
        """No active source file should reference old PyTorch 2.2.1."""
        for path in self.SOURCE_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("2.2.1", text, f"{path.name} still contains '2.2.1'")

    def test_no_stale_pytorch_2_6_0_references(self):
        """No active source file should reference old PyTorch 2.6.0."""
        for path in self.SOURCE_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("2.6.0", text, f"{path.name} still contains '2.6.0'")

    def test_cuda_available_diagnostics_reference_12_8(self):
        """cuda_available.py diagnostic messages should mention 12.8."""
        text = (SRC_DIR / "cuda_available.py").read_text(encoding="utf-8")
        self.assertIn("12.8", text)

    def test_util_default_expected_cuda_is_12_8(self):
        """util.py print_cuda_diagnostics default should be 12.8."""
        text = (SRC_DIR / "util.py").read_text(encoding="utf-8")
        self.assertIn('expected_cuda: str = "12.8"', text)

    def test_insanely_fast_whisper_passes_12_8_to_diagnostics(self):
        """insanely_fast_whisper.py should pass expected_cuda='12.8'."""
        text = (SRC_DIR / "insanely_fast_whisper.py").read_text(encoding="utf-8")
        self.assertIn('expected_cuda="12.8"', text)


# ===========================================================================
# 5. Dockerfile consistency
# ===========================================================================


class TestDockerfileConsistency(unittest.TestCase):
    """Verify Dockerfile base image matches the PyTorch/CUDA versions in source."""

    def test_dockerfile_base_image_version(self):
        dockerfile = PROJECT_ROOT / "Dockerfile"
        if not dockerfile.exists():
            self.skipTest("Dockerfile not found")
        text = dockerfile.read_text(encoding="utf-8")
        self.assertIn("pytorch/pytorch:2.7.0-cuda12.8", text)

    def test_dockerfile_no_old_base_images(self):
        dockerfile = PROJECT_ROOT / "Dockerfile"
        if not dockerfile.exists():
            self.skipTest("Dockerfile not found")
        text = dockerfile.read_text(encoding="utf-8")
        self.assertNotIn("cuda12.6", text)
        self.assertNotIn("cuda12.1", text)
        self.assertNotIn("2.6.0", text)
        self.assertNotIn("2.2.1", text)


# ===========================================================================
# 6. PyTorch wheel index URL is reachable
# ===========================================================================


class TestPyTorchWheelUrl(unittest.TestCase):
    """Verify that the PyTorch cu128 wheel index URL is reachable."""

    def test_cu128_index_url_reachable(self):
        """HEAD request to the cu128 wheel index should return 200."""
        import urllib.error
        import urllib.request

        url = "https://download.pytorch.org/whl/cu128/"
        req = urllib.request.Request(url, method="HEAD")
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            self.assertIn(resp.status, (200, 301, 302), f"Expected 200/redirect from {url}, got {resp.status}")
        except urllib.error.HTTPError as e:
            self.fail(f"cu128 wheel index returned HTTP {e.code}: {url}")
        except urllib.error.URLError as e:
            self.skipTest(f"Network unavailable: {e}")

    def test_cu128_torch_2_7_0_wheel_exists(self):
        """Verify that torch 2.7.0 cu128 wheel page exists."""
        import urllib.error
        import urllib.request

        url = "https://download.pytorch.org/whl/cu128/torch/"
        req = urllib.request.Request(url, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            body = resp.read().decode("utf-8", errors="replace")
            self.assertIn("2.7.0", body, "torch 2.7.0 wheel not found in cu128 index")
        except urllib.error.HTTPError as e:
            self.fail(f"cu128 torch index returned HTTP {e.code}: {url}")
        except urllib.error.URLError as e:
            self.skipTest(f"Network unavailable: {e}")


# ===========================================================================
# 7. cuda_available.py diagnostic output validation
# ===========================================================================


class TestCudaAvailableDiagnostics(unittest.TestCase):
    """Verify cuda_available.py _print_cuda_diagnostics references correct version."""

    def _run_diagnostics(self, nvidia_smi_output: str) -> str:
        """Run _print_cuda_diagnostics with mocked nvidia-smi and capture stderr."""
        from io import StringIO

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = nvidia_smi_output

        captured = StringIO()
        with (
            patch("transcribe_anything.cuda_available.subprocess.run", return_value=mock_result),
            patch("transcribe_anything.cuda_available.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("sys.stderr", captured),
        ):
            from transcribe_anything.cuda_available import _print_cuda_diagnostics

            _print_cuda_diagnostics()

        return captured.getvalue()

    def test_diagnostics_show_12_8_bundled(self):
        nvidia_output = "Driver Version: 570.86.10    CUDA Version: 12.8\n" "GPU  NVIDIA GeForce RTX 5090\n"
        output = self._run_diagnostics(nvidia_output)
        self.assertIn("12.8", output)
        self.assertNotIn("12.6", output)

    def test_diagnostics_old_driver_error_mentions_12_8(self):
        nvidia_output = "Driver Version: 520.00.00    CUDA Version: 11.8\n" "GPU  NVIDIA GeForce RTX 3090\n"
        output = self._run_diagnostics(nvidia_output)
        self.assertIn("12.8", output)


# ===========================================================================
# 8. util.py print_cuda_diagnostics validation
# ===========================================================================


class TestUtilCudaDiagnostics(unittest.TestCase):
    """Verify util.py print_cuda_diagnostics uses correct default and messages."""

    NVIDIA_SMI_CUDA_128 = "Driver Version: 570.86.10    CUDA Version: 12.8\n" "GPU  NVIDIA GeForce RTX 5090\n"
    NVIDIA_SMI_CUDA_118 = "Driver Version: 520.00.00    CUDA Version: 11.8\n" "GPU  NVIDIA GeForce RTX 3090\n"

    def setUp(self):
        import transcribe_anything.util as util_mod

        self._mod = util_mod
        self._mod._NVIDIA_DRIVER_INFO_CACHE = None
        self._mod._NVIDIA_DRIVER_INFO_CHECKED = False

    def tearDown(self):
        self._mod._NVIDIA_DRIVER_INFO_CACHE = None
        self._mod._NVIDIA_DRIVER_INFO_CHECKED = False

    def _run(self, nvidia_output: str, expected_cuda: str | None = None) -> str:
        from io import StringIO

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = nvidia_output

        captured = StringIO()
        with (
            patch.object(self._mod.subprocess, "run", return_value=mock_result),
            patch.object(self._mod.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch("sys.stderr", captured),
        ):
            if expected_cuda is not None:
                self._mod.print_cuda_diagnostics(expected_cuda=expected_cuda)
            else:
                # Use default parameter
                self._mod.print_cuda_diagnostics()
        return captured.getvalue()

    def test_default_expected_cuda_is_12_8(self):
        output = self._run(self.NVIDIA_SMI_CUDA_128)
        self.assertIn("12.8", output)

    def test_old_driver_error_message_says_12_8(self):
        output = self._run(self.NVIDIA_SMI_CUDA_118)
        self.assertIn("12.8", output)
        self.assertNotIn("12.6+", output)


if __name__ == "__main__":
    unittest.main()
