"""
Test CUDA diagnostics functionality (nvidia-smi parsing).
"""

import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path to import without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import importlib.util

spec = importlib.util.spec_from_file_location("util_module", Path(__file__).parent.parent / "src" / "transcribe_anything" / "util.py")
assert spec is not None
assert spec.loader is not None
util_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util_module)

# Sample nvidia-smi output for CUDA 12.6
NVIDIA_SMI_OUTPUT_CUDA_126 = """\
Mon Mar 23 10:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03    CUDA Version: 12.6       |
|-------------------------------------------+------------------------+---------------------+
| GPU  Name                 Persistence-M   | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 3090          Off | 00000000:01:00.0  On  |                  N/A |
+-------------------------------------------+------------------------+---------------------+
"""

# Sample nvidia-smi output for CUDA 13.0
NVIDIA_SMI_OUTPUT_CUDA_130 = """\
Mon Mar 23 10:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.10              Driver Version: 570.86.10    CUDA Version: 13.0       |
|-------------------------------------------+------------------------+---------------------+
| GPU  Name                 Persistence-M   | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4090          Off | 00000000:01:00.0  On  |                  N/A |
+-------------------------------------------+------------------------+---------------------+
"""


class TestNvidiaDriverInfo(unittest.TestCase):
    """Test nvidia-smi parsing."""

    def setUp(self):
        """Reset cache before each test."""
        util_module._NVIDIA_DRIVER_INFO_CACHE = None
        util_module._NVIDIA_DRIVER_INFO_CHECKED = False

    def tearDown(self):
        """Reset cache after each test."""
        util_module._NVIDIA_DRIVER_INFO_CACHE = None
        util_module._NVIDIA_DRIVER_INFO_CHECKED = False

    @patch.object(util_module.subprocess, "run")
    @patch.object(util_module.shutil, "which", return_value="/usr/bin/nvidia-smi")
    def test_parse_cuda_126(self, mock_which, mock_run):
        """Test parsing nvidia-smi output with CUDA 12.6."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = NVIDIA_SMI_OUTPUT_CUDA_126
        mock_run.return_value = mock_result

        info = util_module.get_nvidia_driver_info()
        self.assertIsNotNone(info)
        self.assertEqual(info.driver_version, "560.35.03")
        self.assertEqual(info.cuda_version, "12.6")

    @patch.object(util_module.subprocess, "run")
    @patch.object(util_module.shutil, "which", return_value="/usr/bin/nvidia-smi")
    def test_parse_cuda_130(self, mock_which, mock_run):
        """Test parsing nvidia-smi output with CUDA 13.0."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = NVIDIA_SMI_OUTPUT_CUDA_130
        mock_run.return_value = mock_result

        info = util_module.get_nvidia_driver_info()
        self.assertIsNotNone(info)
        self.assertEqual(info.driver_version, "570.86.10")
        self.assertEqual(info.cuda_version, "13.0")

    @patch.object(util_module.shutil, "which", return_value=None)
    def test_nvidia_smi_not_found(self, mock_which):
        """Test behavior when nvidia-smi is not installed."""
        info = util_module.get_nvidia_driver_info()
        self.assertIsNone(info)

    @patch.object(util_module.subprocess, "run")
    @patch.object(util_module.shutil, "which", return_value="/usr/bin/nvidia-smi")
    def test_nvidia_smi_failure(self, mock_which, mock_run):
        """Test behavior when nvidia-smi returns non-zero."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        info = util_module.get_nvidia_driver_info()
        self.assertIsNone(info)

    @patch.object(util_module.subprocess, "run")
    @patch.object(util_module.shutil, "which", return_value="/usr/bin/nvidia-smi")
    def test_nvidia_smi_timeout(self, mock_which, mock_run):
        """Test behavior when nvidia-smi times out."""
        mock_run.side_effect = util_module.subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

        info = util_module.get_nvidia_driver_info()
        self.assertIsNone(info)

    @patch.object(util_module.subprocess, "run")
    @patch.object(util_module.shutil, "which", return_value="/usr/bin/nvidia-smi")
    def test_caching(self, mock_which, mock_run):
        """Test that results are cached after first call."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = NVIDIA_SMI_OUTPUT_CUDA_126
        mock_run.return_value = mock_result

        info1 = util_module.get_nvidia_driver_info()
        info2 = util_module.get_nvidia_driver_info()
        self.assertEqual(info1, info2)
        # subprocess.run should only be called once due to caching
        mock_run.assert_called_once()


class TestPrintCudaDiagnostics(unittest.TestCase):
    """Test diagnostic output formatting."""

    def setUp(self):
        util_module._NVIDIA_DRIVER_INFO_CACHE = None
        util_module._NVIDIA_DRIVER_INFO_CHECKED = False

    def tearDown(self):
        util_module._NVIDIA_DRIVER_INFO_CACHE = None
        util_module._NVIDIA_DRIVER_INFO_CHECKED = False

    @patch.object(util_module.subprocess, "run")
    @patch.object(util_module.shutil, "which", return_value="/usr/bin/nvidia-smi")
    def test_diagnostics_cuda_13_warning(self, mock_which, mock_run):
        """Test that CUDA 13 driver prints backward-compatibility note."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = NVIDIA_SMI_OUTPUT_CUDA_130
        mock_run.return_value = mock_result

        captured = StringIO()
        with patch("sys.stderr", captured):
            util_module.print_cuda_diagnostics(expected_cuda="12.6")

        output = captured.getvalue()
        self.assertIn("570.86.10", output)
        self.assertIn("13.0", output)
        self.assertIn("backward-compatible", output)
        self.assertIn("LD_LIBRARY_PATH", output)

    @patch.object(util_module.shutil, "which", return_value=None)
    def test_diagnostics_no_nvidia_smi(self, mock_which):
        """Test diagnostics when nvidia-smi is not found."""
        captured = StringIO()
        with patch("sys.stderr", captured):
            util_module.print_cuda_diagnostics()

        output = captured.getvalue()
        self.assertIn("Could not query nvidia-smi", output)


if __name__ == "__main__":
    unittest.main()
