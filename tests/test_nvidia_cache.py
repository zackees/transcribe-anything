"""
Test NVIDIA detection caching functionality.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src to path to import without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the util module directly to avoid dependency issues
import importlib.util

spec = importlib.util.spec_from_file_location("util_module", Path(__file__).parent.parent / "src" / "transcribe_anything" / "util.py")
assert spec is not None, "Could not find spec for util.py"
assert spec.loader is not None, f"Could not load spec for util.py: {spec.loader}"
util_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util_module)


class TestNvidiaCache(unittest.TestCase):
    """Test NVIDIA detection caching."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing cache before each test
        util_module.clear_nvidia_cache()

    def tearDown(self):
        """Clean up after each test."""
        # Clear cache after each test
        util_module.clear_nvidia_cache()

    def test_nvidia_detection_caching(self):
        """Test that NVIDIA detection results are cached."""
        # Mock both shutil.which and fingerprint to ensure consistent caching
        with patch.object(util_module.shutil, "which") as mock_which:
            with patch.object(util_module, "_get_system_fingerprint") as mock_fingerprint:
                # Use a consistent fingerprint
                mock_fingerprint.return_value = "test-system-fingerprint"

                # First call - should detect and cache
                mock_which.return_value = "/usr/bin/nvidia-smi"
                result1 = util_module.has_nvidia_smi()
                self.assertTrue(result1)

                # Second call - should use cache (mock won't be called again for detection)
                mock_which.return_value = None  # Change mock return value
                result2 = util_module.has_nvidia_smi()
                self.assertTrue(result2)  # Should still be True from cache

                # Verify cache file exists
                self.assertTrue(util_module._NVIDIA_CACHE_FILE.exists())

    def test_cache_clearing(self):
        """Test that cache can be cleared."""
        # Create a cached result
        with patch.object(util_module.shutil, "which") as mock_which:
            mock_which.return_value = "/usr/bin/nvidia-smi"
            result1 = util_module.has_nvidia_smi()
            self.assertTrue(result1)
            self.assertTrue(util_module._NVIDIA_CACHE_FILE.exists())

            # Clear cache
            util_module.clear_nvidia_cache()
            self.assertFalse(util_module._NVIDIA_CACHE_FILE.exists())

            # Next call should re-detect
            mock_which.return_value = None
            result2 = util_module.has_nvidia_smi()
            self.assertFalse(result2)

    def test_different_system_fingerprints(self):
        """Test that different system fingerprints get different cache entries."""
        with patch.object(util_module.shutil, "which") as mock_which:
            with patch.object(util_module, "_get_system_fingerprint") as mock_fingerprint:
                # First system fingerprint
                mock_fingerprint.return_value = "system1"
                mock_which.return_value = "/usr/bin/nvidia-smi"
                result1 = util_module.has_nvidia_smi()
                self.assertTrue(result1)

                # Different system fingerprint
                mock_fingerprint.return_value = "system2"
                mock_which.return_value = None
                result2 = util_module.has_nvidia_smi()
                self.assertFalse(result2)

                # Back to first system - should use cached result
                mock_fingerprint.return_value = "system1"
                mock_which.return_value = None  # This shouldn't matter due to cache
                result3 = util_module.has_nvidia_smi()
                self.assertTrue(result3)  # Should be True from cache


if __name__ == "__main__":
    unittest.main()
