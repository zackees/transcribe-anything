import unittest
import subprocess

class TranscribeAnythingTester(unittest.TestCase):
    def test_fetch_url(self) -> None:
        """Check that the command is installed by the setup process."""
        subprocess.check_output(['transcribe_anything', '-h'])


if __name__ == '__main__':
    unittest.main()
