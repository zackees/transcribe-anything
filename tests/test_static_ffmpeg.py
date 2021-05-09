import unittest
import subprocess

class TranscribeAnythingTester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = []

    def test_fetch_url(self) -> None:
        subprocess.check_output(['transcribe_anything', '-h'])


if __name__ == '__main__':
    unittest.main()
