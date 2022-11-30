"""
Tests transcribe_anything
"""

# pylint: disable=no-self-use

import os
import subprocess
import tempfile
import unittest


class TranscribeAnythingTester(unittest.TestCase):
    """Tester for transcribe anything."""

    def test_fetch_command_help(self) -> None:
        """Check that the command is installed by the setup process."""
        subprocess.check_output(['transcribe_anything', '-h'])

    def test_fetch_command_installed(self) -> None:
        """Check that the command works on a live short video."""

        tmp_file = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)  # pylint: disable=consider-using-with
        tmp_name = tmp_file.name
        try:
            tmp_file.close()
            subprocess.check_output([
                'transcribe_anything',
                'https://www.youtube.com/watch?v=8Wg8f2g_GQY',
                '--out',
                tmp_name
            ])
        finally:
            os.remove(tmp_name)


if __name__ == '__main__':
    unittest.main()
