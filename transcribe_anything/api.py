"""
    Api for using transcribe_anything from python. Allows bulk processing.
"""

import os
import stat
import sys
import tempfile
import time
import subprocess

from transcribe_anything.audio import fetch_audio
from transcribe_anything.logger import (
    log_error,
)

PERMS = (
    stat.S_IRUSR
    | stat.S_IRGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IWUSR
    | stat.S_IWGRP
)


def transcribe(url_or_file: str) -> str:
    """
    Runs the whole program on the input resource and writes
      * out_file if not None
      * or prints to sys.stdout
    """
    tmp_mp3 = "out.mp3"
    tmp_file = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
        suffix=".txt", delete=False
    )
    # Temp file cleaned up at end of try-finally block.
    tmp_file.close()  # TODO: remove, we switched to current directory. # pylint: disable=fixme
    os.chmod(tmp_file.name, PERMS)

    try:
        fetch_audio(url_or_file, tmp_mp3)
        assert os.path.exists(tmp_mp3), f"Path {tmp_mp3} doesn't exist."
        cmd = f"whisper {tmp_mp3}"
        sys.stderr.write(f"Running:\n  {cmd}\n")
        # proc = CapturingProcess(cmd, stdout=StringIO(), stderr=StringIO())
        proc = subprocess.Popen(cmd, shell=True)  # pylint: disable=consider-using-with
        while True:
            rtn = proc.poll()
            if rtn is None:
                time.sleep(0.25)
                continue
            if rtn != 0:
                msg = f"Failed to execute {cmd}\n "
                raise OSError(msg)
            break
        with open(tmp_file.name, encoding="utf-8", mode="r") as filed:
            content = filed.read()
        return content
    finally:
        for name in [tmp_mp3, tmp_file.name]:
            try:
                if os.path.exists(name):
                    os.remove(name)
            except OSError as err:
                log_error(f"Failed to remove {name} because of {err}")


if __name__ == "__main__":
    transcribe(url_or_file="https://www.youtube.com/watch?v=-4EDhdAHrOg")
