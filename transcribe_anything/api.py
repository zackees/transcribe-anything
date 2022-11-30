"""
    Api for using transcribe_anything from python. Allows bulk processing.
"""

import os
import stat
import sys
import time
import subprocess

from transcribe_anything.audio import fetch_audio

PERMS = (
    stat.S_IRUSR
    | stat.S_IRGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IWUSR
    | stat.S_IWGRP
)


def transcribe(url_or_file: str) -> None:
    """
    Runs the whole program on the input resource and writes
      * out_file if not None
      * or prints to sys.stdout
    """

    basename = os.path.basename(url_or_file)
    dirname = os.path.splitext(basename)[0].split("?")[0]
    os.makedirs(dirname, exist_ok=True)
    tmp_mp3 = os.path.join(dirname, "out.mp3")
    try:
        fetch_audio(url_or_file, tmp_mp3)
        assert os.path.exists(tmp_mp3), f"Path {tmp_mp3} doesn't exist."
        cmd = f"whisper {tmp_mp3} --output_dir {dirname}"
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
        return
    finally:
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)


if __name__ == "__main__":
    transcribe(url_or_file="https://www.youtube.com/watch?v=-4EDhdAHrOg")
