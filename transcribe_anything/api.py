"""
    Api for using transcribe_anything from python. Allows bulk processing.
"""

import os
import stat
import sys
import time
import subprocess

from transcribe_anything.audio import fetch_audio
from transcribe_anything.util import get_computing_device

PERMS = (
    stat.S_IRUSR
    | stat.S_IRGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IWUSR
    | stat.S_IWGRP
)


def sanitize_folder_name(folder_name: str) -> str:
    """Sanitize a folder name."""
    folder_name = folder_name.split("?")[0]
    return "".join([c for c in folder_name if c.isalnum() or c in ["-", "_"]])


def transcribe(url_or_file: str, output_dirname: str | None = None, model: str = "small") -> None:
    """
    Runs the program.
    """
    basename = os.path.basename(url_or_file)
    if output_dirname is None:
        output_dirname = sanitize_folder_name(basename)
        # sanitize dirname
        output_dirname = "".join(
            [c for c in output_dirname if c.isalnum() or c in ["-", "_"]]
        )
    os.makedirs(output_dirname, exist_ok=True)
    tmp_mp3 = os.path.join(output_dirname, "out.mp3")
    try:
        fetch_audio(url_or_file, tmp_mp3)
        assert os.path.exists(tmp_mp3), f"Path {tmp_mp3} doesn't exist."
        device = get_computing_device()
        if device == "cuda":
            print("Using GPU")
        elif device == "cpu":
            print("WARNING: Using CPU, this will be at least 10x slower.")
        else:
            raise ValueError(f"Unknown device {device}")
        print(f"Using device {device}")
        cmd = f"whisper {tmp_mp3} --device {device} --model {model} --output_dir {output_dirname}"
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
