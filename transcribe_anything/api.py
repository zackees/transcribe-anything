"""
    Api for using transcribe_anything from python. Allows bulk processing.
"""

# pylint: disable=too-many-arguments,broad-except,too-many-locals,unsupported-binary-operation,too-many-branches,too-many-statements

import os
import stat
import sys
import time
import subprocess
import shutil
from typing import Optional

from transcribe_anything.audio import fetch_audio
from transcribe_anything.util import (
    get_computing_device,
    sanitize_path,
    chop_double_extension,
)
from transcribe_anything.logger import log_error

PERMS = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWOTH | stat.S_IWUSR | stat.S_IWGRP


def transcribe(
    url_or_file: str,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    language: Optional[str] = None,
    keep_audio: bool = False,
    device: Optional[str] = None,
    other_args: Optional[list[str]] = None,
) -> str:
    """
    Runs the program.
    """
    basename = os.path.basename(url_or_file)
    if not basename or basename == ".":  # if url_or_file is a directory
        # Defense against paths with a trailing /, for example:
        # https://example.com/, which will yield a basename of "".
        basename = os.path.basename(os.path.dirname(url_or_file))
        basename = sanitize_path(basename)
    if output_dir is None:
        if url_or_file.startswith("http"):
            # Try and the title of the video using yt-dlp
            # If that fails, use the basename of the url
            try:
                yt_dlp = subprocess.run(
                    ["yt-dlp", "--get-title", url_or_file],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                output_dir = "text_" + yt_dlp.stdout.strip()
                output_dir = sanitize_path(output_dir[:80].strip())
            except Exception:
                log_error("yt-dlp failed to get title, using basename instead.")
                output_dir = basename
        else:
            output_dir = os.path.splitext(basename)[0]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tmp_mp3 = os.path.join(output_dir, "out.mp3")
    fetch_audio(url_or_file, tmp_mp3)
    assert os.path.exists(tmp_mp3), f"Path {tmp_mp3} doesn't exist."
    device = device or get_computing_device()
    if device == "cuda":
        print("#####################################")
        print("######### GPU ACCELERATED! ##########")
        print("#####################################")
    elif device == "cpu":
        print("WARNING: NOT using GPU acceleration, using 10x slower CPU instead.")
    else:
        raise ValueError(f"Unknown device {device}")
    print(f"Using device {device}")
    model_str = f" --model {model}" if model else ""
    output_dir_str = f' --output_dir "{output_dir}"' if output_dir else ""
    task_str = f" --task {task}" if task else ""
    language_str = f" --language {language}" if language else ""
    cmd_list = [
        "whisper",
        f'"{tmp_mp3}"',
        "--device",
        device,
        model_str,
        output_dir_str,
        task_str,
        language_str,
    ]
    if other_args:
        cmd_list.extend(other_args)
    # Remove the empty strings.
    cmd_list = [x.strip() for x in cmd_list if x.strip()]
    cmd = " ".join(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
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
    files = [os.path.join(output_dir, name) for name in os.listdir(output_dir)]
    for file in files:
        # Change the filename to remove the double extension
        file_name = os.path.basename(file)
        base_path = os.path.dirname(file)
        new_file = os.path.join(base_path, chop_double_extension(file_name))
        if file != new_file:
            if os.path.exists(new_file):
                os.remove(new_file)
            os.rename(file, new_file)
    if not keep_audio:
        os.remove(tmp_mp3)
    return output_dir


if __name__ == "__main__":
    # test case for twitter video
    # transcribe(url_or_file="https://twitter.com/wlctv_ca/status/1598895698870951943")
    transcribe(url_or_file="https://www.youtube.com/watch?v=DWtpNPZ4tb4", output_dir="test")
