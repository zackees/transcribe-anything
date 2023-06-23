"""
    Api for using transcribe_anything from python. Allows bulk processing.
"""

# pylint: disable=too-many-arguments,broad-except,too-many-locals,unsupported-binary-operation,too-many-branches,too-many-statements,disable=notimplemented-raised,unused-variable

# flake8: noqa F401,E303,F821

import atexit
import os
import stat
import sys
import time
import subprocess
from typing import Optional
import tempfile
import shutil
from hashlib import md5  # pylint: disable=unused-import

from appdirs import user_config_dir  # type: ignore
from disklru import DiskLRUCache  # type: ignore  # pylint: disable=unused-import

from static_ffmpeg import add_paths as ffmpeg_add_paths  # type: ignore

from transcribe_anything.audio import fetch_audio
from transcribe_anything.util import (
    get_computing_device,
    sanitize_filename,
    chop_double_extension,
)
from transcribe_anything.logger import log_error



CACHE_FILE = os.path.join(user_config_dir("transcript-anything", "cache", roaming=True))

PERMS = (
    stat.S_IRUSR
    | stat.S_IRGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IWUSR
    | stat.S_IWGRP
)

ffmpeg_add_paths()


def make_temp_wav() -> str:
    """
    Makes a temporary mp3 file and returns the path to it.
    """
    tmp = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        suffix=".wav", delete=False
    )

    tmp.close()

    def cleanup() -> None:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    atexit.register(cleanup)
    return tmp.name


def transcribe(
    url_or_file: str,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    language: Optional[str] = None,
    device: Optional[str] = None,
    embed: bool = False,
    other_args: Optional[list[str]] = None,
) -> str:
    """
    Runs the program.
    """
    if not os.path.isfile(url_or_file) and embed:
        raise NotImplementedError(
            "Embedding is only supported for local files. "
            "Please download the file first."
        )
    # cache = DiskLRUCache(CACHE_FILE, 16)
    basename = os.path.basename(url_or_file)
    if not basename or basename == ".":  # if url_or_file is a directory
        # Defense against paths with a trailing /, for example:
        # https://example.com/, which will yield a basename of "".
        basename = os.path.basename(os.path.dirname(url_or_file))
        basename = sanitize_filename(basename)
    output_dir_was_generated = False
    if output_dir is None:
        output_dir_was_generated = True
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
                output_dir = sanitize_filename(output_dir[:80].strip())
            except Exception:
                log_error("yt-dlp failed to get title, using basename instead.")
                output_dir = "text_" + basename
        else:
            output_dir = "text_" + os.path.splitext(basename)[0]
    if output_dir_was_generated and language is not None:
        output_dir = os.path.join(output_dir, language)
    print(f"making dir {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    tmp_wav = make_temp_wav()
    assert os.path.isdir(
        output_dir
    ), f"Path {output_dir} is not found or not a directory."
    # tmp_mp3 = os.path.join(output_dir, "out.mp3")
    fetch_audio(url_or_file, tmp_wav)
    assert os.path.exists(tmp_wav), f"Path {tmp_wav} doesn't exist."
    #filemd5 = md5(file.encode("utf-8")).hexdigest()
    #key = f"{file}-{filemd5}-{model}"
    #cached_data = cache.get_json(key)
    # print(f"Todo: cached data: {cached_data}")å
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
    task_str = f" --task {task}" if task else ""
    language_str = f" --language {language}" if language else ""
    cmd_list = []
    if sys.platform == "win32":
        # Set the text mode to UTF-8 on Windows.
        cmd_list.extend(["chcp", "65001", "&&"])

    print(f"Running whisper on {tmp_wav} (will install models on first run)")
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd_list.extend(
            [
                "whisper",
                f'"{tmp_wav}"',
                "--device",
                device,
                model_str,
                f'--output_dir "{tmpdir}"',
                task_str,
                language_str,
            ]
        )
        if other_args:
            cmd_list.extend(other_args)
        # Remove the empty strings.
        cmd_list = [x.strip() for x in cmd_list if x.strip()]
        cmd = " ".join(cmd_list)
        sys.stderr.write(f"Running:\n  {cmd}\n")
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd, shell=True, universal_newlines=True
        )
        while True:
            rtn = proc.poll()
            if rtn is None:
                time.sleep(0.25)
                continue
            if rtn != 0:
                msg = f"Failed to execute {cmd}\n "
                raise OSError(msg)
            break
        files = [os.path.join(tmpdir, name) for name in os.listdir(tmpdir)]
        srt_file: Optional[str] = None
        for file in files:
            # Change the filename to remove the double extension
            file_name = os.path.basename(file)
            base_path = os.path.dirname(file)
            new_file = os.path.join(base_path, chop_double_extension(file_name))
            _, ext = os.path.splitext(new_file)
            outfile = os.path.join(output_dir, f"out{ext}")
            if os.path.exists(outfile):
                os.remove(outfile)
            assert os.path.isfile(file), f"Path {file} doesn't exist."
            assert not os.path.exists(outfile), f"Path {outfile} already exists."
            shutil.move(file, outfile)
            if ext == ".srt":
                srt_file = outfile
        assert srt_file is not None, "No srt file found."
        if embed:
            assert os.path.isfile(url_or_file), f"Path {url_or_file} doesn't exist."
            # embed_srt(srt_file, url_or_file)
            #print("Embedding not implemented yet.")
            out_mp4 = os.path.join(output_dir, "out.mp4")
            #ffmpeg -i input.mp4 -c copy -vf "subtitles=subtitle.srt" output.mp4
            embed_ffmpeg_cmd = f'ffmpeg -i "{url_or_file}" -i "{srt_file}" -vf "subtitles={srt_file}" "{out_mp4}"'  # pylint: disable=line-too-long
            print(f"Running:\n  {embed_ffmpeg_cmd}")
            os.system(embed_ffmpeg_cmd)
    return output_dir


if __name__ == "__main__":
    # test case for twitter video
    # transcribe(url_or_file="https://twitter.com/wlctv_ca/status/1598895698870951943")
    try:
        transcribe(
            url_or_file="https://www.youtube.com/watch?v=DWtpNPZ4tb4", output_dir="test"
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        sys.exit(1)
