"""
Api for using transcribe_anything from python. Allows bulk processing.
"""

# pylint: disable=too-many-arguments,broad-except,too-many-locals,unsupported-binary-operation,too-many-branches,too-many-statements,disable=notimplemented-raised,unused-variable,line-too-long

# flake8: noqa F401,E303,F821
# ruff: noqa F401

import atexit
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

import static_ffmpeg  # type: ignore
from appdirs import user_config_dir  # type: ignore

from transcribe_anything.audio import fetch_audio
from transcribe_anything.insanely_fast_whisper import run_insanely_fast_whisper
from transcribe_anything.logger import log_error
from transcribe_anything.util import chop_double_extension, sanitize_filename
from transcribe_anything.whisper import get_computing_device, run_whisper
from transcribe_anything.whisper_mac import run_whisper_mac_english

DISABLED_WARNINGS = [
    ".*set_audio_backend has been deprecated.*",
    ".*torchaudio._backend.set_audio_backend has been deprecated.*",
]

IS_GITHUB = os.environ.get("GITHUB_ACTIONS", "false") == "true"

for warning in DISABLED_WARNINGS:
    warnings.filterwarnings("ignore", category=UserWarning, message=warning)

os.environ["PYTHONIOENCODING"] = "utf-8"

CACHE_FILE = os.path.join(user_config_dir("transcript-anything", "cache", roaming=True))

PERMS = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWOTH | stat.S_IWUSR | stat.S_IWGRP


class Device(Enum):
    """Device enum."""

    CPU = "cpu"
    CUDA = "cuda"
    INSANE = "insane"
    MPS = "mps"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def from_str(device: str) -> "Device":
        """Returns the device from a string."""
        if device == "cpu":
            return Device.CPU
        if device == "cuda":
            return Device.CUDA
        if device == "insane":
            return Device.INSANE
        if device == "mps":
            if sys.platform != "darwin":
                raise ValueError("MPS is only supported on macOS.")
            return Device.MPS
        raise ValueError(f"Unknown device {device}")


def make_temp_wav() -> str:
    """
    Makes a temporary mp3 file and returns the path to it.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # pylint: disable=consider-using-with

    tmp.close()

    def cleanup() -> None:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    atexit.register(cleanup)
    return tmp.name


def fix_subtitles_path(_path: str) -> str:
    """Fixes windows subtitles path, which is weird."""
    if sys.platform != "win32":
        return _path
    # On Windows, ffmpeg 5 requires the path to be escaped.
    # For example, "C:\Users\user\file.srt" should be "C\\:/\Users/\user/\file.srt".
    # See https://stackoverflow.com/questions/60440793/how-can-i-use-windows-absolute-paths-with-the-movie-filter-on-ffmpeg
    # Replace backslashes with '/\' and add an extra backslash at the start of the drive letter
    p = Path(_path).absolute()
    # Handle drive letter specially
    drive = p.drive
    parts = p.parts
    out: str = drive.replace(":", "") + "\\\\:"
    for part in parts[1:]:
        out += "/\\" + part
    return out


def get_video_name_from_url(url: str) -> str:
    """
    Returns the video name.
    """
    assert url.startswith("http"), f"Invalid url {url}"

    # Try and the title of the video using yt-dlp
    # If that fails, use the basename of the url
    cmd_list: list[str] = []
    cmd_list.extend(["yt-dlp", "--get-title", f"{url}"])

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd_str = subprocess.list2cmdline(cmd_list)
            print(f"Running:\n  {cmd_str}")
            env = os.environ.copy()
            # env["set PYTHONIOENCODING=utf-8"]
            env["PYTHONIOENCODING"] = "utf-8"
            cp = subprocess.run(
                cmd_str,
                check=True,
                capture_output=True,
                universal_newlines=True,
                cwd=temp_dir,
                env=env,
                shell=True,
            )
            stdout = cp.stdout
            lines = stdout.split("\n")
            lines = [line.strip() for line in lines]
            for line in lines:
                if "OSError" in line:
                    continue
                line = line[:80]
                return sanitize_filename(line)
            log_error("yt-dlp failed to get title, using basename instead.")
            return os.path.basename(url)
    except subprocess.CalledProcessError as exc:
        log_error(f"yt-dlp failed with {exc}, using basename instead\n{exc.stdout}\n{exc.stderr}")
        return os.path.basename(url)
    except Exception as exc:
        log_error(f"yt-dlp failed with {exc}, using basename instead.")
        return os.path.basename(url)


def check_python_in_range() -> None:
    """
    Returns whether the python version is in range.
    """
    # valid ranges are (3, 10, 0) to (3, 11, X)
    in_range = sys.version_info >= (3, 10, 0) and sys.version_info < (3, 12, 0)
    if not in_range:
        msg = f"# WARNING: Python version {sys.version_info} is not in the range (3, 10, 0) to (3, 11, X)."
        header = "\n\n#" + "#" * len(msg)
        footer = "#" + "#" * len(msg) + "\n\n"
        msg += "\n# transcribe-anything may not work correctly. Please use a supported version of Python."
        warnings.warn(f"{header}\n{msg}\n{footer}\n")


def transcribe(
    url_or_file: str,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    language: Optional[str] = None,
    device: Optional[str] = None,
    embed: bool = False,
    hugging_face_token: Optional[str] = None,
    other_args: Optional[list[str]] = None,
) -> str:
    """
    Runs the program.
    """
    # add the paths for any dependent tools that may rely on ffmpeg
    static_ffmpeg.add_paths()
    check_python_in_range()
    if not os.path.isfile(url_or_file) and embed:
        raise NotImplementedError("Embedding is only supported for local files. " + "Please download the file first.")
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
            outname = get_video_name_from_url(url_or_file)
            output_dir = "text_" + sanitize_filename(outname)
        else:
            output_dir = "text_" + os.path.splitext(basename)[0]
    if output_dir_was_generated and language is not None:
        output_dir = os.path.join(output_dir, language)
    print(f"making dir {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    tmp_wav = make_temp_wav()
    assert os.path.isdir(output_dir), f"Path {output_dir} is not found or not a directory."
    # tmp_mp3 = os.path.join(output_dir, "out.mp3")
    fetch_audio(url_or_file, tmp_wav)
    assert os.path.exists(tmp_wav), f"Path {tmp_wav} doesn't exist."
    # filemd5 = md5(file.encode("utf-8")).hexdigest()
    # key = f"{file}-{filemd5}-{model}"
    # cached_data = cache.get_json(key)
    # print(f"Todo: cached data: {cached_data}")
    device = device or get_computing_device()
    device_enum = Device.from_str(device)
    if device_enum == Device.CUDA:
        print("#####################################")
        print("######### GPU ACCELERATED! ##########")
        print("#####################################")
    elif device_enum == Device.INSANE:
        print("#####################################")
        print("####### INSANE GPU MODE! ############")
        print("#####################################")
    elif device_enum == Device.CPU:
        print("WARNING: NOT using GPU acceleration, using 10x slower CPU instead.")
    elif device_enum == Device.MPS:
        print("#####################################")
        print("####### MAC MPS GPU MODE! ###########")
        print("#####################################")
    else:
        raise ValueError(f"Unknown device {device}")
    print(f"Using device {device}")
    model_str = f"{model}" if model else ""
    task_str = f"{task}" if task else "transcribe"
    language_str = f"{language}" if language else ""

    print(f"Running whisper on {tmp_wav} (will install models on first run)")
    with tempfile.TemporaryDirectory() as tmpdir:
        if device_enum == Device.INSANE:
            run_insanely_fast_whisper(
                input_wav=Path(tmp_wav),
                model=model_str,
                output_dir=Path(tmpdir),
                task=task_str,
                language=language_str,
                hugging_face_token=hugging_face_token,
                other_args=other_args,
            )
        elif device_enum == Device.MPS and (language_str == "" or language_str == "en" or language_str == "English") and (task_str == "transcribe"):
            run_whisper_mac_english(input_wav=Path(tmp_wav), model=model_str, output_dir=Path(tmpdir))
        else:
            run_whisper(
                input_wav=Path(tmp_wav),
                device=str(device),
                model=model_str,
                output_dir=Path(tmpdir),
                task=task_str,
                language=language_str,
                other_args=other_args,
            )
        files = [os.path.join(tmpdir, name) for name in os.listdir(tmpdir)]
        srt_file: Optional[str] = None
        for file in files:
            # Change the filename to remove the double extension
            file_name = os.path.basename(file)
            base_path = os.path.dirname(file)
            new_file = os.path.join(base_path, chop_double_extension(file_name))
            _, ext = os.path.splitext(new_file)
            if "speaker.json" in new_file:  # pass through speaker.json
                outfile = os.path.join(output_dir, "speaker.json")
            else:
                outfile = os.path.join(output_dir, f"out{ext}")
            if os.path.exists(outfile):
                os.remove(outfile)
            assert os.path.isfile(file), f"Path {file} doesn't exist."
            assert not os.path.exists(outfile), f"Path {outfile} already exists."
            shutil.move(file, outfile)
            if ext == ".srt":
                srt_file = outfile
        output_dir = os.path.abspath(output_dir)
        assert srt_file is not None, "No srt file found."
        srt_file = os.path.abspath(srt_file)
        if embed:
            assert os.path.isfile(url_or_file), f"Path {url_or_file} doesn't exist."
            out_mp4 = os.path.join(output_dir, "out.mp4")
            static_ffmpeg_path = shutil.which("static_ffmpeg")
            if static_ffmpeg_path is None:
                raise FileNotFoundError("static_ffmpeg not found")
            embed_ffmpeg_cmd_list = [
                "static_ffmpeg",
                "-y",
                "-i",
                url_or_file,
                "-i",
                srt_file,
                "-vf",
                f"subtitles={fix_subtitles_path(srt_file)}",
                out_mp4,
            ]
            embed_ffmpeg_cmd = subprocess.list2cmdline(embed_ffmpeg_cmd_list)
            print(f"Running:\n  {embed_ffmpeg_cmd}")
            try:
                _ = subprocess.run(
                    embed_ffmpeg_cmd,
                    universal_newlines=True,
                    check=True,
                    capture_output=True,
                    shell=True,
                )
            except subprocess.CalledProcessError as exc:
                stdout = exc.stdout
                stderr = exc.stderr
                warnings.warn(f"ffmpeg failed with return code {exc.returncode}\n{stdout}\n{stderr}")
                raise
    print(f"Done! Files were saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    # test case for twitter video
    # transcribe(url_or_file="https://twitter.com/wlctv_ca/status/1598895698870951943")
    try:
        # transcribe(url_or_file="https://www.youtube.com/live/gBHFFM7-aCk?feature=share", output_dir="test")
        transcribe(
            url_or_file=r"E:\james_o_keefe_struggle_sessions.mp4",
            output_dir=r"E:\test2",
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        sys.exit(1)
