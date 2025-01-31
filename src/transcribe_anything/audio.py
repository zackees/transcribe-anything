"""
Fetches audio and handles transcoding it for it.
"""

import os
import shutil
import subprocess
import sys
import tempfile

from transcribe_anything.util import PROCESS_TIMEOUT
from transcribe_anything.ytldp_download import ytdlp_download


def _convert_to_wav(inpath: str, outpath: str, speech_normalization: bool = False) -> None:
    """Converts a file to wav."""
    # static_ffmpeg -y -i C:\Users\niteris\AppData\Local\Temp\tmp3xhzm1sn\out.webm -filter:a "speechnorm=e=12.5:r=0.00001:l=1" -acodec pcm_s16le -ar 44100 -ac 1 C:\Users\niteris\AppData\Local\Temp\tmpu32zsjov.wav

    tmpwav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # pylint: disable=consider-using-with
    tmpwav.close()
    tmpwavepath = tmpwav.name

    cmd_list = ["static_ffmpeg", "-y", "-i", str(inpath)]
    if speech_normalization:
        cmd_list += [
            "-filter:a",
            "speechnorm=e=12.5:r=0.00001:l=1",
        ]
    cmd_list += ["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", str(tmpwavepath)]
    cmd = subprocess.list2cmdline(cmd_list)
    print(f"Running:\n  {cmd}")
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            timeout=PROCESS_TIMEOUT,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Failed to run {cmd} with error {exc}")
        print(f"stdout: {exc.stdout}")
        print(f"stderr: {exc.stderr}")
        raise
    os.remove(outpath)
    # os.rename(tmpwavepath, outpath)
    # overwrite file at outpath
    shutil.copyfile(tmpwavepath, outpath)
    assert os.path.exists(outpath), f"The expected file {outpath} doesn't exist"


def fetch_audio(url_or_file: str, out_wav: str) -> None:
    """Fetches from the internet or from a local file and outputs a wav file."""
    assert out_wav.endswith(".wav")
    if url_or_file.startswith("http") or url_or_file.startswith("ftp"):
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Using temporary directory {tmpdir}")
            downloaded_file = ytdlp_download(url_or_file, os.path.abspath(tmpdir))
            print("Downloaded file: ", downloaded_file)
            _convert_to_wav(downloaded_file, out_wav, speech_normalization=True)
        sys.stderr.write("Downloading complete.\n")
        assert os.path.exists(out_wav), f"The expected file {out_wav} doesn't exist"
    else:
        assert os.path.isfile(url_or_file)
        abspath = os.path.abspath(url_or_file)
        out_wav_abs = os.path.abspath(out_wav)
        with tempfile.TemporaryDirectory() as tmpdir:
            static_ffmpeg_path = shutil.which("static_ffmpeg")
            if static_ffmpeg_path is None:
                raise FileNotFoundError("No path for static_ffmpeg")
            cmd_list = [
                static_ffmpeg_path,
                "-y",
                "-i",
                str(abspath),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "1",
                "out.wav",
            ]
            cmd_str = subprocess.list2cmdline(cmd_list)
            sys.stderr.write(f"Running:\n  {cmd_str}\n")
            try:
                subprocess.run(
                    cmd_list,
                    cwd=tmpdir,
                    shell=False,
                    check=False,
                    capture_output=True,
                    timeout=PROCESS_TIMEOUT,
                )
                shutil.copyfile(os.path.join(tmpdir, "out.wav"), out_wav_abs)
            except subprocess.CalledProcessError as exc:
                print(f"Failed to run {cmd_str} with error {exc}")
                print(f"stdout: {exc.stdout.decode()}")
                print(f"stderr: {exc.stderr.decode()}")
                raise
        assert os.path.exists(out_wav), f"The expected file {out_wav} doesn't exist"


def unit_test() -> None:
    """Runs the program."""
    url = "https://www.youtube.com/watch?v=8Wg8f2g_GQY"
    fetch_audio(url, "out.wav")


if __name__ == "__main__":
    unit_test()
