"""
    Fetches audio and handles transcoding it for it.
"""


import sys
import tempfile
import subprocess
import os
import shutil
import static_ffmpeg  # type: ignore

from transcribe_anything.util import PROCESS_TIMEOUT
from transcribe_anything.ytldp_download import ytdlp_download


def _convert_to_wav(
    inpath: str, outpath: str, speech_normalization: bool = False
) -> None:
    """Converts a file to wav."""
    cmd_audio_filter = ""
    if speech_normalization:
        cmd_audio_filter = "-filter:a speechnorm=e=12.5:r=0.00001:l=1"
    tmpwav = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        suffix=".wav", delete=False
    )
    tmpwav.close()
    tmpwavepath = tmpwav.name
    audio_encoder = "-acodec pcm_s16le -ar 44100 -ac 1"
    cmd = f'ffmpeg -y -i "{inpath}" {cmd_audio_filter} {audio_encoder} "{tmpwavepath}"'
    print(f"Running:\n  {cmd}")
    try:
        subprocess.run(
            cmd, shell=True, check=True, capture_output=True, timeout=PROCESS_TIMEOUT
        )
    except subprocess.CalledProcessError as exc:
        print(f"Failed to run {cmd} with error {exc}")
        print(f"stdout: {exc.stdout}")
        print(f"stderr: {exc.stderr}")
        raise
    os.remove(outpath)
    os.rename(tmpwavepath, outpath)
    assert os.path.exists(outpath), f"The expected file {outpath} doesn't exist"


def fetch_audio(url_or_file: str, out_wav: str) -> None:
    """Fetches from the internet or from a local file and outputs a wav file."""
    assert out_wav.endswith(".wav")
    static_ffmpeg.add_paths()  # pylint: disable=no-member
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
            cmd = f'ffmpeg -y -i "{abspath}" -acodec pcm_s16le -ar 44100 -ac 1 out.wav'
            sys.stderr.write(f"Running:\n  {cmd}\n")
            subprocess.run(
                cmd,
                cwd=tmpdir,
                shell=True,
                check=True,
                capture_output=True,
                timeout=PROCESS_TIMEOUT,
            )
            shutil.copyfile(os.path.join(tmpdir, "out.wav"), out_wav_abs)
        assert os.path.exists(out_wav), f"The expected file {out_wav} doesn't exist"


def unit_test() -> None:
    """Runs the program."""
    url = "https://www.youtube.com/watch?v=8Wg8f2g_GQY"
    fetch_audio(url, "out.wav")


if __name__ == "__main__":
    unit_test()
