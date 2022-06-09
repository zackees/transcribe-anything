"""
    Fetches audio and handles transcoding it for usage in Mozilla's Deepspeech.
"""


import sys
import subprocess
import os

from transcribe_anything.logger import log_debug, log_error

_PROCESS_TIMEOUT = 4 * 60 * 60

def _convert_to_deepspeech_wav(in_media: str, out_wav: str) -> None:
    """
    Convert to wave format compatible with pydeepspeech which is:
      * mono audio channel.
      * sample rate of 16000
    """
    cmd = f'static_ffmpeg -y -i "{in_media}" -ac 1 -ar 16000 "{out_wav}"'
    log_debug(f'Running cmd: "{cmd}"')
    try:
        subprocess.run(
            cmd, shell=True, check=True, capture_output=True, timeout=_PROCESS_TIMEOUT
        )
    except subprocess.TimeoutExpired as te:
        log_error(
            f"{__file__}: Timeout expired for {cmd}\n Stdout: {te.stdout}\n Stderr: {te.stderr}"
        )


def fetch_mono_16000_audio(url_or_file: str, out_wav: str) -> None:
    """Fetches from the internet or from a local file and outputs a wav file."""
    if url_or_file[:4] == "http":
        # Download via yt-dlp
        tmp_m4a = f"{out_wav}.m4a"
        try:
            cmd = f'yt-dlp --no-check-certificate -f "bestaudio[ext=m4a]" {url_or_file} -o {tmp_m4a}'
            sys.stderr.write(f"Running:\n  {cmd}\n")
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            log_debug(
                "Could not just download audio stream, falling back to full video download"
            )
            cmd = f"yt-dlp --no-check-certificate {url_or_file} -o {tmp_m4a}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        sys.stderr.write("Downloading complete.\n")
        assert os.path.exists(tmp_m4a), f"The expected file {tmp_m4a} doesn't exist"
        _convert_to_deepspeech_wav(tmp_m4a, out_wav)
        os.remove(tmp_m4a)
    else:
        assert os.path.isfile(url_or_file)
        _convert_to_deepspeech_wav(url_or_file, out_wav)
