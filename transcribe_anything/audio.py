"""
    Fetches audio and handles transcoding it for usage in Mozilla's Deepspeech.
"""


import sys
import subprocess
import os
import static_ffmpeg  # type: ignore

_PROCESS_TIMEOUT = 4 * 60 * 60


def fetch_audio(url_or_file: str, out_mp3: str) -> None:
    """Fetches from the internet or from a local file and outputs a wav file."""
    assert out_mp3.endswith(".mp3")
    static_ffmpeg.add_paths()
    if url_or_file.startswith("http") or url_or_file.startswith("ftp"):
        # cmd = f'yt-dlp --no-check-certificate -f "bestaudio[ext=m4a]" {url_or_file} -o {tmp_m4a}'
        cmd = (
            f'yt-dlp -f bestaudio "{url_or_file}" '
            + '--exec "ffmpeg -y -hide_banner -v quiet -stats -i '
            + "{}"
            + f" -codec:a libmp3lame -qscale:a 0 {out_mp3}"
        )
        sys.stderr.write(f"Running:\n  {cmd}\n")
        subprocess.run(cmd, shell=True, check=True, timeout=_PROCESS_TIMEOUT)
        sys.stderr.write("Downloading complete.\n")
        assert os.path.exists(out_mp3), f"The expected file {out_mp3} doesn't exist"
    else:
        assert os.path.isfile(url_or_file)
        # static_ffmpeg -i audio.wav -acodec libmp3lame audio.mp3
        cmd = f'static_ffmpeg -i "{url_or_file}" -acodec libmp3lame "{out_mp3}"'
        sys.stderr.write(f"Running:\n  {cmd}\n")
        subprocess.run(
            cmd, shell=True, check=True, capture_output=True, timeout=_PROCESS_TIMEOUT
        )
        assert os.path.exists(out_mp3), f"The expected file {out_mp3} doesn't exist"
