"""
    Entry point for running the transcribe-anything prgram.
"""

import argparse
import os
import subprocess
import tempfile


def convert_to_deepspeech_wav(in_media, out_wav):
    """
    Convert to wave format compatible with pydeepspeech which is:
      * mono audio channel.
      * sample rate of 16000
    """
    cmd = f"static_ffmpeg -y -i {in_media} -ac 1 -ar 16000 {out_wav}"
    print(f'Running cmd: "{cmd}"')
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def fetch_mono_16000_audio(url_or_file: str, out_wav: str) -> None:
    """Fetches from the internet or from a local file and outputs a wav file."""
    if url_or_file[:4] == "http":
        # Download via youtube-dl
        tmp_m4a = f"{out_wav}.m4a"
        cmd = f'youtube-dl -f "bestaudio[ext=m4a]" {url_or_file} -o {tmp_m4a}'
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print("Downloading complete.")
        convert_to_deepspeech_wav(tmp_m4a, out_wav)
        os.remove(tmp_m4a)
    else:
        assert os.path.isfile(url_or_file)
        convert_to_deepspeech_wav(url_or_file, out_wav)


def main():
    """Main entry point for the command line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url_or_file",
        help="Provide file path or url (includes youtube/facebook/twitter/etc)",
    )
    parser.add_argument(
        "out",
        help="First argument is the mp4 or url to youtube/mp4",
        nargs="?",
        default="subtitles.txt",
    )
    args = parser.parse_args()
    tmp_wav = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
        suffix=".wav", delete=False
    )
    tmp_wav.close()
    try:
        fetch_mono_16000_audio(args.url_or_file, tmp_wav.name)
        print(f"Wrote out {tmp_wav.name}")
        os.system(f"pydeepspeech --wav_file {tmp_wav.name} --out_file {args.out}")
    finally:
        os.remove(tmp_wav.name)
