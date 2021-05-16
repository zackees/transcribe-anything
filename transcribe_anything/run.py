"""
    Entry point for running the transcribe-anything prgram.
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import time

_VERBOSE = False


def vprint(msg: str) -> None:
    """Prints but only if _VERBOSE has been set to true"""
    if _VERBOSE:
        sys.stderr.write(f"{msg}\n")


def convert_to_deepspeech_wav(in_media: str, out_wav: str) -> None:
    """
    Convert to wave format compatible with pydeepspeech which is:
      * mono audio channel.
      * sample rate of 16000
    """
    cmd = f"static_ffmpeg -y -i {in_media} -ac 1 -ar 16000 {out_wav}"
    vprint(f'Running cmd: "{cmd}"')
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def fetch_mono_16000_audio(url_or_file: str, out_wav: str) -> None:
    """Fetches from the internet or from a local file and outputs a wav file."""
    if url_or_file[:4] == "http":
        # Download via youtube-dl
        tmp_m4a = f"{out_wav}.m4a"
        try:
            cmd = f'youtube-dl -f "bestaudio[ext=m4a]" {url_or_file} -o {tmp_m4a}'
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            vprint(
                "Could not just download audio stream, falling back to full video download"
            )
            cmd = f"youtube-dl {url_or_file} -o {tmp_m4a}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        vprint("Downloading complete.")
        convert_to_deepspeech_wav(tmp_m4a, out_wav)
        os.remove(tmp_m4a)
    else:
        assert os.path.isfile(url_or_file)
        convert_to_deepspeech_wav(url_or_file, out_wav)


def main() -> None:
    """Main entry point for the command line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url_or_file",
        help="Provide file path or url (includes youtube/facebook/twitter/etc)",
    )
    parser.add_argument(
        "--out",
        help="Output text file name",
        default=None,
    )
    args = parser.parse_args()
    tmp_wav = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
        suffix=".wav", delete=False
    )
    tmp_wav.close()
    tmp_file = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
        suffix=".txt", delete=False
    )
    tmp_file.close()

    try:
        fetch_mono_16000_audio(args.url_or_file, tmp_wav.name)
        cmd = f"pydeepspeech --wav_file {tmp_wav.name} --out_file {tmp_file.name}"
        proc = subprocess.Popen(  # pylint: disable=R1732
            cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        while True:
            rtn = proc.poll()
            if rtn is None:
                time.sleep(0.1)
                continue
            assert rtn == 0, f"Failed to execute {cmd}"
            break
        if args.out is not None:
            shutil.copy(tmp_file.name, args.out)
        else:
            with open(tmp_file.name) as fd:
                content = fd.read()
            sys.stdout.write(f"{content}\n")
    finally:
        os.remove(tmp_wav.name)
        os.remove(tmp_file.name)


if __name__ == "__main__":
    main()
    sys.exit(0)
