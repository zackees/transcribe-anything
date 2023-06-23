"""
Download utility for yt-dlp.
"""


import os
import subprocess

from transcribe_anything.util import PROCESS_TIMEOUT


def ytdlp_download(url: str, outdir: str) -> str:
    """Downloads a file using ytdlp."""
    os.makedirs(outdir, exist_ok=True)
    # remove all files in the directory
    for file in os.listdir(outdir):
        os.remove(os.path.join(outdir, file))
    cmd = f'yt-dlp --no-check-certificate {url} -o "out.%(ext)s"'
    print(f"Running:\n  {cmd}")
    subprocess.run(
        cmd,
        shell=True,
        cwd=outdir,
        check=True,
        timeout=PROCESS_TIMEOUT,
        universal_newlines=True,
    )
    new_files = os.listdir(outdir)
    assert len(new_files) == 1, f"Expected 1 file, got {new_files}"
    downloaded_file = os.path.join(outdir, new_files[0])
    assert os.path.exists(
        downloaded_file
    ), f"The expected file {downloaded_file} doesn't exist"
    return downloaded_file
