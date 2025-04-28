"""
Main entry point.
"""

import logging
import os
import shutil

# for resource loading
from importlib.resources import as_file, files
from pathlib import Path
from tempfile import TemporaryDirectory

from transcribe_anything.api import transcribe

logger = logging.getLogger("transcribe_anything")
logger.setLevel(logging.INFO)

_MODEL = os.environ.get("TRANSCRIBE_ANYTHING_MODEL", "large-v3")
_DEVICE = os.environ.get("TRANSCRIBE_ANYTHING_DEVICE", "insane")
_BATCH_SIZE = os.environ.get("TRANSCRIBE_ANYTHING_BATCH_SIZE", 8)


def main() -> int:
    """Main entry point for the transcribe_anything package."""
    # locate the bundled sample.mp3
    resource = files("transcribe_anything").joinpath("assets/sample.mp3")
    with as_file(resource) as mp3_path:
        with TemporaryDirectory() as tmpdir:
            cwd = tmpdir
            out_path = Path(cwd) / "test.mp3"
            out_txt = Path(cwd) / "test.txt"

            shutil.copy(mp3_path, out_path)
            assert out_path.exists(), f"File {out_path} does not exist"

            print(f"Transcribing {out_path} to {cwd}")
            # result = Api.transcribe_async(str(out_path), str(out_txt)).result()
            other_args = [
                "--batch-size",
                str(_BATCH_SIZE),
            ]
            try:
                transcribe(
                    str(out_path),
                    str(out_txt),
                    language="en",
                    task="transcribe",
                    device=_DEVICE,
                    model=_MODEL,
                    other_args=other_args,
                )
            except Exception as e:
                print(f"Transcription failed with error: {e}")
                return 1
            print("Transcription completed.")
            return 0


if __name__ == "__main__":

    main()
