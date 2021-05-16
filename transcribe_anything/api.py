"""
    Api for using transcribe_anything from python. Allows bulk processing.
"""

import multiprocessing
import os
from io import StringIO
import tempfile
import time
import sys
from typing import Any, Callable, List
from capturing_process import CapturingProcess  # type: ignore
from transcribe_anything.logger import (
    log_debug,
    log_error,
    log_info,
    set_logging_level,
    INFO,
)

from transcribe_anything.audio import fetch_mono_16000_audio


def transcribe(url_or_file: str) -> str:
    """
    Runs the whole program on the input resource and writes
      * out_file if not None
      * or prints to sys.stdout
    """
    tmp_wav = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
        suffix=".wav", delete=False
    )
    tmp_wav.close()
    tmp_file = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
        suffix=".txt", delete=False
    )
    tmp_file.close()

    try:
        fetch_mono_16000_audio(url_or_file, tmp_wav.name)
        cmd = f"pydeepspeech --wav_file {tmp_wav.name} --out_file {tmp_file.name}"
        proc = CapturingProcess(cmd, stdout=StringIO(), stderr=StringIO())
        while True:
            rtn = proc.poll()
            if rtn is None:
                time.sleep(0.25)
                continue
            assert rtn == 0, f"Failed to execute {cmd}"
            break
        with open(tmp_file.name) as fd:
            content = fd.read()
        return content
    finally:
        for name in [tmp_wav.name, tmp_file.name]:
            try:
                if os.path.exists(name):
                    os.remove(name)
            except OSError as err:
                log_error(f"Failed to remove {name} because of {err}")


def bulk_transcribe(
    urls: List[str],
    onresolve: Callable[[str, str], None],
    onfail: Callable[[str], None],
    num_cpus: int = -1,
    verbose: bool = False,
) -> None:
    """On each completed url onresolve(url, subtitle) will be called."""
    if verbose:
        set_logging_level(INFO)
    url_work_q = list(urls)
    if num_cpus == -1:
        num_cpus = max(1, int(multiprocessing.cpu_count() / 2))
    log_info(f"Num cpus={num_cpus}")
    procs: List[Any] = [None for _ in range(num_cpus)]
    while True:
        n_active_procs = len([p for p in procs if p])
        log_info(f"n_active_procs={n_active_procs}")
        log_info(f"len(url_work_q)={len(url_work_q)}")
        if n_active_procs == 0 and len(url_work_q) == 0:
            # No running procs and no commands left to run so bail.
            return
        # Scan through and poll active processes and look to remove processes
        # that are finished and replace them with None.
        for i, p in enumerate(procs):
            if p is None:
                continue
            rtn = p.poll()
            if rtn is None:
                # Still running
                continue
            log_debug(f"{p.url} finished running.")
            if rtn != 0:
                log_error(f"{p.url} failed.")
                onfail(p.url)
            else:
                stdout = p.get_stdout()
                url = p.url
                log_info(f"{p.url} resolved.")
                onresolve(url, stdout)
            procs[i] = None
        # Search for empty processes and replace them with more work from the workq,
        # attaching the url to the process so that it can be accessed during the
        # resolve phase.
        for i, _ in enumerate(procs):
            if procs[i] is None and len(url_work_q) > 0:
                # Push a new process onto the array, attaching the url to the process
                # as an attribute.
                cmd = f"transcribe_anything {url_work_q[0]}"
                log_debug(f"Cmd: {cmd}")
                proc = CapturingProcess(cmd, stdout=sys.stdout, stderr=sys.stderr)
                setattr(proc, "url", url_work_q[0])
                del url_work_q[0]
                procs[i] = proc
        time.sleep(1)


def unit_test() -> None:
    """Unit test bulk_transcribe."""
    set_logging_level(INFO)

    def onresolve(url: str, sub: str) -> None:
        print(url, sub)

    def onfail(url: str) -> None:
        print(f"Failed to resolve {url}")

    urls = [
        "https://www.youtube.com/watch?v=MIw9FZj7GqY",
        "https://www.youtube.com/watch?v=4cZb-Sh220E",
        "https://www.youtube.com/watch?v=EsoJ5k-yQMo",
        "https://www.youtube.com/watch?v=2CmYxfS-SDU",
    ]
    bulk_transcribe(urls, onresolve=onresolve, onfail=onfail, num_cpus=4)


if __name__ == "__main__":
    unit_test()
