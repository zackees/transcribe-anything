"""
    Api for using transcribe_anything from python. Allows bulk processing.
"""

import multiprocessing
import os
import subprocess
import tempfile
import time
from typing import Any, Callable, List


def fetch_subtitle(url: str) -> str:
    """Fetches the subtitle and returns it."""
    tmp_file = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
        suffix=".txt", delete=False
    )  # pylint: disable=R1732
    tmp_file.close()
    try:
        cmd = f"transcribe_anything {url} {tmp_file.name}"
        print(cmd)
        subprocess.check_output(cmd, shell=True)
        with open(tmp_file.name) as fd:
            vid_sub = fd.read()
        print(f"Writing subtitle to {url}")
        return vid_sub
    finally:
        os.remove(tmp_file.name)


def bulk_fetch_subtitles(
    urls: List[str],
    onresolve: Callable[[str, str], None],
    onfail: Callable[[str], None],
    num_cpus: int = -1,
) -> None:
    """On each completed url onresolve(url, subtitle) will be called."""
    url_work_q = list(urls)
    if num_cpus == -1:
        num_cpus = max(1, int(multiprocessing.cpu_count() / 2))
    print(f"Num cpus={num_cpus}")
    procs: List[Any] = [None for _ in range(num_cpus)]
    while True:
        n_active_procs = len([p for p in procs if p])
        if n_active_procs == 0 and len(url_work_q) == 0:
            # No running procs and no commands left to run so bail.
            break
        # Scan through and poll active processes and look to remove processes
        # that are finished and replace them with None.
        for i, p in enumerate(procs):
            if p is None:
                continue
            rtn = p.poll()
            if rtn is None:
                # Still running
                continue
            print(f"{p.url} finished running.")
            if rtn != 0:
                print(f"{p.url} failed.")
                onfail(p.url)
            else:
                stdout = p.stdout.read()
                url = p.url
                print(f"{p.url} resolved.")
                onresolve(url, stdout)
            procs[i] = None
        # Search for empty processes and replace them with more work from the workq.
        for i, _ in enumerate(procs):
            if procs[i] is None and len(url_work_q) > 0:
                # Push a new process onto the array, attaching the url to the process
                # as an attribute.
                cmd = f"transcribe_anything {url_work_q[0]}"
                print(f"Cmd: {cmd}")
                proc = subprocess.Popen(  # pylint: disable=R1732
                    cmd, shell=True, stdout=subprocess.PIPE
                )
                setattr(proc, "url", url_work_q[0])
                del url_work_q[0]
                procs[i] = proc
        time.sleep(0.2)


def unit_test() -> None:
    """Unit test bulk_fetch_subtitles."""

    def onresolve(url: str, sub: str) -> None:
        print(url, sub)

    def onfail(url: str) -> None:
        print(f"Failed to resolve {url}")

    urls = ["https://www.youtube.com/watch?v=Erk4_jFDjzQ"]
    bulk_fetch_subtitles(urls, onresolve=onresolve, onfail=onfail, num_cpus=2)


if __name__ == "__main__":
    unit_test()
