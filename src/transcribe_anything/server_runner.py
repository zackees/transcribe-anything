"""
In-iso-env runner for the transcribe-anything daemon.

Boots inside the daemon iso-env (built by :mod:`server_reqs`). Receives
its config via env vars set by :mod:`cli_serve` (round-tripped through
:func:`server_app.config_to_env` / :func:`server_app.config_from_env`),
then hands off to :func:`server_app.run_server`.

This module is intentionally tiny — the FastAPI app and JobStore live in
:mod:`server_app` so they can be unit-tested in the host venv.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    from transcribe_anything.server_app import run_server
    from transcribe_anything.server_config import config_from_env

    config = config_from_env(os.environ)
    run_server(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
