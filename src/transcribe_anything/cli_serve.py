"""
Host-side entry point for ``transcribe-anything-serve``.

This script:
1. Parses the daemon's CLI args.
2. Validates them (non-loopback bind requires an auth token).
3. Resolves the daemon iso-env (built on first call) and execs the
   server runner inside it.

The host's transcribe_anything source is added to PYTHONPATH so the
iso-env's Python can ``import transcribe_anything.server_app`` even
though the iso-env doesn't have transcribe-anything installed itself —
the iso-env supplies the deps, the host supplies the source.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _src_root() -> Path:
    """Absolute path to the directory containing the ``transcribe_anything`` package."""
    import transcribe_anything

    pkg_dir = Path(transcribe_anything.__file__).resolve().parent
    return pkg_dir.parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe-anything-serve",
        description=("Launch the transcribe-anything daemon (FastAPI). " "Local loopback bind needs no auth; any non-loopback bind requires --auth-token."),
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8765, type=int, help="bind port (default: 8765)")

    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument("--auth-token", default=None, help="shared secret token (Authorization: Bearer)")
    auth_group.add_argument(
        "--auth-token-env",
        default=None,
        help="env-var NAME whose value is the auth token",
    )
    auth_group.add_argument(
        "--auth-token-file",
        default=None,
        type=Path,
        help="file path whose contents is the auth token",
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=[None, "cpu", "cuda", "insane", "insane-flash", "whisperx", "sensevoice", "mlx"],
        help="lock the daemon to this backend (default: auto-detect)",
    )
    parser.add_argument("--model", default="small", help="default whisper model (default: small)")
    parser.add_argument(
        "--allow-client-model",
        action="store_true",
        help="permit clients to override --model per request (off by default; switching models redownloads weights)",
    )
    parser.add_argument(
        "--allow-embed",
        action="store_true",
        help="permit clients to request --embed (burn subtitles into MP4). Off by default.",
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (never echoed to clients)")
    parser.add_argument(
        "--prefetch",
        default="lazy",
        choices=["lazy", "eager", "none"],
        help=(
            "lazy: model downloads on first request (default). "
            "eager: warm up the backend at startup so the first request is fast. "
            "none: refuse requests until weights are already cached locally."
        ),
    )
    parser.add_argument("--max-batch-size", type=int, default=None, help="clamp client-supplied batch_size to this")
    parser.add_argument("--max-queue", type=int, default=8, help="max queued jobs (default: 8)")
    parser.add_argument(
        "--max-upload-size",
        default=2 * 1024 * 1024 * 1024,
        type=int,
        help="max upload size in bytes (default: 2 GB)",
    )
    parser.add_argument("--artifact-ttl", default=3600, type=int, help="artifact retention in seconds (default: 3600)")
    parser.add_argument("--job-root", default=None, type=Path, help="directory for per-job scratch dirs")
    parser.add_argument(
        "--shutdown-grace",
        default=60,
        type=int,
        help="seconds to drain the queue on SIGTERM (default: 60)",
    )
    parser.add_argument(
        "--no-iso-env",
        action="store_true",
        help=(
            "run the server directly in the current Python interpreter instead of building the "
            "daemon iso-env. Requires fastapi/uvicorn/python-multipart already installed in this venv. "
            "Useful for dev/test."
        ),
    )

    return parser.parse_args(argv)


def _resolve_token(args: argparse.Namespace) -> Optional[str]:
    if args.auth_token:
        return args.auth_token
    if args.auth_token_env:
        value = os.environ.get(args.auth_token_env)
        if not value:
            raise SystemExit(f"--auth-token-env {args.auth_token_env!r} is empty or unset in the environment")
        return value
    if args.auth_token_file:
        path: Path = args.auth_token_file
        if not path.is_file():
            raise SystemExit(f"--auth-token-file {path} does not exist")
        return path.read_text(encoding="utf-8").strip()
    return None


def _config_from_args(args: argparse.Namespace) -> "ServerConfig":  # type: ignore[name-defined]
    from transcribe_anything.server_config import ServerConfig

    token = _resolve_token(args)
    return ServerConfig(
        host=args.host,
        port=args.port,
        auth_token=token,
        device=args.device,
        model=args.model,
        allow_client_model=bool(args.allow_client_model),
        allow_embed=bool(args.allow_embed),
        hf_token=args.hf_token,
        prefetch=args.prefetch,
        max_batch_size=args.max_batch_size,
        max_queue=args.max_queue,
        max_upload_size_bytes=args.max_upload_size,
        artifact_ttl_seconds=args.artifact_ttl,
        job_root=str(args.job_root) if args.job_root else None,
        shutdown_grace_seconds=args.shutdown_grace,
    )


def _run_in_iso_env(config) -> int:
    """Build the daemon iso-env and exec the runner inside it."""
    from transcribe_anything.server_config import config_to_env
    from transcribe_anything.server_reqs import get_environment

    iso_env = get_environment()
    src_root = str(_src_root())

    env = dict(os.environ)
    env.update(config_to_env(config))
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if src_root not in parts:
        parts.insert(0, src_root)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    sys.stderr.write(f"transcribe-anything-serve: starting daemon on {config.host}:{config.port} (iso-env)\n")
    proc = iso_env.open_proc(  # pylint: disable=consider-using-with
        ["python", "-m", "transcribe_anything.server_runner"],
        shell=False,
        env=env,
    )
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 130


def _run_inproc(config) -> int:
    """Run the server in the current Python interpreter (no iso-env)."""
    from transcribe_anything.server_app import run_server

    sys.stderr.write(f"transcribe-anything-serve: starting daemon on {config.host}:{config.port} (in-process)\n")
    try:
        run_server(config)
    except KeyboardInterrupt:
        return 130
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """``transcribe-anything-serve`` entry point."""
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    try:
        config = _config_from_args(args)
        # Defer validation to a try/except so we map ValueError → clean exit.
        config.validate()
    except (ValueError, SystemExit) as exc:
        sys.stderr.write(f"transcribe-anything-serve: {exc}\n")
        return 2

    if args.no_iso_env:
        return _run_inproc(config)
    try:
        return _run_in_iso_env(config)
    except Exception as exc:  # pylint: disable=broad-except
        sys.stderr.write(f"transcribe-anything-serve: failed to launch in iso-env ({exc}). " "Pass --no-iso-env to run in the current interpreter if you have " "fastapi/uvicorn installed locally.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
