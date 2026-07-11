"""Compatibility guard for XPU backends managed by uv-iso-env."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from filelock import FileLock
from iso_env import IsoEnv, IsoEnvArgs  # type: ignore
from iso_env.api import installed, purge  # type: ignore
from iso_env.util import get_verbose_from_env  # type: ignore


class XpuIsoEnv(IsoEnv):
    """Install an XPU backend with an XPU-wheel-compatible Python.

    uv-iso-env 1.0.44 invokes ``uv venv`` and ``uv pip compile`` without a
    Python request. That makes both commands inherit the host interpreter,
    even when the generated project declares an older ``requires-python``.
    It also marks the environment installed after a compile failure. Keep
    this strict installer local to XPU environments until upstream exposes
    interpreter selection and propagates compile failures.
    """

    def __init__(self, args: IsoEnvArgs, *, python: str) -> None:
        super().__init__(args)
        self.python = python

    @property
    def _compiled_requirements(self) -> Path:
        return self.args.venv_path / "requirements.compiled.txt"

    def _is_complete(self, *, verbose: bool) -> bool:
        return installed(self.args, verbose=verbose) and self._compiled_requirements.is_file() and self._compiled_requirements.stat().st_size > 0

    def _run_uv(self, cmd: list[str], *, cwd: Path) -> None:
        try:
            subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True, shell=False)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "<no output captured>").strip()
            raise RuntimeError(f"XPU dependency installation failed while running {subprocess.list2cmdline(cmd)}: {detail}") from exc

    def _ensure_installed(self) -> None:
        path = self.args.venv_path.resolve()
        lock_path = path.parent / f".{path.name}.lock"
        lock_path.parent.mkdir(exist_ok=True, parents=True)
        verbose = get_verbose_from_env()

        with FileLock(str(lock_path), timeout=300):
            if self._is_complete(verbose=verbose):
                return
            purge(path)
            path.mkdir(exist_ok=True, parents=True)
            (path / "pyproject.toml").write_text(str(self.args.build_info), encoding="utf-8")
            uv = shutil.which("uv")
            if uv is None:
                raise RuntimeError("XPU dependency installation failed: uv was not found on PATH")
            try:
                self._run_uv([uv, "venv", "--python", self.python], cwd=path)
                self._run_uv(
                    [
                        uv,
                        "pip",
                        "compile",
                        "pyproject.toml",
                        "--python",
                        self.python,
                        "--output-file",
                        "requirements.compiled.txt",
                    ],
                    cwd=path,
                )
            except BaseException:
                (path / "installed").unlink(missing_ok=True)
                raise
            (path / "installed").touch()

    def run(self, cmd_list: list[str], **process_args: Any) -> subprocess.CompletedProcess[Any]:
        self._ensure_installed()
        return super().run(cmd_list, **process_args)

    def open_proc(self, cmd_list: list[str], **process_args: Any) -> subprocess.Popen[Any]:
        self._ensure_installed()
        return super().open_proc(cmd_list, **process_args)
