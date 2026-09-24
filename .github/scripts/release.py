#!/usr/bin/env python3
"""Release helper for the Python auto-release workflow."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from email.parser import HeaderParser
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
ZERO_SHA = "0" * 40
NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")
DIST_SUFFIXES = (".whl", ".tar.gz")
FileReader = Callable[[str], Optional[str]]


def working_tree_reader(root: Path = ROOT) -> FileReader:
    def read(path: str) -> Optional[str]:
        file = root / path
        return file.read_text(encoding="utf-8") if file.is_file() else None

    return read


def git_reader(ref: str, root: Path = ROOT) -> FileReader:
    def read(path: str) -> Optional[str]:
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None

    return read


def read_project(read: FileReader) -> tuple[str, str]:
    import tomllib

    text = read("pyproject.toml")
    if text is None:
        raise ValueError("pyproject.toml not found")
    project = tomllib.loads(text)["project"]
    name = str(project["name"])
    if "version" in project:
        return name, str(project["version"])

    attr = (
        tomllib.loads(text)
        .get("tool", {})
        .get("setuptools", {})
        .get("dynamic", {})
        .get("version", {})
        .get("attr")
    )
    if not attr:
        raise ValueError("project has neither a static nor attr-based version")
    module, _, variable = attr.rpartition(".")
    module_path = module.replace(".", "/")
    for candidate in (
        f"src/{module_path}/__init__.py",
        f"src/{module_path}.py",
        f"{module_path}/__init__.py",
        f"{module_path}.py",
    ):
        source = read(candidate)
        if source is None:
            continue
        for node in ast.parse(source).body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return name, node.value.value
        raise ValueError(f"{candidate} does not assign {variable} a string literal")
    raise ValueError(f"cannot find the module for version attr {attr!r}")


def normalized_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def pypi_files(project: str, version: str) -> set[str]:
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise
    return {entry["filename"] for entry in data.get("urls", [])}


@dataclass
class Decision:
    should_build: bool
    should_publish: bool
    reason: str


def decide(
    event: str,
    ref: str,
    default_branch: str,
    dry_run: bool,
    current_version: str,
    previous_version: Optional[str],
    published_files: set[str],
) -> Decision:
    on_default_branch = ref == f"refs/heads/{default_branch}"
    if event == "pull_request":
        return Decision(True, False, "pull request: dry-run build only")
    if event == "workflow_dispatch":
        if dry_run:
            return Decision(True, False, "manual dispatch with dry-run")
        if not on_default_branch:
            return Decision(True, False, f"publishing only runs on {default_branch}")
        if published_files:
            return Decision(False, False, f"{current_version} is already on PyPI")
        return Decision(True, True, f"manual dispatch: publishing {current_version}")
    if event == "push":
        if not on_default_branch:
            return Decision(False, False, f"push to {ref} is not {default_branch}")
        if previous_version == current_version:
            return Decision(False, False, f"version unchanged ({current_version})")
        if published_files:
            return Decision(False, False, f"{current_version} is already on PyPI")
        return Decision(True, True, f"version bumped {previous_version or '(unknown)'} -> {current_version}")
    return Decision(False, False, f"unsupported event {event!r}")


def dist_files(dist: Path) -> list[str]:
    return sorted(
        path.name
        for path in dist.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.name.endswith(DIST_SUFFIXES)
    )


def verify_dist(dist: Path, name: str, version: str) -> list[str]:
    norm = normalized_dist_name(name)
    files = dist_files(dist)
    wheels = [file for file in files if file.endswith(".whl")]
    sdists = [file for file in files if file.endswith(".tar.gz")]
    expected_wheel = f"{norm}-{version}-py3-none-any.whl"
    expected_sdist = f"{norm}-{version}.tar.gz"
    errors = []
    if wheels != [expected_wheel]:
        errors.append(f"expected one pure-Python wheel {expected_wheel}, found {wheels}")
    if sdists != [expected_sdist]:
        errors.append(f"expected one sdist {expected_sdist}, found {sdists}")
    if expected_wheel not in wheels:
        return errors
    dist_info = f"{norm}-{version}.dist-info"
    with zipfile.ZipFile(dist / expected_wheel) as wheel:
        names = set(wheel.namelist())
        metadata_path = f"{dist_info}/METADATA"
        if metadata_path not in names:
            errors.append(f"{expected_wheel} has no {metadata_path}")
        else:
            metadata = HeaderParser().parsestr(wheel.read(metadata_path).decode("utf-8"))
            if normalized_dist_name(metadata.get("Name", "")) != norm:
                errors.append(f"wheel METADATA Name is {metadata.get('Name')!r}")
            if metadata.get("Version") != version:
                errors.append(f"wheel METADATA Version is {metadata.get('Version')!r}")
        if f"{dist_info}/entry_points.txt" not in names:
            errors.append(f"{expected_wheel} declares no console scripts")
        native = sorted(name for name in names if name.endswith(NATIVE_SUFFIXES))
        if native:
            errors.append(f"pure-Python wheel contains native files: {native}")
    return errors


def wait_pypi(
    project: str,
    version: str,
    expected: set[str],
    timeout_s: float,
    interval_s: float,
    fetch: Callable[[str, str], set[str]] = pypi_files,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        missing = expected - fetch(project, version)
        if not missing:
            return True
        if time.monotonic() >= deadline:
            print(f"still missing on PyPI: {sorted(missing)}", file=sys.stderr)
            return False
        time.sleep(interval_s)


def write_outputs(outputs: dict[str, str]) -> None:
    for key, value in outputs.items():
        print(f"{key}={value}")
    if output_file := os.environ.get("GITHUB_OUTPUT"):
        with open(output_file, "a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                handle.write(f"{key}={value}\n")


def previous_version_at(before: str) -> Optional[str]:
    if not before or before == ZERO_SHA:
        return None
    try:
        return read_project(git_reader(before))[1]
    except (ValueError, KeyError):
        return None


def cmd_version(args: argparse.Namespace) -> int:
    reader = git_reader(args.ref) if args.ref else working_tree_reader()
    name, version = read_project(reader)
    print(f"{name} {version}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    name, version = read_project(working_tree_reader())
    previous = previous_version_at(args.before) if args.event == "push" else None
    on_default_branch = args.ref == f"refs/heads/{args.default_branch}"
    needs_pypi_check = (args.event == "push" and on_default_branch and previous != version) or (
        args.event == "workflow_dispatch" and on_default_branch and args.dry_run.strip().lower() in ("false", "0", "no")
    )
    published = pypi_files(name, version) if needs_pypi_check else set()
    decision = decide(
        args.event,
        args.ref,
        args.default_branch,
        args.dry_run.strip().lower() not in ("false", "0", "no"),
        version,
        previous,
        published,
    )
    print(f"::notice title=Auto-release::{decision.reason}")
    write_outputs(
        {
            "project": name,
            "version": version,
            "should_build": str(decision.should_build).lower(),
            "should_publish": str(decision.should_publish).lower(),
        }
    )
    return 0


def cmd_verify_dist(args: argparse.Namespace) -> int:
    name, version = read_project(working_tree_reader())
    if args.version and args.version != version:
        print(f"expected version {args.version}, project is {version}", file=sys.stderr)
        return 1
    errors = verify_dist(Path(args.dist), name, version)
    for error in errors:
        print(f"::error title=Distribution check::{error}")
    if not errors:
        print(f"{name} {version}: sdist and pure-Python wheel verified")
    return 1 if errors else 0


def cmd_wait_pypi(args: argparse.Namespace) -> int:
    expected = set(dist_files(Path(args.dist)))
    if wait_pypi(args.project, args.version, expected, args.timeout, args.interval):
        print(f"PyPI lists all {len(expected)} files for {args.project} {args.version}")
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    version = sub.add_parser("version")
    version.add_argument("--ref", default="")
    version.set_defaults(func=cmd_version)
    detect = sub.add_parser("detect")
    detect.add_argument("--event", required=True)
    detect.add_argument("--ref", required=True)
    detect.add_argument("--default-branch", required=True)
    detect.add_argument("--before", default="")
    detect.add_argument("--dry-run", default="true")
    detect.set_defaults(func=cmd_detect)
    verify = sub.add_parser("verify-dist")
    verify.add_argument("--dist", default="dist")
    verify.add_argument("--version", default="")
    verify.set_defaults(func=cmd_verify_dist)
    wait = sub.add_parser("wait-pypi")
    wait.add_argument("--project", required=True)
    wait.add_argument("--version", required=True)
    wait.add_argument("--dist", default="dist")
    wait.add_argument("--timeout", type=float, default=300)
    wait.add_argument("--interval", type=float, default=5)
    wait.set_defaults(func=cmd_wait_pypi)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
