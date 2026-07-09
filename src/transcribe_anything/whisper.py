"""
Runs whisper api.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import get_runtime_venv_dir, has_nvidia_smi

# from isolated_environment import isolated_environment  # type: ignore


HERE = Path(__file__).parent
CUDA_AVAILABLE: Optional[bool] = None

# Set the versions
TENSOR_VERSION = "2.10.0"
CUDA_VERSION = "cu128"
XPU_VERSION = "xpu"
# torch 2.10.0+xpu requires exactly this triton version; pin it so the
# resolved artifact is deterministic (index-pinned AND version-pinned).
TRITON_XPU_VERSION = "3.6.0"
CUDA_EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"
XPU_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/xpu"

IS_MAC = sys.platform == "darwin"


def build_pyproject_toml(has_nvidia: bool, use_xpu: bool = False) -> str:
    """Build the uv pyproject content for the isolated whisper env."""
    needs_extra_index = (not IS_MAC and has_nvidia) or use_xpu
    content_lines: list[str] = []

    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")

    content_lines.append("[project]")
    content_lines.append('name = "project"')
    content_lines.append('version = "0.1.0"')
    content_lines.append('requires-python = "==3.10.*"')
    content_lines.append("dependencies = [")
    content_lines.append('  "openai-whisper==20240930",')
    content_lines.append('  "numpy==1.26.4",')
    # f"torch=={TENSOR_VERSION}"
    if needs_extra_index:
        if use_xpu:
            content_lines.append(f'  "torch=={TENSOR_VERSION}+{XPU_VERSION}",')
            # torch+xpu's triton backend (renamed from pytorch-triton-xpu to
            # triton-xpu in torch 2.10). The needed version only exists on the
            # pytorch-xpu index (PyPI has a stale 3.3.0b1 beta), so it must be
            # a declared dependency for its [tool.uv.sources] pin to apply.
            content_lines.append(f"  \"triton-xpu=={TRITON_XPU_VERSION}; sys_platform == 'linux' or sys_platform == 'win32'\",")
        else:
            content_lines.append(f'  "torch=={TENSOR_VERSION}+{CUDA_VERSION}",')
    content_lines.append("]")

    # Constrain setuptools < 82 for build isolation because
    # openai-whisper imports pkg_resources which was removed in setuptools 82.
    content_lines.append("")
    content_lines.append("[tool.uv]")
    if use_xpu:
        # XPU wheels only exist for linux/windows; keep universal
        # resolution from failing on the mac split.
        content_lines.append("environments = [")
        content_lines.append("    \"sys_platform == 'win32'\",")
        content_lines.append("    \"sys_platform == 'linux'\",")
        content_lines.append("]")
    content_lines.append('build-constraint-dependencies = ["setuptools<82"]')

    if needs_extra_index:
        # The extra torch index is explicit: only packages pinned to it via
        # [tool.uv.sources] resolve from it, everything else stays on PyPI.
        content_lines.append("")
        content_lines.append("[tool.uv.sources]")
        content_lines.append("torch = [")
        if use_xpu:
            content_lines.append("  { index = 'pytorch-xpu' },")
        else:
            content_lines.append("  { index = 'pytorch-cu128' },")
        content_lines.append("]")
        if use_xpu:
            content_lines.append("triton-xpu = [")
            content_lines.append("  { index = 'pytorch-xpu' },")
            content_lines.append("]")
        content_lines.append("[[tool.uv.index]]")
        if use_xpu:
            content_lines.append('name = "pytorch-xpu"')
            content_lines.append(f'url = "{XPU_EXTRA_INDEX_URL}"')
        else:
            content_lines.append('name = "pytorch-cu128"')
            content_lines.append(f'url = "{CUDA_EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")

    return "\n".join(content_lines)


def get_environment(use_xpu: bool = False) -> IsoEnv:
    """Returns the environment."""
    venv_dir = get_runtime_venv_dir("whisper-xpu" if use_xpu else "whisper")
    content = build_pyproject_toml(has_nvidia=has_nvidia_smi(), use_xpu=use_xpu)
    pyproject_toml = PyProjectToml(content)
    args = IsoEnvArgs(venv_dir, build_info=pyproject_toml)
    env = IsoEnv(args)
    return env


def get_computing_device() -> str:
    """Get the computing device."""
    global CUDA_AVAILABLE  # pylint: disable=global-statement
    if CUDA_AVAILABLE is None:
        env = get_environment()
        py_file = HERE / "cuda_available.py"
        cp = env.run([py_file], shell=False, check=False)
        CUDA_AVAILABLE = cp.returncode == 0
    return "cuda" if CUDA_AVAILABLE else "cpu"


def _perform_cmd_substituions(cmd_list: list[str]) -> list[str]:
    new_cmd_list = []
    for cmd in cmd_list:
        if cmd == "hf-token":
            print("arg substitution: hf-token -> hf_token")
            cmd = "hf_token"
        new_cmd_list.append(cmd)
    return new_cmd_list


def run_whisper(  # pylint: disable=too-many-arguments
    input_wav: Path,
    device: str,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    other_args: list[str] | None = None,
) -> None:
    """Runs whisper."""
    env = get_environment()
    cmd_list = []
    # if sys.platform == "win32":
    #     # Set the text mode to UTF-8 on Windows.
    #     cmd_list.extend(["C:\Windows\System32\chcp.com", "65001", "&&"])
    cmd_list.append("whisper")
    cmd_list.append(str(input_wav))
    cmd_list.append("--device")
    cmd_list.append(device)
    if model:
        cmd_list.append("--model")
        cmd_list.append(model)
    cmd_list.append("--output_dir")
    cmd_list.append(str(output_dir))
    cmd_list.append("--task")
    cmd_list.append(task)
    if language:
        # cmd_list.append(f'--language "{language}"')
        cmd_list.append("--language")
        cmd_list.append(language)
    if other_args:
        cmd_list.extend(other_args)
    cmd_list = _perform_cmd_substituions(cmd_list)

    # Remove the empty strings.
    cmd_list = [str(x).strip() for x in cmd_list if str(x).strip()]
    # cmd = " ".join(cmd_list)
    cmd = subprocess.list2cmdline(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = env.open_proc(cmd_list, shell=False)
    while True:
        rtn = proc.poll()
        if rtn is None:
            time.sleep(0.25)
            continue
        if rtn != 0:
            msg = f"Failed to execute {cmd}\n "
            raise OSError(msg)
        break
