"""
Runs whisper api.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import has_nvidia_smi

# from isolated_environment import isolated_environment  # type: ignore


HERE = Path(__file__).parent
CUDA_AVAILABLE: Optional[bool] = None

# Set the versions
TENSOR_VERSION = "2.2.1"
CUDA_VERSION = "cu121"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"

IS_MAC = sys.platform == "darwin"


def get_environment() -> IsoEnv:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "whisper"
    needs_extra_index = not IS_MAC and has_nvidia_smi()
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
        content_lines.append(f'  "torch=={TENSOR_VERSION}+{CUDA_VERSION}",')
    content_lines.append("]")

    needs_extra_index = not IS_MAC and has_nvidia_smi()
    if needs_extra_index:
        content_lines.append("[tool.uv.sources]")
        content_lines.append("torch = [")
        content_lines.append("  { index = 'pytorch-cu121' },")
        content_lines.append("]")
        content_lines.append("[[tool.uv.index]]")
        content_lines.append('name = "pytorch-cu121"')
        content_lines.append(f'url = "{EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")

    # if not IS_MAC and has_nvidia_smi():
    #     deps.append(
    #         f"torch=={TENSOR_VERSION}+{CUDA_VERSION} --extra-index-url {EXTRA_INDEX_URL}"
    #     )
    # else:
    #     deps.append(f"torch=={TENSOR_VERSION}")
    content = "\n".join(content_lines)
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
