# pylint: skip-file
# flake8: noqa

"""
Runs whisper api.
"""

import shutil
import sys
import time
import json
from pathlib import Path
import subprocess
from typing import Optional, Any
from filelock import FileLock

from isolated_environment import IsolatedEnvironment  # type: ignore
from transcribe_anything.cuda_available import CudaInfo

HERE = Path(__file__).parent
ENV: Optional[IsolatedEnvironment] = None
CUDA_INFO: Optional[CudaInfo] = None
ENV_LOCK = FileLock(HERE / "insane_whisper_env.lock")

# Set the versions
TENSOR_VERSION = "2.1.2"
CUDA_VERSION = "cu121"
TENSOR_CUDA_VERSION = f"{TENSOR_VERSION}+{CUDA_VERSION}"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"

def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# for insanely fast whisper, use:
#   pipx install insanely-fast-whisper --python python3.11

def has_nvidia_smi() -> bool:
    """Returns True if nvidia-smi is installed."""
    return shutil.which("nvidia-smi") is not None

def get_environment() -> IsolatedEnvironment:
    """Returns the environment."""
    global ENV  # pylint: disable=global-statement
    with ENV_LOCK:
        if ENV is not None:
            return ENV
        venv_dir = HERE / "venv" / "insanely_fast_whisper"
        env = IsolatedEnvironment(venv_dir)
        if not venv_dir.exists():
            env.install_environment()
            if has_nvidia_smi():
                env.pip_install(f"torch=={TENSOR_VERSION}", extra_index=EXTRA_INDEX_URL)
            else:
                env.pip_install(f"torch=={TENSOR_VERSION}")
            env.pip_install("openai-whisper")
            env.pip_install("insanely-fast-whisper")
        ENV = env
        return env


def get_cuda_info() -> CudaInfo:
    """Get the computing device."""
    global CUDA_INFO  # pylint: disable=global-statement
    if CUDA_INFO is None:
        iso_env = get_environment()
        env = iso_env.environment()
        py_file = HERE / "cuda_available.py"
        cp: subprocess.CompletedProcess = subprocess.run([
            "python", py_file
        ], check=False, env=env, universal_newlines=True, stdout=subprocess.PIPE)
        stdout = cp.stdout
        CUDA_INFO = CudaInfo.from_json_str(stdout)
    return CUDA_INFO

def get_device_id() -> str:
    """Get the device id."""
    # on mac, we just return "mps"
    if sys.platform == "darwin":
        return "mps"
    cuda_info = get_cuda_info()
    if not cuda_info.cuda_available:
        raise ValueError("CUDA is not available.")
    device_id = cuda_info.cuda_devices[0].device_id
    return f"{device_id}"

def get_batch_size() -> int | None:
    """Returns the batch size."""
    if sys.platform == "darwin":
        return 4
    return None

def convert_time_to_srt_format(timestamp: float) -> str:
    """Converts timestamp in seconds to SRT time format (hours:minutes:seconds,milliseconds)."""
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def convert_json_to_srt(json_data: dict[str, Any]) -> str:
    """Converts JSON data from speech-to-text tool to SRT format."""
    srt_content = ""
    for index, chunk in enumerate(json_data['chunks'], start=1):
        start_time, end_time = chunk['timestamp']
        start_time_str = convert_time_to_srt_format(start_time)
        end_time_str = convert_time_to_srt_format(end_time)
        text = str(chunk['text']).strip()
        srt_content += f"{index}\n{start_time_str} --> {end_time_str}\n{text}\n\n"
    return srt_content

def convert_json_to_text(json_data: dict[str, Any]) -> str:
    """Converts JSON data from speech-to-text tool to text."""
    text = ""
    for chunk in json_data['chunks']:
        text += str(chunk['text']).strip() + "\n"
    return text


def run_insanely_fast_whisper(  # pylint: disable=too-many-arguments
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    other_args: Optional[list[str]]
) -> None:
    """Runs insanely fast whisper."""
    iso_env = get_environment()
    device_id = get_device_id()
    cmd_list = []
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "out.json"
    model = f"openai/whisper-{model}"
    if sys.platform == "win32":
        # Set the text mode to UTF-8 on Windows.
        cmd_list.extend(["chcp", "65001", "&&"])
    cmd_list += [
        "insanely-fast-whisper",
        "--file-name", str(input_wav),
        "--device-id", f"{device_id}",
        "--model-name", model,
        "--task", task,
        "--language", language,
        "--transcript-path", str(outfile),
    ]
    batch_size = get_batch_size()
    if batch_size is not None:
        cmd_list += ["--batch-size", f"{batch_size}"]
    if other_args:
        cmd_list.extend(other_args)
    # Remove the empty strings.
    cmd_list = [x.strip() for x in cmd_list if x.strip()]
    cmd = " ".join(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd, shell=True, universal_newlines=True,
        encoding="utf-8", env=iso_env.environment()
    )
    while True:
        rtn = proc.poll()
        if rtn is None:
            time.sleep(0.25)
            continue
        if rtn != 0:
            msg = f"Failed to execute {cmd}\n "
            raise OSError(msg)
        break
    assert outfile.exists(), f"Expected {outfile} to exist."
    json_text = outfile.read_text(encoding="utf-8")
    json_data = json.loads(json_text)
    srt_content = convert_json_to_srt(json_data)
    srt_file = output_dir / "out.srt"
    txt_content = convert_json_to_text(json_data)
    srt_file.write_text(srt_content, encoding="utf-8")
    txt_file = output_dir / "out.txt"
    txt_file.write_text(txt_content, encoding="utf-8")

