# pylint: skip-file
# flake8: noqa

"""
Runs whisper api.
"""

import json  # type: ignore
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import wave
from pathlib import Path
from typing import Any, Optional

import static_ffmpeg  # type: ignore
import webvtt  # type: ignore

from transcribe_anything.cuda_available import CudaInfo
from transcribe_anything.generate_speaker_json import generate_speaker_json
from transcribe_anything.insanley_fast_whisper_reqs import get_environment

HERE = Path(__file__).parent
CUDA_INFO: Optional[CudaInfo] = None


def get_cuda_info() -> CudaInfo:
    """Get the computing device."""
    global CUDA_INFO  # pylint: disable=global-statement
    if CUDA_INFO is None:
        env = get_environment()
        py_file = HERE / "cuda_available.py"
        # tempfile
        with tempfile.TemporaryDirectory() as dir_name:
            temp = Path(dir_name) / "stdout.txt"
            abs_name = temp.absolute()
            try:
                env.run(
                    [str(py_file), "-o", str(abs_name)],
                    shell=False,
                    check=True,
                    universal_newlines=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 1:  # 1 is the expected return code
                    raise
                pass
                # print(f"Failed to run python {py_file} -o {abs_name}: {exc}")
                # print(f"stdout: {exc.stdout}")
                # print(f"stderr: {exc.stderr}")
            # stdout = cp.stdout
            stdout = temp.read_text(encoding="utf-8")
            try:
                CUDA_INFO = CudaInfo.from_json_str(stdout)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to decode json: {exc}") from exc
            assert CUDA_INFO is not None, f"Expected CUDA_INFO to be set, but the stdout was {stdout}"
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
        return 1
    return None


def convert_time_to_srt_format(timestamp: float) -> str:
    """Converts timestamp in seconds to SRT time format (hours:minutes:seconds,milliseconds)."""
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:02},{milliseconds:03}"


def convert_to_webvtt(srt_file: Path, out_webvtt_file: Path) -> None:
    """Convert to webvtt format."""
    STYLE_ELEMENT = """STYLE
    ::::cue {
    line: 80%;
    }
    """
    assert srt_file.suffix == ".srt"
    assert out_webvtt_file.suffix == ".vtt"
    webvtt.from_srt(str(srt_file)).save(str(out_webvtt_file))
    content = out_webvtt_file.read_text(encoding="utf-8")
    content = content.replace("WEBVTT\n\n", f"WEBVTT\n\n{STYLE_ELEMENT}")
    out_webvtt_file.write_text(content, encoding="utf-8")


def convert_json_to_srt(json_data: dict[str, Any], duration: float) -> str:
    """Converts JSON data from speech-to-text tool to SRT format."""
    srt_content = ""
    num_chunks = len(json_data["chunks"])
    for index, chunk in enumerate(json_data["chunks"], start=1):
        # start_time, end_time = chunk["timestamp"]
        time_pair = chunk["timestamp"]
        start_time = time_pair[0]
        end_time = time_pair[1]
        try:
            if start_time is None and end_time is None:
                print(f"Skipping chunk {index} because both start and end time are None.")
                stack_trace = traceback.format_stack()
                print("Stack trace: ", stack_trace)
                continue
            if end_time is None:
                # assert index == num_chunks
                if index != num_chunks:
                    print(f"Setting end time to duration because it's None for chunk {index}.")
                end_time = duration  # Sometimes happens at the end
            try:
                start_time_str = convert_time_to_srt_format(start_time)
            except Exception as exc:
                print(f"Failed to convert start time {start_time} to srt format: {exc}")
                stack_trace = traceback.format_stack()
                print("Stack trace: ", stack_trace)
                continue
            try:
                end_time_str = convert_time_to_srt_format(end_time)
            except Exception as exc:
                print(f"Failed to convert end time {end_time} to srt format: {exc}")
                stack_trace = traceback.format_stack()
                print("Stack trace: ", stack_trace)
                continue
        except Exception as exc:
            print(f"Failed to convert times for chunk {index}: {exc}")
            stack_trace = traceback.format_stack()
            print("Stack trace: ", stack_trace)
            continue
        try:
            text = str(chunk["text"]).strip()
            srt_content += f"{index}\n{start_time_str} --> {end_time_str}\n{text}\n\n"
        except Exception as exc:
            print(f"Failed to add chunk {index} to srt content: {exc}")
            stack_trace = traceback.format_stack()
            print("Stack trace: ", stack_trace)
            continue
    return srt_content


def convert_json_to_text(json_data: dict[str, Any]) -> str:
    """Converts JSON data from speech-to-text tool to text."""
    return json_data["text"]


def get_wave_duration(wave_file: Path) -> float:
    """Returns the duration of a wave file."""
    with wave.open(str(wave_file), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / float(rate)
        return duration


def trim_text_chunks(json_data: dict[str, Any]) -> None:
    """'text' chunks seem to have an extra space at the beginning, remove it."""

    # visit all the nodes in the json data, when we see one that has a 'text' key,
    # then apply a trim.
    def visit(node: dict[str, Any]) -> None:
        if isinstance(node, dict):
            if "text" in node:
                node["text"] = node["text"].strip()
            for key, value in node.items():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(json_data)


def run_insanely_fast_whisper(
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str,
    language: str,
    hugging_face_token: str | None = None,
    other_args: list[str] | None = None,
) -> None:
    """Runs insanely fast whisper."""
    # ffmpeg paths have to be installed or else the backend tool will fail.
    static_ffmpeg.add_paths()
    iso_env = get_environment()
    env = dict(os.environ.copy())
    if sys.platform == "darwin":
        # Attempts fixed recommended for the mps machines. This seems
        # to be necessary since a recent update.
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
        # env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device_id = get_device_id()
    cmd_list = []
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "out.json"
    if "/" not in model:
        # Assume it's not a namespace model, so add the namespace.
        model = f"openai/whisper-{model}"
    wave_duration = get_wave_duration(input_wav)
    # if sys.platform == "win32":
    # Set the text mode to UTF-8 on Windows.
    # cmd_list.extend(["cmd.exe", "/c"])
    # cmd_list.extend(["chcp", "65001", "&&"])
    cmd_list += [
        "insanely-fast-whisper",
        "--file-name",
        str(input_wav),
        "--device-id",
        f"{device_id}",
    ]
    if model:
        cmd_list += [
            "--model-name",
            model,
        ]
    cmd_list += [
        "--task",
        task,
        "--transcript-path",
        str(outfile),
    ]
    if hugging_face_token:
        cmd_list += ["--hf-token", hugging_face_token]
        # remove --hf-token from other_args if it's there.
        if other_args and "--hf-token" in other_args:
            idx = other_args.index("--hf-token")
            idx2 = idx + 1
            other_args.pop(idx2)
            other_args.pop(idx)
    if language:
        cmd_list += ["--language", language]
    batch_size = get_batch_size()
    if batch_size is not None:
        cmd_list += ["--batch-size", f"{batch_size}"]
    if other_args:
        cmd_list.extend(other_args)
    # Remove the empty strings.
    cmd_list = [x.strip() for x in cmd_list if x.strip()]
    cmd = subprocess.list2cmdline(cmd_list)
    sys.stderr.write(f"Running:\n  {cmd}\n")
    proc = iso_env.open_proc(  # pylint: disable=consider-using-with
        cmd_list,
        shell=False,
        universal_newlines=True,
        encoding="utf-8",
        env=env,
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
    proc.wait()
    assert outfile.exists(), f"Expected {outfile} to exist."
    json_text = outfile.read_text(encoding="utf-8")
    json_data = json.loads(json_text)
    trim_text_chunks(json_data)
    json_data_str = json.dumps(json_data, indent=2)

    if hugging_face_token:
        print("### HUGGING FACE TOKEN IS ACTIVE - GENERATING SPEAKER JSON ###")
        # Speaker diarization is active so generate the file
        try:
            speaker_json = generate_speaker_json(json_data)
            speaker_json_str = json.dumps(speaker_json, indent=2)
            speaker_json_file = output_dir / "speaker.json"
            speaker_json_file.write_text(speaker_json_str, encoding="utf-8")
        except Exception as exc:
            warnings.warn(f"Failed to generate speaker json beause of exception: {exc}")
    else:
        print("### HUGGING FACE TOKEN IS NOT ACTIVE - NO SPEAKER JSON GENERATED ###")

    # now write the pretty formatted json data back to the text file.
    outfile.write_text(json_data_str, encoding="utf-8")
    try:
        srt_content = convert_json_to_srt(json_data, wave_duration)
        srt_file = output_dir / "out.srt"
    except Exception as exc:
        print(f"Failed to convert to srt: {exc}")
        print("Json data: ", json_data_str)
        error_file = Path("transcribe-anything-error.json")
        error_file.write_text(json_text, encoding="utf-8")
        raise
    try:
        txt_content = convert_json_to_text(json_data)
    except Exception as exc:
        error_file = Path("transcribe-anything-error.json")
        error_file.write_text(json_text, encoding="utf-8")
        raise
    # Disable srt_wrapping because it breaks conversion to webvtt.
    srt_file.write_text(srt_content, encoding="utf-8")
    # srt_wrap(srt_file)
    # srt_content = srt_file.read_text(encoding="utf-8")
    txt_file = output_dir / "out.txt"
    txt_file.write_text(txt_content, encoding="utf-8")
    convert_to_webvtt(srt_file, output_dir / "out.vtt")
    # print srt file
    print(srt_content)
