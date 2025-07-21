"""
Runs whisper api with Apple MLX support using lightning-whisper-mlx.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import webvtt  # type: ignore
from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

HERE = Path(__file__).parent


def get_mlx_cache_dir() -> Path:
    """Get the cache directory for MLX models, consistent with other backends.

    Returns the cache directory path where MLX models should be stored.
    This matches the pattern used by other whisper backends (~/.cache/whisper/)
    but keeps MLX models in a separate subdirectory for organization.
    """
    # Use the same cache pattern as standard whisper but with mlx_models subdirectory
    cache_dir = Path.home() / ".cache" / "whisper" / "mlx_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_environment() -> IsoEnv:
    """Returns the environment for lightning-whisper-mlx."""
    venv_dir = HERE / "venv" / "whisper_mlx"
    content_lines: list[str] = []

    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")
    content_lines.append("[project]")
    content_lines.append('name = "project"')
    content_lines.append('version = "0.1.0"')
    content_lines.append('requires-python = ">=3.10"')
    content_lines.append("dependencies = [")
    content_lines.append('  "lightning-whisper-mlx @ git+https://github.com/aj47/lightning-whisper-mlx.git",')
    content_lines.append('  "webvtt-py",')
    content_lines.append('  "numpy",')
    content_lines.append("]")
    content = "\n".join(content_lines)

    # Debug: Log the pyproject.toml content hash to track changes
    # content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    # print(f"Debug: whisper_mac.py pyproject.toml hash: {content_hash}", file=sys.stderr)

    pyproject_toml = PyProjectToml(content)
    args = IsoEnvArgs(venv_dir, build_info=pyproject_toml)
    env = IsoEnv(args)
    return env


def _format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format."""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# Constants for timestamp conversion
# Based on Whisper's audio processing: SAMPLE_RATE=16000, HOP_LENGTH=160
# Each mel frame represents HOP_LENGTH/SAMPLE_RATE = 160/16000 = 0.01 seconds
FRAME_HOP_SECONDS = 0.01


def _json_to_srt(json_data: Dict[str, Any]) -> str:
    """Convert lightning-whisper-mlx JSON output to SRT format."""
    srt_content = ""

    if "segments" not in json_data:
        # If no segments, try to create a single segment from the full text
        if "text" in json_data:
            srt_content = "1\n00:00:00,000 --> 00:01:00,000\n" + json_data["text"] + "\n\n"
        return srt_content

    for i, segment in enumerate(json_data["segments"], start=1):
        # Handle both old format (start/end) and new format (list with start, end, text)
        if isinstance(segment, list) and len(segment) >= 3:
            # New format: [start_seek, end_seek, text]
            # Convert mel frame indices to seconds using correct conversion factor
            start_time = segment[0] * FRAME_HOP_SECONDS
            end_time = segment[1] * FRAME_HOP_SECONDS
            text = segment[2].strip()

            # Sanity check for timestamp ordering
            if end_time < start_time:
                sys.stderr.write(f"Warning: segment {i} end time {end_time} < start time {start_time}, skipping\n")
                continue
        else:
            # Old format: dict with start/end/text (already in seconds)
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 5)  # Default to 5 seconds if no end time
            text = segment.get("text", "").strip()

        if text:  # Only include non-empty segments
            srt_content += f"{i}\n"
            srt_content += f"{_format_timestamp(start_time)} --> {_format_timestamp(end_time)}\n"
            srt_content += f"{text}\n\n"

    return srt_content


def _generate_output_files(json_data: Dict[str, Any], output_dir: Path, initial_prompt: str | None = None) -> None:
    """Generate all output files from the transcription result."""
    # 1. SRT file
    srt_content = _json_to_srt(json_data)
    srt_file = output_dir / "out.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt_content)

    # 2. Text file
    txt_file = output_dir / "out.txt"
    text_content = json_data.get("text", "")
    # Remove initial prompt from the beginning if it was used
    if initial_prompt and text_content.startswith(initial_prompt.strip()):
        text_content = text_content[len(initial_prompt.strip()) :].strip()
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text_content)

    # 3. JSON file
    json_out_file = output_dir / "out.json"
    with open(json_out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 4. VTT file
    try:
        vtt_file = output_dir / "out.vtt"
        webvtt.from_srt(srt_file).save(vtt_file)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to convert SRT to VTT: {e}\n")


def _parse_other_args(other_args: list[str]) -> dict[str, Any]:
    """Parse other_args into a dictionary of parameters for lightning-whisper-mlx.

    Supported arguments:
    - --initial_prompt: Custom vocabulary/context prompt
    - --language: Language code (e.g., 'en', 'es', 'fr')
    - --task: 'transcribe' or 'translate'
    - --word_timestamps: Enable word-level timestamps
    - --verbose: Enable verbose output
    - --temperature: Sampling temperature
    - --batch_size: Batch size for processing
    """
    if not other_args:
        return {}

    parsed_args: dict = {}
    i = 0
    while i < len(other_args):
        arg = other_args[i]

        if arg == "--initial_prompt" and i + 1 < len(other_args):
            parsed_args["initial_prompt"] = other_args[i + 1]
            i += 2
        elif arg == "--language" and i + 1 < len(other_args):
            parsed_args["language"] = other_args[i + 1]
            i += 2
        elif arg == "--task" and i + 1 < len(other_args):
            parsed_args["task"] = other_args[i + 1]
            i += 2
        elif arg == "--word_timestamps":
            parsed_args["word_timestamps"] = True
            i += 1
        elif arg == "--verbose":
            parsed_args["verbose"] = True
            i += 1
        elif arg == "--temperature" and i + 1 < len(other_args):
            try:
                parsed_args["temperature"] = float(other_args[i + 1])
            except ValueError:
                sys.stderr.write(f"Warning: Invalid temperature value '{other_args[i + 1]}', using default\n")
            i += 2
        elif arg == "--batch_size" and i + 1 < len(other_args):
            try:
                parsed_args["batch_size"] = int(other_args[i + 1])
            except ValueError:
                sys.stderr.write(f"Warning: Invalid batch_size value '{other_args[i + 1]}', using default\n")
            i += 2
        else:
            # Skip unsupported arguments with a warning
            if arg.startswith("--"):
                sys.stderr.write(f"Warning: Argument '{arg}' is not supported by lightning-whisper-mlx backend\n")
                if i + 1 < len(other_args) and not other_args[i + 1].startswith("--"):
                    i += 2  # Skip both the argument and its value
                else:
                    i += 1  # Skip just the argument
            else:
                i += 1

    return parsed_args


def run_whisper_mac_mlx(  # pylint: disable=too-many-arguments
    input_wav: Path,
    model: str,
    output_dir: Path,
    language: str | None = None,
    task: str = "transcribe",
    other_args: list[str] | None = None,
) -> None:
    """Runs whisper with Apple MLX acceleration using lightning-whisper-mlx.

    This function uses the lightning-whisper-mlx library to transcribe audio using Apple's
    MLX framework for acceleration on Apple Silicon hardware.
    It generates SRT, VTT, TXT, and JSON output files.

    Args:
        input_wav: Path to the input WAV file
        model: Whisper model to use (tiny, small, medium, large, etc.)
        output_dir: Directory to save output files
        language: Language code (e.g., 'en', 'es', 'fr'). If None, auto-detect.
        task: Task to perform ('transcribe' or 'translate')
        other_args: Additional arguments to pass to the transcription
    """
    input_wav_abs = input_wav.resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Parse additional arguments
    parsed_args = _parse_other_args(other_args or [])

    # Override with explicit parameters
    if language:
        parsed_args["language"] = language
    if task:
        parsed_args["task"] = task

    # Set defaults
    batch_size = parsed_args.get("batch_size", 12)
    initial_prompt = parsed_args.get("initial_prompt")
    # unusued
    # word_timestamps = parsed_args.get("word_timestamps", False)
    verbose = parsed_args.get("verbose", False)
    # unused
    # temperature = parsed_args.get("temperature", 0.0)

    # Get the environment and run transcription
    env = get_environment()

    # Get the cache directory for consistent model storage
    cache_dir = get_mlx_cache_dir()

    # Create a Python script to run the transcription in the isolated environment
    script_content = f"""
import sys
import json
import os
from pathlib import Path

try:
    from lightning_whisper_mlx.transcribe import transcribe_audio
    from huggingface_hub import hf_hub_download

    # Model mapping (same as in lightning_whisper_mlx)
    models = {{
        "tiny": {{"base": "mlx-community/whisper-tiny"}},
        "small": {{"base": "mlx-community/whisper-small-mlx"}},
        "base": {{"base": "mlx-community/whisper-base-mlx"}},
        "medium": {{"base": "mlx-community/whisper-medium-mlx"}},
        "large": {{"base": "mlx-community/whisper-large-mlx"}},
        "large-v2": {{"base": "mlx-community/whisper-large-v2-mlx"}},
        "large-v3": {{"base": "mlx-community/whisper-large-v3-mlx"}},
        "distil-small.en": {{"base": "mustafaaljadery/distil-whisper-mlx"}},
        "distil-medium.en": {{"base": "mustafaaljadery/distil-whisper-mlx"}},
        "distil-large-v2": {{"base": "mustafaaljadery/distil-whisper-mlx"}},
        "distil-large-v3": {{"base": "mustafaaljadery/distil-whisper-mlx"}},
    }}

    model_name = "{model}"
    cache_dir = Path("{cache_dir}")

    # Determine repo_id and model directory
    if model_name not in models:
        raise ValueError(f"Model {{model_name}} not supported")

    repo_id = models[model_name]["base"]
    model_dir = cache_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download model files to cache directory
    weights_file = model_dir / "weights.npz"
    config_file = model_dir / "config.json"

    if not weights_file.exists():
        hf_hub_download(repo_id=repo_id, filename="weights.npz", local_dir=str(model_dir))
    if not config_file.exists():
        hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=str(model_dir))

    # Transcribe the audio using the cached model
    result = transcribe_audio(
        audio="{input_wav_abs}",
        path_or_hf_repo=str(model_dir),
        language={repr(parsed_args.get("language"))},
        batch_size={batch_size},
        initial_prompt={repr(initial_prompt)}
    )

    # Print the result as JSON
    print(json.dumps(result, ensure_ascii=False))

except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    # Write the script to a temporary file
    script_file = output_dir / "transcribe_script.py"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    try:
        # Execute the script in the isolated environment
        if verbose:
            sys.stderr.write(f"Running lightning-whisper-mlx transcription on {input_wav_abs}\n")

        result = env.run([str(script_file)], shell=False, check=False, capture_output=True, text=True)

        # Check for errors and display stderr if there was a problem
        if result.returncode != 0:
            error_msg = f"lightning-whisper-mlx script failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr}"
            if result.stdout:
                error_msg += f"\nSTDOUT: {result.stdout}"
            raise RuntimeError(error_msg)

        # Parse the JSON output
        try:
            json_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse lightning-whisper-mlx output JSON: {e}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr}"
            if result.stdout:
                error_msg += f"\nSTDOUT: {result.stdout}"
            raise ValueError(error_msg)

        # Generate output files
        _generate_output_files(json_data, output_dir, initial_prompt)

    finally:
        # Clean up the temporary script
        if script_file.exists():
            script_file.unlink()


# Keep the old function name for backward compatibility
def run_whisper_mac_english(  # pylint: disable=too-many-arguments
    input_wav: Path,
    model: str,
    output_dir: Path,
    other_args: list[str] | None = None,
) -> None:
    """Legacy function for backward compatibility. Calls run_whisper_mac_mlx."""
    run_whisper_mac_mlx(
        input_wav=input_wav,
        model=model,
        output_dir=output_dir,
        language="en",
        task="transcribe",
        other_args=other_args,
    )
