"""
Runs whisper api with Apple MLX support using lightning-whisper-mlx.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import webvtt  # type: ignore
from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

HERE = Path(__file__).parent


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
    content_lines.append('  "lightning-whisper-mlx>=0.1.0",')
    content_lines.append('  "webvtt-py",')
    content_lines.append('  "numpy",')
    content_lines.append("]")
    content = "\n".join(content_lines)
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
            start_time = segment[0] * 0.02  # Convert seek to seconds (assuming 50fps)
            end_time = segment[1] * 0.02
            text = segment[2].strip()
        else:
            # Old format: dict with start/end/text
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
        text_content = text_content[len(initial_prompt.strip()):].strip()
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

    parsed_args = {}
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
    word_timestamps = parsed_args.get("word_timestamps", False)
    verbose = parsed_args.get("verbose", False)
    temperature = parsed_args.get("temperature", 0.0)

    # Get the environment and run transcription
    env = get_environment()

    # Create a Python script to run the transcription in the isolated environment
    script_content = f'''
import sys
import json
from pathlib import Path

try:
    from lightning_whisper_mlx import LightningWhisperMLX

    # Initialize the model
    whisper = LightningWhisperMLX(
        model="{model}",
        batch_size={batch_size},
        quant=None
    )

    # Transcribe the audio
    result = whisper.transcribe(
        audio_path="{input_wav_abs}",
        language={repr(parsed_args.get("language"))}
    )

    # Print the result as JSON
    print(json.dumps(result, ensure_ascii=False))

except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

    # Write the script to a temporary file
    script_file = output_dir / "transcribe_script.py"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    try:
        # Execute the script in the isolated environment
        if verbose:
            sys.stderr.write(f"Running lightning-whisper-mlx transcription on {input_wav_abs}\n")

        result = env.run([str(script_file)], shell=False, check=True, capture_output=True, text=True)

        # Parse the JSON output
        try:
            json_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse lightning-whisper-mlx output JSON: {e}")

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
