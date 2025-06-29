# AI_README.md - transcribe-anything Project Documentation

This document provides a comprehensive technical analysis of the `transcribe-anything` project for AI agents that need to understand, modify, or extend this codebase.

## Project Overview

**transcribe-anything** is a Python CLI tool and library that provides a unified interface for transcribing audio/video content using multiple Whisper AI backends. It's designed for ease of use while supporting advanced features like GPU acceleration, speaker diarization, and multi-platform optimization.

### Key Features
- Multiple Whisper backends: CPU, CUDA, "insane" (GPU-accelerated), and MLX (Mac Apple Silicon)
- Speaker diarization with `speaker.json` output
- Support for local files and online URLs (YouTube, Rumble, etc.)
- Subtitle embedding into video files
- Custom vocabulary support via initial prompts
- Docker containerization with GPU support
- Isolated environment management for dependencies

## Architecture Overview

### Core Components

```
src/transcribe_anything/
├── __init__.py           # Main API export
├── __main__.py          # Module entry point
├── _cmd.py              # CLI interface and argument parsing
├── api.py               # Main transcription API
├── whisper.py           # Standard OpenAI Whisper backend
├── insanely_fast_whisper.py  # GPU-accelerated backend
├── whisper_mac.py       # Apple Silicon MLX backend
├── audio.py             # Audio processing and conversion
├── util.py              # Utility functions and system detection
├── logger.py            # Logging utilities
└── [other modules...]   # Specialized functionality
```

### Backend Architecture

The project uses an **isolated environment pattern** via `uv-iso-env` to manage complex AI dependencies:

1. **Parent Environment**: Contains the main CLI and coordination logic
2. **Backend Environments**: Isolated Python environments for each Whisper backend
3. **Communication**: Subprocess-based IPC between parent and backend environments

## Backends Deep Dive

### 1. Standard Whisper Backend (`whisper.py`)
- **Environment**: `src/transcribe_anything/venv/whisper/`
- **Dependencies**: `openai-whisper==20240930`, `torch`, `numpy`
- **Use Case**: CPU processing, universal compatibility
- **Key Features**: Full OpenAI Whisper argument support

### 2. Insanely Fast Whisper Backend (`insanely_fast_whisper.py`)
- **Environment**: Managed by `insanley_fast_whisper_reqs.py`
- **Dependencies**: `insanely-fast-whisper`, PyTorch with CUDA
- **Use Case**: GPU acceleration, speaker diarization
- **Key Features**: 
  - Batch processing for speed
  - HuggingFace token integration for speaker diarization
  - Flash Attention 2 support
  - Generates `speaker.json` files

### 3. MLX Backend (`whisper_mac.py`)
- **Environment**: `src/transcribe_anything/venv/whisper_mlx/`
- **Dependencies**: `lightning-whisper-mlx` (via Git)
- **Use Case**: Apple Silicon optimization
- **Key Features**: 
  - 4x faster than MPS backend
  - Multi-language support
  - Custom vocabulary via initial prompts

## Key Files and Their Purposes

### Entry Points
- `_cmd.py`: Complete CLI argument parsing, validation, and coordination
- `api.py`: Python API with `transcribe()` function
- `__main__.py`: Module execution entry point

### Backend Modules
- `whisper.py`: Standard Whisper with CUDA detection
- `insanely_fast_whisper.py`: High-performance GPU backend
- `whisper_mac.py`: Apple Silicon optimized backend
- `cuda_available.py`: GPU capability detection

### Utility Modules
- `audio.py`: Audio fetching and conversion (yt-dlp + ffmpeg)
- `util.py`: System detection, filename sanitization, NVIDIA caching
- `generate_speaker_json.py`: Speaker diarization post-processing
- `srt_wrap.py`: Subtitle formatting utilities

### Configuration
- `WHISPER_OPTIONS.json`: Complete Whisper parameter definitions
- `pyproject.toml`: Main project configuration and dependencies

## Key Design Patterns

### 1. Device Detection and Selection
```python
# From util.py - Cached NVIDIA detection
def has_nvidia_smi() -> bool:
    # Uses system fingerprinting and caching to avoid repeated detection
    
# From api.py - Device enum pattern
class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda" 
    INSANE = "insane"
    MLX = "mlx"
```

### 2. Isolated Environment Management
```python
# Each backend creates its own pyproject.toml and isolated environment
def get_environment() -> IsoEnv:
    venv_dir = HERE / "venv" / "backend_name"
    # Dynamic pyproject.toml generation based on system capabilities
```

### 3. Subprocess Communication Pattern
```python
# All backends use subprocess communication
proc = env.open_proc(cmd_list, shell=False)
# Wait for completion and handle errors
```

## Testing Strategy

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: Full transcription pipeline testing
- **Backend-Specific Tests**: Each backend has dedicated test files
- **Test Data**: `tests/localfile/` contains test audio/video files

### Key Test Files
- `test_transcribe_anything.py`: Main API testing
- `test_insanely_fast_whisper.py`: GPU backend testing
- `test_insanely_fast_whisper_mlx.py`: Apple Silicon testing
- `test_nvidia_fix.py`: Hardware detection consistency
- `test_initial_prompt.py`: Custom vocabulary testing

### Running Tests
```bash
bash test  # Uses pytest with verbose output
```

## Common Modification Patterns

### Adding a New Backend

1. **Create Backend Module**
   ```python
   # src/transcribe_anything/my_backend.py
   def get_environment() -> IsoEnv:
       # Define dependencies in pyproject.toml format
   
   def run_my_backend(input_wav: Path, model: str, output_dir: Path, ...):
       # Implement transcription logic
   ```

2. **Update Device Enum**
   ```python
   # In api.py
   class Device(Enum):
       MY_BACKEND = "my_backend"
   ```

3. **Add CLI Support**
   ```python
   # In _cmd.py parse_arguments()
   choices = [None, "cpu", "cuda", "insane", "my_backend"]
   ```

4. **Integrate in API**
   ```python
   # In api.py transcribe()
   elif device_enum == Device.MY_BACKEND:
       run_my_backend(...)
   ```

### Adding New Arguments

1. **Update CLI Parser** (`_cmd.py`)
2. **Update API Function** (`api.py`)
3. **Pass to Backend** (via `other_args` or explicit parameters)
4. **Update Tests**

### Modifying Audio Processing

- **Input Processing**: Modify `audio.py` `fetch_audio()`
- **Format Conversion**: Update `_convert_to_wav()`
- **URL Handling**: Extend `ytldp_download.py`

## Dependencies and Environment Management

### Main Dependencies
```toml
# From pyproject.toml
dependencies = [
    "static-ffmpeg>=2.7",      # Audio/video processing
    "yt-dlp>=2025.1.15",       # URL downloading
    "appdirs>=1.4.4",          # Cross-platform directories
    "disklru>=1.0.7",          # Disk caching
    "FileLock",                # File system synchronization
    "webvtt-py==0.4.6",        # Subtitle format conversion
    "uv-iso-env>=1.0.43",      # Isolated environments
    "python-dotenv>=1.0.1",    # Environment variables
]
```

### Backend-Specific Dependencies
Each backend dynamically generates its `pyproject.toml` based on:
- System platform detection
- GPU availability
- Model requirements

## Error Handling Patterns

### Common Error Types
1. **Environment Setup Errors**: Backend installation failures
2. **Hardware Detection Errors**: GPU/device detection issues
3. **Audio Processing Errors**: Format conversion or download failures
4. **Model Loading Errors**: Backend-specific model issues

### Error Recovery Strategies
- **Fallback Backends**: CPU fallback when GPU unavailable
- **Retry Logic**: For network operations and model downloads
- **Caching**: NVIDIA detection caching to avoid repeated failures
- **Graceful Degradation**: Disable features when dependencies unavailable

## Performance Considerations

### Backend Performance Characteristics
- **CPU**: Slowest, most compatible
- **CUDA**: Medium speed, requires NVIDIA GPU
- **Insane**: Fastest GPU mode, high memory usage
- **MLX**: Optimized for Apple Silicon

### Memory Management
- **Batch Size Tuning**: Critical for GPU backends
- **Temporary File Cleanup**: Automatic cleanup via `atexit`
- **Model Caching**: Persistent model storage across runs

## Security Considerations

### Input Validation
- **URL Sanitization**: In `ytldp_download.py`
- **Filename Sanitization**: In `util.py` `sanitize_filename()`
- **Path Validation**: Absolute path handling in backends

### Token Management
- **HuggingFace Tokens**: Cached securely for speaker diarization
- **Environment Variables**: Support for `HF_TOKEN` env var

## Docker and Containerization

### Docker Architecture
- **Base Image**: `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`
- **GPU Support**: NVIDIA Container Toolkit integration
- **Optimization**: Pre-initialization of GPU dependencies
- **Entry Point**: `entrypoint.sh` for runtime configuration

## File Output Formats

### Standard Outputs
- **`out.txt`**: Plain text transcription
- **`out.srt`**: SubRip subtitle format
- **`out.vtt`**: WebVTT subtitle format  
- **`out.json`**: Full transcription metadata
- **`speaker.json`**: Speaker diarization results (when available)

### Format Conversions
- JSON → SRT: `convert_json_to_srt()`
- SRT → VTT: `convert_to_webvtt()`
- Speaker segmentation: `generate_speaker_json()`

## Debugging and Development

### Debug Information
- **Verbose Logging**: Available in most backends
- **Cache Inspection**: NVIDIA detection cache at `~/.transcribe_anything_nvidia_cache.json`
- **Environment Debugging**: Each backend environment can be inspected individually

### Development Workflow
```bash
./install     # Install development dependencies  
./lint        # Run code quality checks
./test        # Run full test suite
./clean       # Clean up generated files
```

## Extending the Project

### Common Extension Points
1. **New Audio Sources**: Extend `ytldp_download.py`
2. **Additional Output Formats**: Add converters in utility modules
3. **Custom Post-Processing**: Extend speaker diarization logic
4. **UI Interfaces**: Build on top of `api.py`
5. **Cloud Integration**: Add cloud storage backends

### Integration Patterns
- **Library Usage**: Import `transcribe_anything` and use `transcribe()` function
- **CLI Extension**: Add new arguments and backend support
- **Batch Processing**: Use API in loops with different configurations
- **Pipeline Integration**: Incorporate into larger media processing workflows

## Maintenance Notes

### Version Management
- **Manual Versioning**: Version in `pyproject.toml` updated manually
- **Backend Dependencies**: Pin specific versions for stability
- **Model Compatibility**: Backend-specific model version management

### Testing Requirements
- **Cross-Platform**: Tests must work on Windows, macOS, Linux
- **Hardware-Specific**: GPU tests conditional on hardware availability
- **Isolation**: Each test should clean up after itself

This documentation should provide AI agents with comprehensive understanding needed to successfully modify, extend, or debug the transcribe-anything codebase. 