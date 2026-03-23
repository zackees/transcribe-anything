# Live Transcription Architecture

## Overview
This document outlines the architecture for adding live transcription support to transcribe-anything.

## Requirements (from FEATURE.md)
- Hot key ready
- Engages microphone or signals that permissions are needed
- Function is on standby with a semaphore waiting for audio

## Research Findings

### Audio Streaming Solutions
1. **WhisperLive** - Real-time implementation with faster_whisper, tensorrt, and openvino backends
2. **faster-whisper** - Optimized Whisper using CTranslate2 (already used in codebase)
3. **Silero VAD** - Voice Activity Detection for detecting speech segments
4. **sounddevice** - Cross-platform audio I/O library for microphone capture

### Hotkey Libraries
1. **keyboard** - Global hotkey support (Windows/Linux, experimental macOS)
2. **pynput** - Cross-platform hotkey listener
3. **global-hotkeys** - System-wide keybindings

## Proposed Architecture

### Components

#### 1. Microphone Capture Module (`live_audio.py`)
- Use `sounddevice` for cross-platform microphone access
- Implement permission checking and error handling
- Buffer audio chunks in real-time
- Sample rate: 16000 Hz (matching existing audio processing)

#### 2. Voice Activity Detection (`vad.py`)
- Integrate Silero VAD for detecting speech
- Detect silence periods to segment audio
- Configurable silence threshold and duration
- Prevents continuous transcription of background noise

#### 3. Live Transcription Engine (`live_transcribe.py`)
- Queue-based architecture with threading/asyncio
- Producer: Audio capture thread
- Consumer: Transcription thread
- Use existing faster-whisper backend for speed
- Support for all device types: cpu, cuda, insane, mlx

#### 4. Hotkey Handler (`hotkey.py`)
- Global hotkey listener (default: configurable, e.g., Ctrl+Shift+T)
- Start/stop transcription toggle
- Platform-specific implementation:
  - Windows/Linux: `keyboard` library
  - macOS: `pynput` (more reliable on macOS)
- Permission checks and error handling

#### 5. CLI Integration (`_cmd.py` updates)
- Add `--live` flag for live transcription mode
- Add `--hotkey` option to configure hotkey
- Add `--vad-threshold` for voice detection sensitivity
- Add `--continuous` flag for continuous vs. segment-based output

### Data Flow

```
[Hotkey Press]
    ↓
[Check Microphone Permissions]
    ↓
[Start Audio Capture Thread]
    ↓
[Microphone] → [Audio Buffer] → [VAD Detector]
                                      ↓
                               [Speech Detected?]
                                      ↓ Yes
                               [Audio Queue]
                                      ↓
                          [Transcription Thread]
                                      ↓
                          [Whisper Backend (faster-whisper)]
                                      ↓
                          [Output Text (stdout/file)]
```

### Implementation Strategy

#### Phase 1: Basic Live Transcription
1. Create `live_audio.py` with microphone capture
2. Create `vad.py` with Silero VAD integration
3. Create `live_transcribe.py` with queue-based processing
4. Integrate with existing faster-whisper backend
5. Add CLI support

#### Phase 2: Hotkey Support
1. Create `hotkey.py` with cross-platform support
2. Add permission checking
3. Integrate with live transcription engine
4. Add visual/audio feedback for start/stop

#### Phase 3: Polish & Testing
1. Add comprehensive error handling
2. Create unit tests
3. Add integration tests
4. Update documentation
5. Add examples

### Dependencies to Add

```toml
# Core dependencies
sounddevice>=0.4.6  # Audio capture
silero-vad>=4.0.0   # Voice activity detection
keyboard>=0.13.5    # Hotkey support (Windows/Linux)
pynput>=1.7.6       # Hotkey support (macOS fallback)

# Optional for better performance
webrtcvad>=2.0.10   # Alternative VAD
```

### Configuration

Add to config or CLI args:
```python
LiveTranscriptionConfig:
    hotkey: str = "ctrl+shift+t"
    vad_threshold: float = 0.5
    sample_rate: int = 16000
    chunk_duration_ms: int = 30  # VAD chunk size
    silence_duration_ms: int = 500  # Silence before segment end
    continuous: bool = False  # Continuous vs. segmented output
    output_format: str = "text"  # text, srt, json
```

### Platform Compatibility

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| Audio Capture | ✓ | ✓ | ✓ |
| VAD | ✓ | ✓ | ✓ |
| Hotkeys | ✓ (keyboard) | ✓ (keyboard, needs sudo) | ✓ (pynput) |
| GPU Acceleration | ✓ (CUDA) | ✓ (CUDA) | ✓ (MLX) |

### Challenges & Solutions

#### Challenge 1: Can we stream audio to transcribe-anything?
**Solution**: Yes, using a queue-based architecture with separate audio capture and transcription threads. Audio chunks are buffered and sent to the existing Whisper backends.

#### Challenge 2: Microphone Permissions
**Solution**:
- Check permissions before starting
- Provide clear error messages with platform-specific instructions
- Windows: Usually auto-prompt
- macOS: Requires microphone access in System Preferences
- Linux: Requires audio group membership

#### Challenge 3: Real-time Performance
**Solution**:
- Use faster-whisper backend by default
- Implement chunk-based processing (process while recording)
- Use VAD to avoid processing silence
- Support all existing device modes (cpu, cuda, insane, mlx)

#### Challenge 4: Hotkey Conflicts
**Solution**:
- Make hotkey configurable
- Detect conflicts and warn user
- Provide alternative activation methods (CLI flag to start immediately)

## Low-Latency Streaming Architecture

### Overview
For ultra-low latency transcription (targeting <500ms end-to-end latency), we need to optimize every stage of the pipeline and potentially add streaming-specific backends.

### Latency Sources & Optimizations

| Stage | Typical Latency | Low-Latency Optimization |
|-------|----------------|--------------------------|
| Audio Capture | 10-30ms | Use smallest buffer size that doesn't cause dropouts |
| VAD Detection | 30-100ms | Use WebRTC VAD (faster) or reduce Silero chunk size |
| Queue Transfer | 1-5ms | Use lock-free queues (queue.SimpleQueue) |
| Whisper Processing | 500-2000ms | Use streaming-aware model chunks |
| Output | 1-10ms | Direct stdout write, bypass formatting |

**Total Traditional**: ~600-2200ms
**Total Optimized**: ~200-800ms

### Low-Latency Architecture Changes

#### 1. Streaming Audio Buffer (`streaming_buffer.py`)

Replace simple queue with a **sliding window buffer**:

```python
class StreamingAudioBuffer:
    """
    Maintains a sliding window of audio with overlap for context.
    Enables streaming transcription without waiting for complete segments.
    """
    window_size_ms: int = 3000      # 3 second processing window
    stride_ms: int = 1500            # 1.5 second stride (50% overlap)
    min_chunk_ms: int = 500          # Minimum chunk before processing
    overlap_ms: int = 500            # Context overlap for continuity
```

**Benefits**:
- Start transcribing before speaker finishes
- Maintain context across chunks
- Reduce waiting time for segment completion

#### 2. Streaming Whisper Backend (`whisper_streaming.py`)

Implement a streaming-aware transcription layer:

```python
class StreamingWhisperBackend:
    """
    Wraps faster-whisper with streaming optimizations.
    """
    # Use beam_size=1 for lowest latency (greedy decoding)
    # Use smaller models (tiny, base, small) for speed
    # Enable prefix caching to reuse context
    # Use --condition_on_previous_text for continuity
```

**Key optimizations**:
- **Greedy decoding** (beam_size=1): 3-5x faster than beam search
- **Prefix caching**: Reuse computation from overlapping audio
- **Small models**: `tiny` (39M params) or `base` (74M params) for <200ms inference
- **No temperature**: Temperature=0 for deterministic, faster output

#### 3. Parallel Processing Pipeline (`parallel_pipeline.py`)

Move from sequential to **pipelined parallel processing**:

```
Traditional (Sequential):
[Capture] → wait → [VAD] → wait → [Transcribe] → wait → [Output]
Total: Sum of all stages

Low-Latency (Pipelined):
[Capture] → [Buffer 1] ↘
                         [VAD] → [Queue] ↘
[Capture] → [Buffer 2] ↗                  [Transcribe] → [Output]
                                          ↗
                         [VAD] → [Queue] ↗

Total: Max of any single stage (+ small queue delays)
```

**Implementation**:
- 3 parallel threads/processes:
  1. Audio capture (continuous)
  2. VAD processing (processes chunks as they arrive)
  3. Transcription (processes validated speech chunks)
- Use `multiprocessing.Queue` for inter-process communication (avoids GIL)
- Pre-warm Whisper model to avoid cold-start latency

#### 4. Low-Latency Data Flow

```
[Microphone - 10ms chunks]
    ↓ (continuous stream)
[Ring Buffer - 100ms capacity] ← Lock-free, minimal latency
    ↓ (every 30ms)
[WebRTC VAD - 10ms processing] ← Fastest VAD option
    ↓ (on speech detection)
[Sliding Window Buffer - 3s window, 1.5s stride]
    ↓ (when min_chunk reached OR speaker pauses)
[Transcription Process Pool (3 workers)] ← Parallel processing
    ↓ (immediate)
[De-duplication Filter] ← Remove overlap redundancy
    ↓ (immediate)
[Output Stream (stdout/callback)] ← Real-time display
```

#### 5. Advanced Features for Low-Latency

##### A. Adaptive Chunk Sizing
```python
class AdaptiveChunker:
    """
    Dynamically adjust chunk size based on:
    - Speech rate (faster speech = larger chunks)
    - Pause detection (natural boundaries = chunk)
    - Processing speed (if falling behind = smaller chunks)
    """
```

##### B. Speculative Decoding
```python
class SpeculativeDecoder:
    """
    Start transcribing with tiny model immediately (50ms latency)
    Refine with larger model in background (200ms latency)
    Display fast result, update if refined version differs
    """
```

##### C. Context Caching
```python
class ContextCache:
    """
    Cache whisper encoder outputs for overlapping audio
    Reduces redundant computation by 30-50%
    """
    cache_size: int = 10  # Keep last 10 encoder states
```

### Configuration for Low-Latency Mode

```python
LowLatencyConfig:
    # Audio settings
    buffer_size_ms: int = 10           # Smallest stable buffer
    sample_rate: int = 16000           # Standard whisper rate

    # VAD settings
    vad_engine: str = "webrtc"         # Fastest VAD
    vad_aggressiveness: int = 3        # Max sensitivity

    # Streaming settings
    window_size_ms: int = 3000         # Processing window
    stride_ms: int = 1500              # 50% overlap
    min_chunk_ms: int = 500            # Start processing early

    # Transcription settings
    model: str = "base"                # Fast small model
    beam_size: int = 1                 # Greedy decoding
    best_of: int = 1                   # Single pass
    temperature: float = 0.0           # Deterministic
    condition_on_previous_text: bool = True  # Continuity

    # Performance settings
    num_workers: int = 3               # Parallel transcription
    use_prefix_cache: bool = True      # Reuse computation
    enable_speculative: bool = False   # Optional fast preview
```

### Latency Comparison

| Mode | Model | Latency | Accuracy | Use Case |
|------|-------|---------|----------|----------|
| **Ultra-Low** | tiny + greedy | 200-400ms | 85% | Live captions, gaming |
| **Low** | base + greedy | 300-600ms | 90% | Meetings, dictation |
| **Balanced** | small + beam=3 | 500-1000ms | 93% | General use |
| **High Quality** | large-v3 + beam=5 | 1000-2000ms | 96% | Archival, accuracy-critical |

### Implementation Priority

#### Phase 1: Basic Streaming (Target: <1000ms)
1. Implement sliding window buffer
2. Add WebRTC VAD option
3. Optimize Whisper parameters (beam_size=1, small model)
4. Pipelined processing threads

#### Phase 2: Advanced Optimization (Target: <500ms)
1. Implement parallel transcription workers
2. Add prefix caching
3. Optimize queue transfers (lock-free queues)
4. Pre-warm model loading

#### Phase 3: Ultra-Low Latency (Target: <300ms)
1. Implement speculative decoding
2. Add adaptive chunk sizing
3. Optimize for specific hardware (CUDA streams, MLX optimizations)
4. Profile and eliminate bottlenecks

### Benchmarking & Monitoring

Add real-time latency monitoring:

```python
class LatencyMonitor:
    """Track latency at each pipeline stage"""
    metrics = {
        "capture_latency": [],
        "vad_latency": [],
        "queue_latency": [],
        "transcribe_latency": [],
        "total_latency": []
    }

    def report_percentiles(self):
        """Report p50, p95, p99 latencies"""
```

### Hardware Recommendations

| Hardware | Expected Latency | Recommended Config |
|----------|------------------|-------------------|
| **CPU Only** | 800-2000ms | model=tiny, workers=1 |
| **NVIDIA GTX 1060+** | 400-800ms | model=base, workers=2, device=cuda |
| **NVIDIA RTX 3060+** | 200-500ms | model=small, workers=3, device=insane |
| **Apple M1/M2** | 300-600ms | model=base, workers=2, device=mlx |
| **Apple M3/M4** | 200-400ms | model=small, workers=3, device=mlx |

### Example: Low-Latency Usage

```bash
# Ultra-low latency mode (prioritize speed)
transcribe-anything --live --low-latency ultra \
  --model tiny --vad webrtc --workers 3

# Balanced low-latency mode
transcribe-anything --live --low-latency balanced \
  --model base --window-size 3000 --stride 1500

# Custom low-latency configuration
transcribe-anything --live \
  --model base --beam-size 1 --vad webrtc \
  --window-size 2000 --stride 1000 --workers 2 \
  --enable-prefix-cache
```

### API: Low-Latency Streaming

```python
from transcribe_anything import StreamingTranscriber

# Low-latency streaming transcriber
transcriber = StreamingTranscriber(
    model="base",
    device="cuda",
    latency_mode="low",  # ultra, low, balanced, quality
    callback=lambda text: print(f">> {text}", end="", flush=True)
)

# Start streaming
transcriber.start()

# Get real-time metrics
stats = transcriber.get_latency_stats()
print(f"Average latency: {stats['avg_latency_ms']}ms")
print(f"P95 latency: {stats['p95_latency_ms']}ms")
```

### Output Formats

1. **Continuous Text** (stdout): Real-time text output as it's transcribed
2. **Segmented Text**: Text with timestamps for each speech segment
3. **SRT File**: Traditional subtitle format with timestamps
4. **JSON**: Structured output with metadata

### Example Usage

```bash
# Start live transcription with default hotkey (Ctrl+Shift+T)
transcribe-anything --live

# Start immediately without hotkey
transcribe-anything --live --start-immediately

# Custom hotkey and output file
transcribe-anything --live --hotkey "ctrl+alt+r" --output live_transcript.txt

# With specific device
transcribe-anything --live --device insane --model large-v3

# With custom VAD settings
transcribe-anything --live --vad-threshold 0.6 --silence-duration 1000
```

### API Usage

```python
from transcribe_anything import LiveTranscriber

# Create transcriber
transcriber = LiveTranscriber(
    model="base",
    device="cuda",
    hotkey="ctrl+shift+t",
    output_file="transcript.txt"
)

# Start with hotkey
transcriber.start_with_hotkey()

# Or start immediately
transcriber.start()

# Stop
transcriber.stop()
```

## Next Steps

1. Implement basic audio capture and VAD
2. Integrate with existing Whisper backends
3. Add hotkey support
4. Create comprehensive tests
5. Update documentation
