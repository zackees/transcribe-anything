# NVIDIA Detection Cache Fix

## Problem

On Windows development environments, tests were taking very long due to repeated downloads of the 2.2GB+ torch package. This was caused by:

1. **Inconsistent NVIDIA Detection**: The `has_nvidia_smi()` function was returning different values between runs
2. **Dynamic PyProject Generation**: Multiple modules generate different `pyproject.toml` content based on NVIDIA GPU detection
3. **uv-iso-env Behavior**: The `uv-iso-env` package performs a "nuke and pave" reinstall whenever the `pyproject.toml` fingerprint changes
4. **Repeated Downloads**: Each fingerprint change triggered a complete reinstall including the large torch download

## Root Cause

The issue was that `has_nvidia_smi()` was being called multiple times during test runs, and on Windows systems, the detection could be inconsistent due to:
- System state changes
- Process timing issues
- Environment variable changes
- Path resolution inconsistencies

This caused different `pyproject.toml` content to be generated between runs, changing the fingerprint and triggering reinstalls.

## Solution

### 1. NVIDIA Detection Caching

Enhanced `has_nvidia_smi()` in `src/transcribe_anything/util.py` to:
- Cache detection results based on system fingerprint
- Store cache in `~/.transcribe_anything_nvidia_cache.json`
- Use system information (platform, machine, version) + nvidia-smi existence as fingerprint
- Provide consistent results across runs for the same system configuration

### 2. Debug Logging

Added debug logging to environment generation functions:
- `src/transcribe_anything/whisper.py`
- `src/transcribe_anything/insanley_fast_whisper_reqs.py`
- `src/transcribe_anything/whisper_mac.py`

Each now logs the MD5 hash of generated `pyproject.toml` content to help track changes.

### 3. Cache Management

Added command-line option to clear cache when needed:
```bash
transcribe-anything --clear-nvidia-cache
```

### 4. Testing

Created comprehensive tests in `tests/test_nvidia_cache.py` to verify:
- Caching behavior works correctly
- Cache clearing functionality
- Different system fingerprints are handled properly

## Files Modified

- `src/transcribe_anything/util.py` - Enhanced NVIDIA detection with caching
- `src/transcribe_anything/whisper.py` - Added debug logging
- `src/transcribe_anything/insanley_fast_whisper_reqs.py` - Added debug logging  
- `src/transcribe_anything/whisper_mac.py` - Added debug logging
- `src/transcribe_anything/_cmd.py` - Added clear cache command-line option
- `tests/test_nvidia_cache.py` - New test file for cache functionality

## Usage

### Normal Operation
The caching is automatic and transparent. The first run will detect NVIDIA availability and cache the result. Subsequent runs will use the cached result, ensuring consistent `pyproject.toml` generation.

### Debugging
If you suspect caching issues, you can:

1. **View debug output**: The system will print debug messages showing:
   - Cached vs fresh NVIDIA detection results
   - PyProject.toml content hashes for each module

2. **Clear cache**: If hardware changes or you need to force re-detection:
   ```bash
   transcribe-anything --clear-nvidia-cache
   ```

### Expected Behavior
- **First run**: Detects NVIDIA, caches result, generates environment
- **Subsequent runs**: Uses cached result, generates identical environment
- **No more repeated downloads**: Same fingerprint = no reinstall needed

## Benefits

1. **Faster Testing**: Eliminates repeated 2.2GB+ torch downloads
2. **Consistent Behavior**: Same system configuration always produces same results
3. **Debuggable**: Clear logging shows what's happening
4. **Manageable**: Easy cache clearing when needed
5. **Backward Compatible**: No changes to existing API or behavior

## Technical Details

The cache file (`~/.transcribe_anything_nvidia_cache.json`) stores mappings from system fingerprints to detection results:

```json
{
  "Windows-AMD64-10.0.19041-nvidia_smi:true": true,
  "Linux-x86_64-5.4.0-nvidia_smi:false": false
}
```

The system fingerprint includes:
- Platform system (Windows, Linux, Darwin)
- Machine architecture (AMD64, x86_64, arm64)
- Platform version
- Whether nvidia-smi executable exists

This ensures that hardware or driver changes are properly detected while maintaining consistency for the same configuration.
