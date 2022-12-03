
# transcribe-anything

Input a local file or url and this service will transcribe it using Whisper AI into subtitle files and raw text.

Uses whisper AI so this is state of the art translation service - completely free. 🤯🤯🤯

If you install from PYPI then by default it will install the CPU version only.
To enable the GPU version, you *must install by cloning the github and running `install_cuda.sh` script, due to security lockdowns that can't be bypassed (pypi only allows dependencies within the ecosystem). Please note that whatever torch version you have will be purged with `install_cuda.sh` script.

# Usage

```bash
> pip install transcribe-anything
> transcribe_anything <YOUTUBE_URL>
# Outputs the srt, vtt and txt files in YOUTUBE_URL/out.vtt
> transcribe_anything <LOCAL.MP4/MP3/WAV>
# Same but in LOCAL/out.vtt ...
```

# Build Status

[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/MacOS_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_macos.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Win_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_win.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Ubuntu_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_ubuntu.yml)

## Install CPU Version

If you want GPU acceleration then you need special installation instructions:

```bash
> git clone https://github.com/zackees/transcribe-anything
> cd transcribe_anything
> ./install_cuda.sh  # will uninstall torch and replace with torch+cuda 1.12.0
# Should now be installed
> transcribe_anything https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

## Develop

Works for Ubuntu/MacOS/Win32(in git-bash)
This will create a virtual environment

```bash
> cd transcribe_anything
> ./install_dev.sh
# Enter the environment:
> source activate.sh
```

The environment is now active and the next step will only install to the local python. If the terminal
is closed then to get back into the environment `cd transcribe_anything` and execute `source activate.sh`

## Required: Install to current python environment
  * `pip install transcribe-anything`
    * The command `transcribe_anything` will magically become available.
  * `transcribe_anything <YOUTUBE_URL>`


# Tech Stack
  * OpenAI whisper
  * yt-dlp: https://github.com/yt-dlp/yt-dlp
  * static-ffmpeg
    * github: https://github.com/zackees/static_ffmpeg
    * pypi: https://pypi.org/project/static-ffmpeg/

# Testing
  * All tests are run by `tox`, simply go to the project directory root and run it.

# Versions
  * 2.0.4: Fix bad filename on trailing urls ending with /, adds --keep-audio
  * 2.0.3: GPU support is now added. Run the `install_cuda.sh` script to enable.
  * 2.0.2: Minor cleanup of file names (no more out.mp3.txt, it's now out.txt)
  * 2.0.1: Fixes missing dependencies and adds whisper option.
  * 2.0.0: New! Now a front end for Whisper ai!
