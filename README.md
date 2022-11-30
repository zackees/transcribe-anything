
# transcribe-anything

Input a local file or url and this service will transcribe it using Whisper AI

# Usage

```bash
> pip install transcribe-anything
> transcribe_anything <YOUTUBE_URL>
> transcribe_anything <LOCAL.MP4/MP3/WAV>
```

# Build Status

[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/MacOS_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_macos.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Win_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_win.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Ubuntu_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_ubuntu.yml)


## Install dev

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
  * `transcribe_anything <YOUTUBE_URL> > out_subtitles.txt`
  * -or- `transcribe_anything <MY_LOCAL.MP4/WAV> > out_subtitles.txt`


# Tech Stack
  * OpenAI whisper
  * yt-dlp: https://github.com/yt-dlp/yt-dlp
  * static-ffmpeg
    * github: https://github.com/zackees/static_ffmpeg
    * pypi: https://pypi.org/project/static-ffmpeg/

# Testing
  * All tests are run by `tox`, simply go to the project directory root and run it.

# Versions
  * 2.0.0: Now uses whisper AI as the backend.
  * 1.2.6: Supports spaces in file names now.
  * 1.2.5:
    * Improved handling of YouTube downloads by switching `youtube-dl` -> `yt-dlp`
