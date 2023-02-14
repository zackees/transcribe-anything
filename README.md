
# transcribe-anything
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/MacOS_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_macos.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Win_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_win.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Ubuntu_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_ubuntu.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Lint/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/lint.yml)

### USES WHISPER AI

Input a local file or url and this service will transcribe it using Whisper AI into subtitle files and raw text.

Uses whisper AI so this is state of the art translation service - completely free. 🤯🤯🤯

# Usage (CPU Version)

```bash
> pip install transcribe-anything
# Outputs the srt, vtt and txt files in title/out.vtt
> transcribe_anything https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

# Usage (GPU Accelerated Version)

```bash
> curl https://raw.githubusercontent.com/zackees/transcribe-anything/main/install_cuda.py | python
# Outputs the srt, vtt and txt files in title/out.vtt
> transcribe_anything https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

Will output:

```
Detecting language using up to the first 30 seconds. Use `--language` to specify the language
Detected language: English
[00:00.000 --> 00:27.000]  We're no strangers to love, you know the rules, and so do I
[00:27.000 --> 00:31.000]  I've built commitments while I'm thinking of
[00:31.000 --> 00:35.000]  You wouldn't get this from any other guy
[00:35.000 --> 00:40.000]  I just wanna tell you how I'm feeling
[00:40.000 --> 00:43.000]  Gotta make you understand
[00:43.000 --> 00:45.000]  Never gonna give you up
[00:45.000 --> 00:47.000]  Never gonna let you down
[00:47.000 --> 00:51.000]  Never gonna run around and desert you
[00:51.000 --> 00:53.000]  Never gonna make you cry
[00:53.000 --> 00:55.000]  Never gonna say goodbye
[00:55.000 --> 00:58.000]  Never gonna tell a lie
[00:58.000 --> 01:00.000]  And hurt you
[01:00.000 --> 01:04.000]  We've known each other for so long
[01:04.000 --> 01:09.000]  Your heart's been aching but you're too shy to say it
[01:09.000 --> 01:13.000]  Inside we both know what's been going on
[01:13.000 --> 01:17.000]  We know the game and we're gonna play it
[01:17.000 --> 01:22.000]  And if you ask me how I'm feeling
[01:22.000 --> 01:25.000]  Don't tell me you're too much to see
[01:25.000 --> 01:27.000]  Never gonna give you up
[01:27.000 --> 01:29.000]  Never gonna let you down
[01:29.000 --> 01:33.000]  Never gonna run around and desert you
[01:33.000 --> 01:35.000]  Never gonna make you cry
[01:35.000 --> 01:38.000]  Never gonna say goodbye
[01:38.000 --> 01:40.000]  Never gonna tell a lie
[01:40.000 --> 01:42.000]  And hurt you
[01:42.000 --> 01:44.000]  Never gonna give you up
[01:44.000 --> 01:46.000]  Never gonna let you down
[01:46.000 --> 01:50.000]  Never gonna run around and desert you
[01:50.000 --> 01:52.000]  Never gonna make you cry
[01:52.000 --> 01:54.000]  Never gonna say goodbye
[01:54.000 --> 01:57.000]  Never gonna tell a lie
[01:57.000 --> 01:59.000]  And hurt you
[02:08.000 --> 02:10.000]  Never gonna give
[02:12.000 --> 02:14.000]  Never gonna give
[02:16.000 --> 02:19.000]  We've known each other for so long
[02:19.000 --> 02:24.000]  Your heart's been aching but you're too shy to say it
[02:24.000 --> 02:28.000]  Inside we both know what's been going on
[02:28.000 --> 02:32.000]  We know the game and we're gonna play it
[02:32.000 --> 02:37.000]  I just wanna tell you how I'm feeling
[02:37.000 --> 02:40.000]  Gotta make you understand
[02:40.000 --> 02:42.000]  Never gonna give you up
[02:42.000 --> 02:44.000]  Never gonna let you down
[02:44.000 --> 02:48.000]  Never gonna run around and desert you
[02:48.000 --> 02:50.000]  Never gonna make you cry
[02:50.000 --> 02:53.000]  Never gonna say goodbye
[02:53.000 --> 02:55.000]  Never gonna tell a lie
[02:55.000 --> 02:57.000]  And hurt you
[02:57.000 --> 02:59.000]  Never gonna give you up
[02:59.000 --> 03:01.000]  Never gonna let you down
[03:01.000 --> 03:05.000]  Never gonna run around and desert you
[03:05.000 --> 03:08.000]  Never gonna make you cry
[03:08.000 --> 03:10.000]  Never gonna say goodbye
[03:10.000 --> 03:12.000]  Never gonna tell a lie
[03:12.000 --> 03:14.000]  And hurt you
[03:14.000 --> 03:16.000]  Never gonna give you up
[03:16.000 --> 03:23.000]  If you want, never gonna let you down Never gonna run around and desert you
[03:23.000 --> 03:28.000]  Never gonna make you hide Never gonna say goodbye
[03:28.000 --> 03:42.000]  Never gonna tell you I ain't ready
```

## Api

```python
from transcribe_anyting.api import transcribe

transcribe(
    url_or_file="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_dir="output_dir",
)
```

## Install GPU/CUDA Accelerated version

GPU acceleration is *much* faster than the CPU version. Install it using the following:

```bash
> curl https://raw.githubusercontent.com/zackees/transcribe-anything/main/install_cuda.py | python
# transcribe-anything should now be installed
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
  * 2.1.1: Updates keywords for easier pypi finding.
  * 2.1.0: Unknown args are now assumed to be for whisper and passed to it as-is. Fixes https://github.com/zackees/transcribe-anything/issues/3
  * 2.0.13: Now works with python 3.9
  * 2.0.12: Adds --device to argument parameters. This will default to CUDA if available, else CPU.
  * 2.0.11: Automatically deletes files in the out directory if they already exist.
  * 2.0.10: fixes local file issue https://github.com/zackees/transcribe-anything/issues/2
  * 2.0.9: fixes sanitization of path names for some youtube videos
  * 2.0.8: fix `--output_dir` not being respected.
  * 2.0.7: `install_cuda.sh` -> `install_cuda.py`
  * 2.0.6: Fixes twitter video fetching. --keep-audio -> --no-keep-audio
  * 2.0.5: Fix bad filename on trailing urls ending with /, adds --keep-audio
  * 2.0.3: GPU support is now added. Run the `install_cuda.sh` script to enable.
  * 2.0.2: Minor cleanup of file names (no more out.mp3.txt, it's now out.txt)
  * 2.0.1: Fixes missing dependencies and adds whisper option.
  * 2.0.0: New! Now a front end for Whisper ai!
