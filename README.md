
# transcribe-anything
Input a local file or url and this service will transcribe it using Mozilla Deepspeech 0.9.3.

# Build Status

[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/MacOS_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_macos.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Win_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_win.yml)
[![Actions Status](https://github.com/zackees/transcribe-anything/workflows/Ubuntu_Tests/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/push_ubuntu.yml)


# Example

  * Example (cmd):
    * `transcribe_anything <YOUTUBE_URL> > out_subtitles.txt`
    * `transcribe_anything <LOCAL.MP4/WAV> > out_subtitles.txt`
  * Example (api):
    ```
    from transcribe_anything.api import bulk_transcribe

    urls = ['https://www.youtube.com/watch?v=Erk4_jFDjzQ']
    def onresolve(url, sub): print(url, sub)
    def onfail(url): print(f'Failed: {url}')
    bulk_transcribe(urls, onresolve=onresolve, onfail=onfail)
    ```

# Quick start

## Optional: Create a virtual python package
  * Works for Ubuntu/MacOS/Win32
  * `mkdir transcribe_anything`
  * `cd transcribe_anything`
  * Download and install virtual env:
    * `curl -X GET https://raw.githubusercontent.com/zackees/make_venv/main/make_venv.py -o make_env.py`
    * `python make_env.py`
  * Enter the environment:
    * `source activate.sh`

The environment is now active and the next step will only install to the local python. If the terminal
is closed then to get back into the environment `cd transcribe_anything` and execute `source activate.sh`

## Required: Install to current python environment
  * `pip install transcribe-anything`
    * The command `transcribe_anything` will magically become available.
  * `transcribe_anything <YOUTUBE_URL> > out_subtitles.txt`
  * -or- `transcribe_anything <MY_LOCAL.MP4/WAV> > out_subtitles.txt`

# How does it work?

This program performs fetching using `yt-dlp` for downloading videos from video services, and then
stripping the audio track out.

[static_ffmpeg](https://pypi.org/project/static-ffmpeg/) is then called to transcode the audio track into a specific format that DeepSpeech requires.

Once the audio file has been prepared, [pydeepspeech](https://pypi.org/project/pydeepspeech/) is called. This little
utility automatically downloads the proper AI models and installs them into the proper path so that deepspeech can be
called. It also partitions the input wav file into chunks, split at the parts of silence, in order to make processing
go easier (DeepSpeech degrades performance significantly with longer audio clips, so they have to be kept short.)



# Tech Stack
  * Mozilla DeepSpeech: https://github.com/mozilla/DeepSpeech
  * pydeepspeech: https://github.com/zackees/pydeepspeech
    * mic_vad_streaming: https://github.com/hadran9/DeepSpeech-examples/tree/r0.9/mic_vad_streaming
  * yt-dlp: https://github.com/yt-dlp/yt-dlp
  * static-ffmpeg
    * github: https://github.com/zackees/static_ffmpeg
    * pypi: https://pypi.org/project/static-ffmpeg/

# Testing
  * All tests are run by `tox`, simply go to the project directory root and run it.

# Versions
  * 1.2.5:
    * Improved handling of YouTube downloads by switching `youtube-dl` -> `yt-dlp`
