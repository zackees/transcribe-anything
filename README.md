# transcribe-anything
[![MacOS_Tests](https://github.com/zackees/transcribe-anything/actions/workflows/test_macos.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_macos.yml) 
[![Win_Tests](https://github.com/zackees/transcribe-anything/actions/workflows/test_win.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_win.yml)
[![Ubuntu_Tests](https://github.com/zackees/transcribe-anything/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/test_ubuntu.yml)
[![Lint](https://github.com/zackees/transcribe-anything/actions/workflows/lint.yml/badge.svg)](https://github.com/zackees/transcribe-anything/actions/workflows/lint.yml)

![image](https://github.com/zackees/transcribe-anything/assets/6856673/94bdd1fe-3225-438a-ac1b-09c81f1d4108)



### USES WHISPER AI

Over 600+⭐'s because this program this app just works! This whisper front-end app is the only one to generate a `speaker.json` file which partitions the conversation by who doing the speaking.

[![Star History Chart](https://api.star-history.com/svg?repos=zackees/transcribe-anything&type=Date)](https://star-history.com/#zackees/transcribe-anything&Date)


### New in 3.0!

Mac acceleration option using the new [whisper-mps](https://github.com/AtomGradient/whisper-mps) backend. Enable with `--device mps`. English only, and does not support the `speaker.json` output, but is quite fast.

## About

Easiest whisper implementation to install and use. Just install with `pip install transcribe-anything`. All whisper backends are executed in an isolated environment. GPU acceleration is *automatic*, using the *blazingly* fast [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) as the backend for `--device insane`. This is the only tool to optionally produces a `speaker.json` file, representing speaker-assigned text that has been de-chunkified.

Hardware acceleration on Windows/Linux `--device insane`

MacArm acceleration when using `--device mps`

Input a local file or youtube/rumble url and this tool will transcribe it using Whisper AI into subtitle files and raw text.

Uses whisper AI so this is state of the art translation service - completely free. 🤯🤯🤯

Your data stays private and is not uploaded to any service.

The new version now has state of the art speed in transcriptions, thanks to the new backend `--device insane`, as well as producing a `speaker.json` file.



```bash
pip install transcribe-anything
# slow cpu mode, works everywhere
transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ
# insanely fast using the insanely-fast-whisper backend.
transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ --device insane
# translate from any language to english
transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ --device insane --task translate
# Mac accelerated back-end
transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ --device mps
```

*python api*
```python
from transcribe_anything import transcribe_anything

transcribe_anything(
    url_or_file="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_dir="output_dir",
    task="transcribe",
    model="large",
    device="cuda"
)

# Full function signiture:
def transcribe(
    url_or_file: str,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,              # tiny,small,medium,large
    task: Optional[str] = None,               # transcribe or translate
    language: Optional[str] = None,           # auto detected if none, "en" for english...
    device: Optional[str] = None,             # cuda,cpu,insane,mps
    embed: bool = False,                      # Produces a video.mp4 with the subtitles burned in.
    hugging_face_token: Optional[str] = None, # If you want a speaker.json
    other_args: Optional[list[str]] = None,   # Other args to be passed to to the whisper backend
) -> str:

```

#### Insanely fast on `cuda` platforms

If you pass in `--device insane` on a cuda platform then this tool will use this state of the art version of whisper: https://github.com/Vaibhavs10/insanely-fast-whisper, which is MUCH faster and has a pipeline for speaker identification (diarization) using the `--hf_token` option.

Compatible with Python 3.10 and above. Backends use an isolated environment with pinned requirements and python version.

#### Speaker.json

When diarization is enabled via `--hf_token` (hugging face token) then the output json will contain speaker info labeled as `SPEAKER_00`, `SPEAKER_01` etc. For licensing agreement reasons, you must get your own hugging face token if you want to enable this feature. Also there is an additional step to agree to the user policies for the `pyannote.audio` located here: https://huggingface.co/pyannote/segmentation-3.0. If you don't do this then you'll see runtime exceptions from `pyannote` when the `--hf_token` is used.

What's special to this app is that we also generate a `speaker.json` which is a de-chunkified version of the output json speaker section.

### speaker.json example:
```json
[
  {
    "speaker": "SPEAKER_00",
    "timestamp": [
      0.0,
      7.44
    ],
    "text": "for that. But welcome, Zach Vorhees. Great to have you back on. Thank you, Matt. Craving me back onto your show. Man, we got a lot to talk about.",
    "reason": "beginning"
  },
  {
    "speaker": "SPEAKER_01",
    "timestamp": [
      7.44,
      33.52
    ],
    "text": "Oh, we do. 2023 was the year that OpenAI released, you know, chat GPT-4, which I think most people would say has surpassed average human intelligence, at least in test taking, perhaps not in, you know, reasoning and things like that. But it was a major year for AI. I think that most people are behind the curve on this. What's your take of what just happened in the last 12 months and what it means for the future of human cognition versus machine cognition?",
    "reason": "speaker-switch"
  },
  {
    "speaker": "SPEAKER_00",
    "timestamp": [
      33.52,
      44.08
    ],
    "text": "Yeah. Well, you know, at the beginning of 2023, we had a pretty weak AI system, which was a chat GPT 3.5 turbo was the best that we had. And then between the beginning of last",
    "reason": "speaker-switch"
  }
]
```

Note that `speaker.json `is only generated when using `--device insane` and not for `--device cuda` nor `--device cpu`.

#### `cuda` vs `insane`

Insane mode eats up a lot of memory and it's common to get out of memory errors while transcribing. For example a 3060 12GB nividia card produced out of memory errors are common for big content. If you experience this then pass in `--batch-size 8` or smaller. Note that any arguments not recognized by `transcribe-anything` are passed onto the backend transcriber.

Also, please don't use `distil-whisper/distil-large-v2`, it produces extremely bad stuttering and it's not entirely clear why this is. I've had to switch it out of production environments because it's so bad. It's also non-deterministic so I think that somehow a fallback non-zero temperature is being used, which produces these stutterings.

`cuda` is the original AI model supplied by openai. It's more stable but MUCH slower. It also won't produce a `speaker.json` file which looks like this:


`--embed`. This app will optionally embed subtitles directly "burned" into an output video.

# Install

This front end app for whisper boasts the easiest install in the whisper ecosystem thanks to [isolated-environment](https://pypi.org/project/isolated-environment/). You can simply install it with pip, like this:

```bash
pip install transcribe-anything
```

# GPU Acceleration

GPU acceleration will be automatically enabled for windows and linux. Mac users are stuck with `--device cpu` mode. But it's possible that `--device insane` and `--model mps` on Mac M1+ will work, but this has been completely untested.

Windows/Linux:
  * Use `--device insane`

Mac:
  * Use `--device mps`

# Usage

```bash
 transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ
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
from transcribe_anything.api import transcribe

transcribe(
    url_or_file="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_dir="output_dir",
)
```

## Develop

Works for Ubuntu/MacOS/Win32(in git-bash)
This will create a virtual environment

```bash
> cd transcribe_anything
> ./install.sh
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
  * insanely-fast-whisper
  * yt-dlp: https://github.com/yt-dlp/yt-dlp
  * static-ffmpeg
    * github: https://github.com/zackees/static_ffmpeg
    * pypi: https://pypi.org/project/static-ffmpeg/

# Testing

  * Every commit is tested for standard linters and a batch of unit tests.

## Updated version 2.3.0

`transcribe-anything` now works much better across different configurations and is now much faster. Why? I switched the environment isolation that I was using from my own homespun version built on top of `venv` to the AMAZING `uv` system. The biggest improvement is the runtime speed and re-installs. UV is just insane at how fast it is for checking the environment. Also it turns out that `uv` has  strict package dependency checking which found a minor bug where a certain version of one of the `pytorch` dependencies was being constantly re-installed because of a dependency conflict that pip was apparently perfectly happy to never warn about. This manifested as certain packages being constantly re-installed with the previous version. `uv` identified this as an error immediately and was fixed.

The real reason behind `transcribe-anything`'s surprising popularity comes from the fact that it just works. And the reason for this is that I can isolate environments for different configurations and install them lazily. If you have the same problem then consider my other tool: https://github.com/zackees/iso-env


# Versions
  * 3.0.0: Implemented new Mac-arm accelerated [whisper-mps](https://github.com/AtomGradient/whisper-mps) backend, enable with `--device mps`. Only does english, but is quite fast.
  * 2.3.0: Swapped out the environment isolator. Now based on `uv`, should fix the missing dll's on some windows systems.
  * 2.7.39: Fix `--hf-token` usage for insanely fast whisper backend.
  * 2.7.37: Fixed breakage due to numpy 2.0 being released.
  * 2.7.36: Fixed some ffmpeg dependencies.
  * 2.7.35: All `ffmpeg` commands are now `static_ffmpeg` commands. Fixes issue.
  * 2.7.34: Various fixes.
  * 2.7.33: Fixes linux
  * 2.7.32: Fixes mac m1 and m2.
  * 2.7.31: Adds a warning if using python 3.12, which isn't supported yet in the backend.
  * 2.7.30: adds --query-gpu-json-path
  * 2.7.29: Made to json -> srt more robust for `--device insane`, bad entries will be skipped but warn.
  * 2.7.28: Fixes bad title fetching with weird characters.
  * 2.7.27: `pytorch-audio` upgrades broke this package. Upgrade to latest version to resolve.
  * 2.7.26: Add model option `distil-whisper/distil-large-v2`
  * 2.7.25: Windows (Linux/MacOS) bug with `--device insane` and python 3.11 installing wrong `insanely-fast-whisper` version.
  * 2.7.22: Fixes `transcribe-anything` on Linux.
  * 2.7.21: Tested that Mac Arm can run `--device insane`. Added tests to ensure this.
  * 2.7.20: Fixes wrong type being returned when speaker.json happens to be empty.
  * 2.7.19: speaker.json is now in plain json format instead of json5 format
  * 2.7.18: Fixes tests
  * 2.7.17: Fixes speaker.json nesting.
  * 2.7.16: Adds `--save_hf_token`
  * 2.7.15: Fixes 2.7.14 breakage.
  * 2.7.14: (Broken) Now generates `speaker.json` when diarization is enabled.
  * 2.7.13: Default diarization model is now pyannote/speaker-diarization-3.1
  * 2.7.12: Adds srt_swap for line breaks and improved isolated_environment usage.
  * 2.7.11: `--device insane` now generates a *.vtt translation file
  * 2.7.10: Better support for namespaced models. Trims text output in output json. Output json is now formatted with indents. SRT file is now printed out for `--device insane`
  * 2.7.9: All SRT translation errors fixed for `--device insane`. All tests pass.
  * 2.7.8: During error of `--device insane`, write out the error.json file into the destination.
  * 2.7.7: Better error messages during failure.
  * 2.7.6: Improved generation of out.txt, removes linebreaks.
  * 2.7.5: `--device insane` now generates better conforming srt files.
  * 2.7.3: Various fixes for the `insane` mode backend.
  * 2.7.0: Introduces an `insanely-fast-whisper`, enable by using `--device insane`
  * 2.6.0: GPU acceleration now happens automatically on Windows thanks to `isolated-environment`. This will also prevent
           interference with different versions of torch for other AI tools.
  * 2.5.0: `--model large` now aliases to `--model large-v3`. Use `--model large-legacy` to use original large model.
  * 2.4.0: pytorch updated to 2.1.2, gpu install script updated to same + cuda version is now 121.
  * 2.3.9: Fallback to `cpu` device if `gpu` device is not compatible.
  * 2.3.8: Fix --models arg which
  * 2.3.7: Critical fix: fixes dependency breakage with open-ai. Fixes windows use of embedded tool.
  * 2.3.6: Fixes typo in readme for installation instructions.
  * 2.3.5: Now has `--embed` to burn the subtitles into the video itself. Only works on local mp4 files at the moment.
  * 2.3.4: Removed `out.mp3` and instead use a temporary wav file, as that is faster to process. --no-keep-audio has now been removed.
  * 2.3.3: Fix case where there spaces in name (happens on windows)
  * 2.3.2: Fix windows transcoding error
  * 2.3.1: static-ffmpeg >= 2.5 now specified
  * 2.3.0: Now uses the official version of whisper ai
  * 2.2.1: "test_" is now prepended to all the different output folder names.
  * 2.2.0: Now explictly setting a language will put the file in a folder with that language name, allowing multi language passes without overwriting.
  * 2.1.2: yt-dlp pinned to new minimum version. Fixes downloading issues from old lib. Adds audio normalization by default.
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

## Notes:

  * Insanely Fast whisper for GPU
    * https://github.com/Vaibhavs10/insanely-fast-whisper
  * Fast Whisper for CPU
    * https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file
  * A better whisper CLI that supports more options but has a manual install.
    * https://github.com/ochen1/insanely-fast-whisper-cli/blob/main/requirements.txt
  * Subtitles translator:
    * https://github.com/TDHM/Subtitles-Translator
  * Forum post on how to avoid stuttering
    * https://community.openai.com/t/how-to-avoid-hallucinations-in-whisper-transcriptions/125300/23
  * More stable transcriptions:
    * https://github.com/jianfch/stable-ts?tab=readme-ov-file

