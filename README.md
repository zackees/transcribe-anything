# transcribe-anything
Input a local file or url and this service will transcribe it using Mozilla Deepspeech 0.9.3

# Quick start
  * `pip install transcribe-anything`
    * The command `transcribe_anything` will magically become available.
  * `transcribe_anything <YOUTUBE_URL> out_subtitles.txt`
  * -or- `transcribe_anything <MY_LOCAL.MP4/WAV> out_subtitles.txt`

# Tech Stack
  * Mozilla DeepSpeech: https://github.com/mozilla/DeepSpeech
  * mic_vad_streaming: https://github.com/hadran9/DeepSpeech-examples/tree/r0.9/mic_vad_streaming
  * youtube-dl:
    * github: https://github.com/ytdl-org/youtube-dl
  * static-ffmpeg
    * github: https://github.com/zackees/static_ffmpeg
    * pypi: https://pypi.org/project/static-ffmpeg/
