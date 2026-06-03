# Dev Instructions

This project uses `bash`

Useful scripts:

  * `./install`
  * `./clean`
  * `./test`
  * `./lint`

# uv-iso-env

This project is unique in that complex dependencies for the AI models are siloed away via an `uv-iso-env` pyproject.toml file for each backend.

WhisperX is an additional backend exposed as `--device whisperx`; it does not replace `--device insane`.

`--device insane-flash` is a separate insanely-fast-whisper environment that forces and verifies FlashAttention2. It uses `venv/insanely_fast_whisper_flash`, while normal `--device insane` continues to use `venv/insanely_fast_whisper` without requiring `flash-attn`.

The flash dependency is intentionally pinned by direct wheel URL plus sha256 in `src/transcribe_anything/flash_attention_wheels.py`. Do not replace it with an unpinned PyPI `flash-attn` source dependency; unsupported tuples should fail early or get a new checked manifest entry after GPU validation.

If you end up down this rabbit hole implementing a feature and want to test a backend, it's actually easy.

Install the program and then see where a `.venv` is created for the backend you are
interested in. You can then invoked it via `uv run <command>`.

The parent environment and the backend environment communicate via subprocess.Popen. It's not possible to call a function in the sub environment from the parent environment, you must use a subprocess.Popen interface to communicate between the two.

# Before you issue a PR

Run
  * `./lint`
  * `./test`

If you haven't already, you should run `./install` to install the dependencies for the front end. Keep in mind if the `pyproject.toml` that is generated for the backends changes *at all*, then it will nuke and rebuild backend from the `pyproject.toml` file.
