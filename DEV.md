# Dev Instructions

This project uses `bash`

Useful scripts:

  * `./install`
  * `./clean`
  * `./test`
  * `./lint`

# uv-iso-env

This project is unique in that complex dependencies for the AI models are siloed away via an `uv-iso-env` pyproject.toml file for each backend.

If you end up down this rabbit hole implementing a feature and want to test a backend, it's actually easy.

Install the program and then see where a `.venv` is created for the backend you are
interested in. You can then invoked it via `uv run <command>`.

The parent environment and the backend environment communicate via subprocess.Popen. It's not possible to call a function in the sub environment from the parent environment, you must use a subprocess.Popen interface to communicate between the two.

# Before you issue a PR

Run
  * `./lint`
  * `./test`

If you haven't already, you should run `./install` to install the dependencies for the front end. Keep in mind if the `pyproject.toml` that is generated for the backends changes *at all*, then it will nuke and rebuild backend from the `pyproject.toml` file.