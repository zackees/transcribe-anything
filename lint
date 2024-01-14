#! /bin/bash
set -e

# does --clean exist
if [[ $* == *--no-ruff* ]]; then
    echo skipping ruff...
else
    echo running ruff...
    ruff transcribe_anything tests install_cuda.py
fi

# echo running flake8...
# flake8 transcribe_anything tests install_cuda.py

echo running pylint...
pylint transcribe_anything tests install_cuda.py

echo running mypy...
mypy transcribe_anything tests install_cuda.py