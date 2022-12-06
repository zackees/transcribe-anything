#! /bin/bash
set -e
echo running pylint...
pylint transcribe_anything tests install_cuda.py

echo running flake8...
flake8 transcribe_anything tests install_cuda.py

echo running mypy...
mypy transcribe_anything tests install_cuda.py