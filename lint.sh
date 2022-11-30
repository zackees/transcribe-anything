#! /bin/bash
set -e
echo running pylint...
pylint transcribe_anything tests

echo running flake8...
flake8 transcribe_anything tests

echo running mypy...
mypy transcribe_anything tests