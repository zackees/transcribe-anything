#! /bin/bash
set -e
pylint transcribe_anything tests
flake8 transcribe_anything tests
mypy transcribe_anything tests