#! /bin/bash

set -e
pip uninstall -y torch torchvision torchaudio
pip cache purge

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -e .

echo "transcribe audio is installed, run it with transcribe_audio <URL OR FILE>"