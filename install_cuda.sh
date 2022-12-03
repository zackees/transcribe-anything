#! /bin/bash

set -e
pip uninstall -y torch torchvision torchaudio

echo "Purging pip cache to remove any torch packages that are cpu only"
pip cache purge

echo "Installing torch+cuda"
pip install "torch<1.13.0" --extra-index-url https://download.pytorch.org/whl/cu116

echo "Install transcribe_anything"
pip install .

echo "transcribe audio is installed, run it with transcribe_audio <URL OR FILE>"
