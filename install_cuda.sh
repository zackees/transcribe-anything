#! /bin/bash

set -e

# get the stdout from pip list
TENSOR_VERSION="1.12.1"
CUDA_VERSION="cu116"
TENSOR_CUDA_VERSION="$TENSOR_VERSION+$CUDA_VERSION"

# Delete the torch package if it doesn't have the cuda version
if [[ $(pip list --format json) == *"$TENSOR_CUDA_VERSION"* ]]; then
  echo "Tensorflow $TENSOR_CUDA_VERSION is current installed"
else
  # The substring "1.12.1+cu116" does not exist in the string
  echo "The substring '1.12.1+cu116' does not exist in the string."
  pip uninstall -y torch
  echo "Purging pip cache to remove any torch packages that are cpu only"
  pip cache purge
fi

echo "Installing torch+cuda"
pip install "torch==$TENSOR_VERSION" --extra-index-url https://download.pytorch.org/whl/cu116

echo "Install transcribe_anything"
pip install .

echo "transcribe audio is installed, run it with transcribe_audio <URL OR FILE>"
