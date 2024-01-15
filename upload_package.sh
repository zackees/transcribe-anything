#!/bin/bash

# Check the operating system
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    # Linux or Mac OS
    python3 setup.py upload
else
    # Other OS (like Windows)
    python setup.py upload
fi
