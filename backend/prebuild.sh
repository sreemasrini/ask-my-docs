#!/bin/bash
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
. venv/bin/activate

# Print Python version
python3 --version

# Upgrade pip and install wheel
python3 -m pip install --upgrade pip
python3 -m pip install wheel

# Force pre-built wheels with compatibility
python3 -m pip install --only-binary=:all: -r requirements.txt
