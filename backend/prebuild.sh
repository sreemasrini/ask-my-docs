#!/bin/bash
set -e

# Print Python version
python3 --version

# Upgrade pip and install wheel
python3 -m pip install --upgrade pip
python3 -m pip install wheel

# Force pre-built wheels with compatibility
python3 -m pip install --only-binary=:all: -r requirements.txt
