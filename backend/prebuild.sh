#!/bin/bash
set -e

# Upgrade pip and install wheel
pip install --upgrade pip
pip install wheel

# Force pre-built wheels
pip install --only-binary=:all: -r requirements.txt
