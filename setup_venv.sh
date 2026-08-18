#!/usr/bin/env bash
# Create and provision the Python virtual environment (Linux / macOS)
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Environment ready. Activate later with: source .venv/bin/activate"
