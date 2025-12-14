#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting setup of Python components..."

cd python
pip install -e .
cd ..

echo "✅ Set up Python components"
