#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting installation of basic dependencies..."

sudo apt-get update -y
sudo apt-get install -y curl build-essential pkg-config libssl-dev python3 python3-pip git-lfs npm

echo "✅ Installed basic dependencies"
