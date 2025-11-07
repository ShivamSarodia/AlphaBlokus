#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting installation of Rust..."

curl -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "✅ Installed Rust"
