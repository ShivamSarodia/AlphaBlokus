#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting Cargo tests..."

cargo test

echo "✅ Ran Cargo tests"
