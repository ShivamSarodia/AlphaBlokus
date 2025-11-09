#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting tmux setup..."

echo "set -g mouse on" > ~/.tmux.conf
tmux source-file ~/.tmux.conf

echo "✅ Set up tmux"
