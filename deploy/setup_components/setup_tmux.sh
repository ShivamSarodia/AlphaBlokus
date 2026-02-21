#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting tmux setup..."

if ! echo "set -g mouse on" > ~/.tmux.conf; then
  echo "⚠️ Failed to write ~/.tmux.conf; skipping tmux setup"
elif ! tmux source-file ~/.tmux.conf; then
  echo "⚠️ Failed to run \"tmux source-file ~/.tmux.conf\"; skipping tmux setup"
else
  echo "✅ Set up tmux"
fi
