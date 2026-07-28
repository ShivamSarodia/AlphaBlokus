#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting repo setup..."

git config --global user.email "ssarodia@gmail.com"
git config --global user.name "Shivam Sarodia"
git remote set-url origin "https://${GITHUB_PAT}@github.com/ShivamSarodia/AlphaBlokus.git"

pip install pre-commit
pre-commit install

mkdir /tmp/games

echo "✅ Set up repo"
