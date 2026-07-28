#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting installation of Pentobi..."

# I don't think I need these because I'm not building the GUI? qt6-declarative-dev qt6-tools-dev 
sudo apt install -y cmake g++ gettext itstool librsvg2-bin make libxkbcommon-dev libxkbcommon-x11-dev
cd /workspace
git clone https://github.com/enz/pentobi.git
cd pentobi
cmake . -DPENTOBI_BUILD_GTP=ON -DPENTOBI_BUILD_GUI=OFF
make

cd /workspace/AlphaBlokus

echo "✅ Installed Pentobi"
