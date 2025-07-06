#!/bin/bash

# REPLACE WITH RSYNC?
git clone https://github.com/ShivamSarodia/AlphaBlokus.git
cd AlphaBlokus

# Install poetry if not already installed
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies using poetry
poetry install
