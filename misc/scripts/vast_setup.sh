# Disable tmux
touch ~/.no_auto_tmux

# Create a /workspace directory, if needed.
mkdir -p /workspace
cd /workspace

# Clone the repo
git config --global user.email "ssarodia@gmail.com"
git config --global user.name "Shivam Sarodia"
git clone https://github.com/ShivamSarodia/AlphaBlokus.git

cd AlphaBlokus

# Set up my GitHub Personal Access Token so I can push.
git remote set-url origin https://$GITHUB_PAT@github.com/ShivamSarodia/AlphaBlokus.git

# Install Python and start a venv.
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
python3 -m venv venv
source venv/bin/activate

# Install poetry and all remaining pacakges
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

poetry install --extras linux,dev

pre-commit install
