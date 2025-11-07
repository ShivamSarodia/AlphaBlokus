#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

#################################
#                               #
#  Create workspace directory   #
#                               #
#################################

mkdir -p /workspace
cd /workspace

echo "✅ Set up workspace"

#################################
#                               #
#  Install basic dependencies   #
#                               #
#################################

sudo apt-get update -y
sudo apt-get install -y curl build-essential pkg-config libssl-dev python3 python3-pip git-lfs npm

echo "✅ Installed basic dependencies"

#################################
#                               #
#  Install TensorRT             #
#                               #
#################################

# Detect CUDA version
CUDA_DETECTED="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+')" || true

if [ -z "${CUDA_DETECTED:-}" ]; then
  echo "❌ Could not detect CUDA version from nvidia-smi" >&2
  exit 1
fi

echo "✅ Detected CUDA version: $CUDA_DETECTED"

case "$CUDA_DETECTED" in
  11.*)
    echo "❌ CUDA 11.x is not supported." >&2
    exit 1
    ;;
  12.*)
    CUDA_VERSION="12.9"
    ;;
  13.*)
    CUDA_VERSION="$CUDA_DETECTED"
    ;;
  *)
    echo "❌ Unsupported CUDA version \"$CUDA_DETECTED\" (expected 12.x or 13.x)." >&2
    exit 1
    ;;
esac

echo "✅ Using CUDA_VERSION=${CUDA_VERSION}"

export TRT_VERSION=$(
  apt-cache madison tensorrt \
    | awk '{print $3}' \
    | grep -F "+cuda$CUDA_VERSION" \
    | sort -V \
    | tail -1
)

if [ -z "${TRT_VERSION:-}" ]; then
  echo "❌ Could not set TensorRT version"
  exit 1
fi

echo "✅ Using TensorRT version: $TRT_VERSION"

# From here: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#net-repo-install-debian
sudo apt-mark unhold \
libnvinfer-bin \
libnvinfer-dev \
libnvinfer-dispatch-dev \
libnvinfer-dispatch10 \
libnvinfer-headers-dev \
libnvinfer-headers-plugin-dev \
libnvinfer-headers-python-plugin-dev \
libnvinfer-lean-dev \
libnvinfer-lean10 \
libnvinfer-plugin-dev \
libnvinfer-plugin10 \
libnvinfer-vc-plugin-dev \
libnvinfer-vc-plugin10 \
libnvinfer-win-builder-resource10 \
libnvinfer10 \
libnvonnxparsers-dev \
libnvonnxparsers10 \
python3-libnvinfer-dev \
python3-libnvinfer-dispatch \
python3-libnvinfer-lean \
python3-libnvinfer \
tensorrt-dev \
tensorrt-libs \
tensorrt

sudo apt-get install -y \
libnvinfer-bin=${TRT_VERSION} \
libnvinfer-dev=${TRT_VERSION} \
libnvinfer-dispatch-dev=${TRT_VERSION} \
libnvinfer-dispatch10=${TRT_VERSION} \
libnvinfer-headers-dev=${TRT_VERSION} \
libnvinfer-headers-plugin-dev=${TRT_VERSION} \
libnvinfer-headers-python-plugin-dev=${TRT_VERSION} \
libnvinfer-lean-dev=${TRT_VERSION} \
libnvinfer-lean10=${TRT_VERSION} \
libnvinfer-plugin-dev=${TRT_VERSION} \
libnvinfer-plugin10=${TRT_VERSION} \
libnvinfer-vc-plugin-dev=${TRT_VERSION} \
libnvinfer-vc-plugin10=${TRT_VERSION} \
libnvinfer-win-builder-resource10=${TRT_VERSION} \
libnvinfer10=${TRT_VERSION} \
libnvonnxparsers-dev=${TRT_VERSION} \
libnvonnxparsers10=${TRT_VERSION} \
python3-libnvinfer-dev=${TRT_VERSION} \
python3-libnvinfer-dispatch=${TRT_VERSION} \
python3-libnvinfer-lean=${TRT_VERSION} \
python3-libnvinfer=${TRT_VERSION} \
tensorrt-dev=${TRT_VERSION} \
tensorrt-libs=${TRT_VERSION} \
tensorrt=${TRT_VERSION}

sudo apt-mark hold \
libnvinfer-bin \
libnvinfer-dev \
libnvinfer-dispatch-dev \
libnvinfer-dispatch10 \
libnvinfer-headers-dev \
libnvinfer-headers-plugin-dev \
libnvinfer-headers-python-plugin-dev \
libnvinfer-lean-dev \
libnvinfer-lean10 \
libnvinfer-plugin-dev \
libnvinfer-plugin10 \
libnvinfer-vc-plugin-dev \
libnvinfer-vc-plugin10 \
libnvinfer-win-builder-resource10 \
libnvinfer10 \
libnvonnxparsers-dev \
libnvonnxparsers10 \
python3-libnvinfer-dev \
python3-libnvinfer-dispatch \
python3-libnvinfer-lean \
python3-libnvinfer \
tensorrt-dev \
tensorrt-libs \
tensorrt

echo "✅ Installed TensorRT"

#################################
#                               #
#  Install Rust.                #
#                               #
#################################

curl -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "✅ Installed Rust"

#################################
#                               #
#  Install Node.                #
#                               #
#################################

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 25

echo "✅ Installed Node"

#################################
#                               #
#  Set up repo                  #
#                               #
#################################

# Clone the repo
git config --global user.email "ssarodia@gmail.com"
git config --global user.name "Shivam Sarodia"
git remote set-url origin "https://${GITHUB_PAT}@github.com/ShivamSarodia/AlphaBlokus.git"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Install Codex for development
npm install -g @openai/codex

echo "✅ Set up repo"

#################################
#                               #
#  Install Grafana Alloy        #
#                               #
#################################
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install -y alloy
cp deploy/config.alloy /etc/alloy/config.alloy
sudo systemctl reload alloy

echo "✅ Installed Grafana Alloy"

#################################
#                               #
#  Run Rust tests               #
#                               #
#################################

# Run cargo test to confirm things work (and install dependencies)
cargo test

echo "✅ Environment setup complete!"