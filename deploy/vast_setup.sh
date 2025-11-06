# Change into the repo directory, if we're not already there.
cd /workspace/AlphaBlokus

#################################
#                               #
#  Install basic dependencies   #
#                               #
#################################

sudo apt-get update -y
sudo apt-get install -y curl build-essential pkg-config libssl-dev python3 python3-pip git-lfs npm

#################################
#                               #
#  Install TensorRT             #
#                               #
#################################

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+') || true

if [ -z "${CUDA_VERSION:-}" ]; then
  echo "❌ Could not detect CUDA version from nvidia-smi"
  exit 1
fi

echo "✅ Detected CUDA version: $CUDA_VERSION"

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

echo "✅ Detected TensorRT version: $TRT_VERSION"

# From here: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#net-repo-install-debian
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

#################################
#                               #
#  Install Rust.                #
#                               #
#################################

curl -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

#################################
#                               #
#  Install Node.                #
#                               #
#################################

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 25

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
# /etc/alloy/config.alloy < todo, hide secrets (untested)
# sudo systemctl reload alloy

#################################
#                               #
#  Set up repo for development  #
#                               #
#################################

git config --global user.email "ssarodia@gmail.com"
git config --global user.name "Shivam Sarodia"
git remote set-url origin "https://${GITHUB_PAT}@github.com/ShivamSarodia/AlphaBlokus.git"

pip install pre-commit
pre-commit install

npm install -g @openai/codex

#################################
#                               #
#  Run Rust tests               #
#                               #
#################################

# Run cargo test to confirm things work (and install dependencies)
cargo test

echo "✅ Environment setup complete!"