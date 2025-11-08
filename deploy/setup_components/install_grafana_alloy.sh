#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting installation of Grafana Alloy..."

# Install and start NVIDIA stats exporter
NVIDIA_GPU_EXPORTER_VERSION=1.4.1
wget https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v${NVIDIA_GPU_EXPORTER_VERSION}/nvidia_gpu_exporter_${NVIDIA_GPU_EXPORTER_VERSION}_linux_x86_64.tar.gz
tar -xvzf nvidia_gpu_exporter_${NVIDIA_GPU_EXPORTER_VERSION}_linux_x86_64.tar.gz
mv nvidia_gpu_exporter /usr/bin
nohup nvidia_gpu_exporter >> /tmp/nvidia_gpu_exporter.log 2>&1 &

# Install Alloy
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install -y alloy

# Start Alloy
nohup alloy run --server.http.listen-addr=0.0.0.0:12345 deploy/config.alloy >> /tmp/alloy.log 2>&1 &

mkdir -p /tmp/log/alphablokus/
echo "Alloy running (1)" >> /tmp/log/alphablokus/initial.log
echo "Alloy running (2)" >> /tmp/log/alphablokus/initial.log

echo "✅ Installed Grafana Alloy"
