#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting installation of Grafana Alloy..."

sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install -y alloy

nohup alloy run --server.http.listen-addr=0.0.0.0:12345 deploy/config.alloy >> /tmp/alloy.log 2>&1 &

mkdir -p /tmp/log/alphablokus/
echo "Alloy running (1)" >> /tmp/log/alphablokus/initial.log
echo "Alloy running (2)" >> /tmp/log/alphablokus/initial.log

echo "✅ Installed Grafana Alloy"
