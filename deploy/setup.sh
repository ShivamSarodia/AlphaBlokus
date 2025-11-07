#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

# This script must be called from the root of the repo,
# after it's already been cloned.

source setup_components/install_dependencies.sh"
source setup_components/install_tensorrt.sh"
source setup_components/install_rust.sh"
source setup_components/install_node.sh"
source setup_components/setup_repo.sh"
source setup_components/install_grafana_alloy.sh"
source setup_components/run_rust_tests.sh"

echo "🎉 Setup complete!"
