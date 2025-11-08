#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

# This script must be called from the root of the repo,
# after it's already been cloned.

source deploy/setup_components/install_dependencies.sh
source deploy/setup_components/install_tensorrt.sh
source deploy/setup_components/install_rust.sh
source deploy/setup_components/install_node.sh
source deploy/setup_components/setup_repo.sh
source deploy/setup_components/install_grafana_alloy.sh
source deploy/setup_components/run_cargo_tests.sh

echo "🎉 Setup complete!"
