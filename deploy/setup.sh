#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

# This script must be called from the root of the repo,
# after it's already been cloned.

usage() {
  cat <<'EOF'
Usage: deploy/setup.sh [--python-only|--with-python]

Options:
  --python-only  Install only the Python components.
  --with-python  Run the standard setup and install Python components too.
  -h, --help     Show this help message.
EOF
}

python_only=false
with_python=false
for arg in "$@"; do
  case "$arg" in
    --python-only)
      python_only=true
      ;;
    --with-python)
      with_python=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if "$python_only" && "$with_python"; then
  echo "Error: --python-only cannot be combined with --with-python" >&2
  usage >&2
  exit 1
fi

if "$python_only"; then
  source deploy/setup_components/setup_python.sh
  echo "🎉 Python setup complete!"
  exit 0
fi

source deploy/setup_components/setup_tmux.sh
source deploy/setup_components/install_dependencies.sh
source deploy/setup_components/install_tensorrt.sh
source deploy/setup_components/install_rust.sh
source deploy/setup_components/install_node.sh
source deploy/setup_components/install_pentobi.sh
source deploy/setup_components/setup_repo.sh
if "$with_python"; then
  source deploy/setup_components/setup_python.sh
fi
source deploy/setup_components/install_grafana_alloy.sh
source deploy/setup_components/run_cargo_tests.sh

echo "🎉 Setup complete!"
