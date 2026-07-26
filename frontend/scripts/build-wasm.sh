#!/usr/bin/env sh
set -eu

# On macOS, Homebrew Rust can precede Rustup on PATH. wasm-pack discovers
# rustc directly, so use the Rustup toolchain (where the WASM target lives)
# consistently for local browser builds.
rustc_path="$(rustup which rustc)"
export PATH="$(dirname "$rustc_path"):$PATH"

wasm-pack build ../crates/browser-wasm --target web --dev --out-dir ../../frontend/src/wasm
