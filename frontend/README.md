# AlphaBlokus browser frontend

This is a static WebGPU application: it does not use the Rust `web-play`
server or make `/api` calls. The ONNX model pair and canonical move table are
checked into `static` with Git LFS and are served alongside the app.

```bash
cd frontend
npm install
npm run build:wasm
npm run dev
```

The WASM build uses the Rustup toolchain explicitly, so it continues to work
when a Homebrew Rust installation appears earlier on `PATH`. Install the
target once if needed: `rustup target add wasm32-unknown-unknown`.

The browser and native application both load `static/move_data/full.bin`.
Regenerate it only when the precomputed move data changes. GitHub Pages copies
the checked-in artifact; it never regenerates it:

```bash
cargo run --release --bin generate-move-data -- \
  --config configs/generate_move_data/full.toml
```
