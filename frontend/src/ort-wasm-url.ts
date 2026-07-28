// Keep the model-specific ONNX Runtime Web backend fingerprinted by Vite.
// The generated loader and binary must stay paired: this reduced build
// exports only the runtime surface AlphaBlokus needs.
export default {
  mjs: new URL('./assets/ort-wasm-alphablokus.jsep.mjs', import.meta.url).href,
  wasm: new URL('./assets/ort-wasm-alphablokus.jsep.wasm', import.meta.url).href,
}
