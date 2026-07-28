// Keep the model-specific ONNX Runtime Web backend fingerprinted by Vite.
// The generated loader and binary must stay paired: this reduced non-JSEP
// build exports only the CPU WASM runtime surface AlphaBlokus needs.
export default {
  mjs: new URL('./assets/ort-wasm-alphablokus.mjs', import.meta.url).href,
  wasm: new URL('./assets/ort-wasm-alphablokus.wasm', import.meta.url).href,
}
