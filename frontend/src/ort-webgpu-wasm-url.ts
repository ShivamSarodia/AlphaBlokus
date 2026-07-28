// Keep the model-specific ONNX Runtime Web JSEP backend fingerprinted by Vite.
// This module is imported only when WebGPU inference is selected.
export default {
  mjs: new URL('./assets/ort-wasm-alphablokus.jsep.mjs', import.meta.url).href,
  wasm: new URL('./assets/ort-wasm-alphablokus.jsep.wasm', import.meta.url).href,
}
