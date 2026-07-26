// Keep the ONNX Runtime Web backend fingerprinted by Vite. The package does
// not export its `.wasm` file as an importable subpath, so reference it as a
// static URL from this source module instead.
export default new URL(
  '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm',
  import.meta.url,
).href
