# AlphaBlokus ONNX Runtime Web build

The browser uses a model-specific ONNX Runtime Web 1.24.1 JSEP build to keep
Safari's WebAssembly compilation footprint bounded. The paired generated files
are committed under `src/assets/`.

From an ONNX Runtime 1.24.1 checkout, rebuild them with:

```sh
python3 tools/ci_build/build.py \
  --config MinSizeRel \
  --parallel \
  --skip_submodule_sync \
  --build_wasm \
  --enable_wasm_simd \
  --enable_wasm_threads \
  --use_jsep \
  --target onnxruntime_webassembly \
  --enable_wasm_api_exception_catching \
  --disable_rtti \
  --include_ops_by_config /absolute/path/to/frontend/onnxruntime/alpha-blokus.required-operators.config \
  --enable_reduced_operator_type_support \
  --build_dir /tmp/alphablokus-ort-jsep-build \
  --skip_tests
```

Copy the resulting paired `.jsep.mjs` and `.jsep.wasm` files from the build
configuration directory to `src/assets/`. Regenerate the operator config before
rebuilding if the browser model graph changes.
