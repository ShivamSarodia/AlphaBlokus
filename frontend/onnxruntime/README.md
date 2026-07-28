# AlphaBlokus ONNX Runtime Web build

The browser uses a model-specific ONNX Runtime Web 1.24.1 non-JSEP build. It
runs inference through the SIMD CPU WebAssembly execution provider, avoiding
Safari's excessive memory growth when compiling the JSEP runtime. The paired
generated files are committed under `src/assets/`.

`tools/ci_build/build.py` belongs to the upstream ONNX Runtime repository, not
AlphaBlokus. Check out the matching release and initialize its submodules:

```sh
git clone --branch v1.24.1 --depth 1 \
  https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git submodule update --init --recursive
```

From that ONNX Runtime checkout, rebuild the artifacts with:

```sh
python3 tools/ci_build/build.py \
  --config MinSizeRel \
  --parallel \
  --skip_submodule_sync \
  --build_wasm \
  --enable_wasm_simd \
  --enable_wasm_threads \
  --target onnxruntime_webassembly \
  --enable_wasm_api_exception_catching \
  --disable_rtti \
  --include_ops_by_config /absolute/path/to/frontend/onnxruntime/alpha-blokus.required-operators.config \
  --enable_reduced_operator_type_support \
  --build_dir /tmp/alphablokus-ort-wasm-build \
  --skip_tests
```

Copy the resulting paired `.mjs` and `.wasm` files from the build configuration
directory to `src/assets/`, naming them `ort-wasm-alphablokus.mjs` and
`ort-wasm-alphablokus.wasm`. Regenerate the operator config before rebuilding
if the browser model graph changes.
