# AlphaBlok ONNX Runtime Web build

The browser includes two model-specific ONNX Runtime Web 1.24.1 builds:

- a non-JSEP SIMD CPU WebAssembly runtime; and
- a JSEP runtime used by the WebGPU execution provider.

Their paired generated files are committed under `src/assets/`. Runtime
selection happens before model initialization, and only the selected pair is
fetched and compiled.

`tools/ci_build/build.py` belongs to the upstream ONNX Runtime repository, not
AlphaBlok. Check out the matching release and initialize its submodules:

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

Build the WebGPU/JSEP pair with the same command plus `--use_jsep` and a
separate build directory, then copy the resulting `.jsep.mjs` and `.jsep.wasm`
files to `src/assets/` as `ort-wasm-alphablokus.jsep.mjs` and
`ort-wasm-alphablokus.jsep.wasm`.
