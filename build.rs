fn main() {
    println!("cargo:rerun-if-changed=src/tensorrt/cpp/tensorrt.cpp");
    println!("cargo:rerun-if-changed=src/tensorrt/cpp/tensorrt.h");
    println!("cargo:rustc-check-cfg=cfg(cuda)");

    if !should_compile_tensorrt() {
        println!("Skipping TensorRT build: CUDA not available");
        return;
    }

    println!("cargo:rustc-cfg=cuda");

    cxx_build::bridge("src/tensorrt/bridge.rs")
        .file("src/tensorrt/cpp/tensorrt.cpp")
        .include("/usr/local/cuda/include")
        .compile("tensorrt");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvinfer_plugin");
    println!("cargo:rustc-link-lib=nvonnxparser");
    println!("cargo:rustc-link-lib=cudart");
}

fn should_compile_tensorrt() -> bool {
    // If we're not on a supported OS (i.e. Linux), skip TensorRT compilation.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        return false;
    }

    // If we're in a CI env, skip TensorRT compilation because my GitHub Actions
    // runner doesn't have an NVIDIA GPU.
    if std::env::var("CI").unwrap_or_default() == "true" {
        return false;
    }

    // Otherwise, compile TensorRT.
    true
}
