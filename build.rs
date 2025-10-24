fn main() {
    println!("cargo:rerun-if-changed=src/tensorrt/cpp/tensorrt.cpp");
    println!("cargo:rerun-if-changed=src/tensorrt/cpp/tensorrt.h");
    println!("cargo:rustc-check-cfg=cfg(cuda)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        println!(
            "Skipping TensorRT build script for unsupported target OS: {}",
            target_os
        );
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
