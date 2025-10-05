fn main() {
    cxx_build::bridge("src/tensorrt/bridge.rs")
        .file("src/tensorrt/cpp/tensorrt.cpp")
        .compile("tensorrt");

    println!("cargo:rerun-if-changed=src/tensorrt/cpp/tensorrt.cpp");
    println!("cargo:rerun-if-changed=src/tensorrt/cpp/tensorrt.h");
}
