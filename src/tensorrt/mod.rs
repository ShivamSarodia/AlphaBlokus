#[cfg(target_os = "linux")]
pub mod bridge;

#[cfg(target_os = "linux")]
pub use bridge::print_hello;

#[cfg(not(target_os = "linux"))]
pub fn print_hello() {
    panic!("TensorRT support is only available on Linux targets.");
}
