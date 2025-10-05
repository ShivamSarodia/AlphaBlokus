#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("alpha_blokus/src/tensorrt/cpp/tensorrt.h");

        fn print_hello();
    }
}

pub fn print_hello() {
    ffi::print_hello();
}
