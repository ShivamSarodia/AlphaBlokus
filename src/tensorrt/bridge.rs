#![allow(clippy::missing_safety_doc)]

#[cxx::bridge]
pub mod ffi {
    #[namespace = "alpha_blokus"]
    unsafe extern "C++" {
        include!("alpha_blokus/src/tensorrt/cpp/tensorrt.h");

        type TrtEngine;

        fn create_engine(
            model_path: &CxxString,
            max_batch_size: usize,
        ) -> Result<UniquePtr<TrtEngine>>;

        fn get_tensor_shape(engine: &TrtEngine, tensor_name: &CxxString) -> Vec<i32>;

        fn get_tensor_dtype(engine: &TrtEngine, tensor_name: &CxxString) -> i32;

        fn set_input_shape(engine: Pin<&mut TrtEngine>, batch_size: usize) -> Result<()>;

        fn set_tensor_address(
            engine: Pin<&mut TrtEngine>,
            tensor_name: &CxxString,
            device_ptr: usize,
        ) -> Result<()>;

        fn enqueue(engine: Pin<&mut TrtEngine>, stream: usize) -> Result<()>;

        fn print_hello();
    }

    #[namespace = "alpha_blokus"]
    unsafe extern "C++" {
        fn cuda_malloc(size: usize) -> Result<usize>;

        fn cuda_free(ptr: usize);

        fn cuda_malloc_host(size: usize) -> Result<usize>;

        fn cuda_free_host(ptr: usize);

        fn create_stream() -> Result<usize>;

        fn destroy_stream(stream: usize);

        fn create_event(blocking: bool) -> Result<usize>;

        fn destroy_event(event: usize);

        unsafe fn memcpy_h2d_async(
            dst_device: usize,
            src_host: *const u8,
            size: usize,
            stream: usize,
        ) -> Result<()>;

        unsafe fn memcpy_d2h_async(
            dst_host: *mut u8,
            src_device: usize,
            size: usize,
            stream: usize,
        ) -> Result<()>;

        fn event_record(event: usize, stream: usize) -> Result<()>;

        fn stream_wait_event(stream: usize, event: usize) -> Result<()>;

        fn event_synchronize(event: usize) -> Result<()>;
    }
}

pub use ffi::print_hello;
