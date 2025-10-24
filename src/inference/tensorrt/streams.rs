use anyhow::{Context, Result};

use crate::tensorrt::bridge::ffi;

use super::utils::log_ffi_error;

pub struct Streams {
    h2d: usize,
    compute: usize,
    d2h: usize,
}

impl Streams {
    pub fn new() -> Result<Self> {
        let h2d = ffi::create_stream().context("Failed to create H2D CUDA stream")?;
        let compute = match ffi::create_stream().context("Failed to create compute CUDA stream") {
            Ok(stream) => stream,
            Err(err) => {
                log_ffi_error(
                    ffi::destroy_stream(h2d),
                    "Failed to destroy H2D stream after compute stream creation failure",
                );
                return Err(err);
            }
        };
        let d2h = match ffi::create_stream().context("Failed to create D2H CUDA stream") {
            Ok(stream) => stream,
            Err(err) => {
                log_ffi_error(
                    ffi::destroy_stream(compute),
                    "Failed to destroy compute stream after D2H stream creation failure",
                );
                log_ffi_error(
                    ffi::destroy_stream(h2d),
                    "Failed to destroy H2D stream after D2H stream creation failure",
                );
                return Err(err);
            }
        };
        Ok(Self { h2d, compute, d2h })
    }

    pub fn h2d_handle(&self) -> usize {
        self.h2d
    }

    pub fn compute_handle(&self) -> usize {
        self.compute
    }

    pub fn d2h_handle(&self) -> usize {
        self.d2h
    }
}

impl Drop for Streams {
    fn drop(&mut self) {
        log_ffi_error(
            ffi::destroy_stream(self.h2d),
            "Failed to destroy H2D CUDA stream during drop",
        );
        log_ffi_error(
            ffi::destroy_stream(self.compute),
            "Failed to destroy compute CUDA stream during drop",
        );
        log_ffi_error(
            ffi::destroy_stream(self.d2h),
            "Failed to destroy D2H CUDA stream during drop",
        );
    }
}

pub struct CudaEvent {
    handle: usize,
}

impl CudaEvent {
    pub fn new(blocking: bool) -> Result<Self> {
        Ok(Self {
            handle: ffi::create_event(blocking).context("Failed to create CUDA event")?,
        })
    }

    pub fn handle(&self) -> usize {
        self.handle
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        log_ffi_error(
            ffi::destroy_event(self.handle),
            "Failed to destroy CUDA event during drop",
        );
    }
}
