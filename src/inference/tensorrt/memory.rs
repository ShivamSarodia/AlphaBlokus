use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};

use crate::tensorrt::bridge::ffi;

use super::utils::log_ffi_error;

pub struct MemoryBlockSizes {
    pub input_device_bytes: usize,
    pub value_device_bytes: usize,
    pub policy_device_bytes: usize,
    pub input_host_bytes: usize,
    pub value_host_bytes: usize,
    pub policy_host_bytes: usize,
}

pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

struct MemoryPoolInner {
    blocks: Vec<Arc<MemoryBlock>>,
    available: Mutex<Vec<usize>>,
    condvar: Condvar,
}

pub struct MemoryBlockItem {
    pool: Arc<MemoryPoolInner>,
    index: usize,
    block: Arc<MemoryBlock>,
}

pub struct MemoryBlock {
    pub input_device: DeviceBuffer,
    pub value_device: DeviceBuffer,
    pub policy_device: DeviceBuffer,
    pub input_host: HostBuffer,
    pub value_host: HostBuffer,
    pub policy_host: HostBuffer,
}

pub struct DeviceBuffer {
    ptr: usize,
}

pub struct HostBuffer {
    ptr: usize,
}

impl MemoryPool {
    pub fn new(pool_size: usize, sizes: MemoryBlockSizes) -> Result<Self> {
        let mut blocks = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            blocks.push(Arc::new(MemoryBlock {
                input_device: DeviceBuffer::new(sizes.input_device_bytes)?,
                value_device: DeviceBuffer::new(sizes.value_device_bytes)?,
                policy_device: DeviceBuffer::new(sizes.policy_device_bytes)?,
                input_host: HostBuffer::new(sizes.input_host_bytes)?,
                value_host: HostBuffer::new(sizes.value_host_bytes)?,
                policy_host: HostBuffer::new(sizes.policy_host_bytes)?,
            }));
        }

        let available = (0..pool_size).rev().collect::<Vec<_>>();

        Ok(Self {
            inner: Arc::new(MemoryPoolInner {
                blocks,
                available: Mutex::new(available),
                condvar: Condvar::new(),
            }),
        })
    }

    pub fn acquire(&self) -> Result<MemoryBlockItem> {
        let mut available = self
            .inner
            .available
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock memory pool available"))?;
        loop {
            if let Some(index) = available.pop() {
                let block = Arc::clone(&self.inner.blocks[index]);
                return Ok(MemoryBlockItem {
                    pool: Arc::clone(&self.inner),
                    index,
                    block,
                });
            }
            tracing::warn!("Memory block from pool was not immediately available");
            available = self
                .inner
                .condvar
                .wait(available)
                .map_err(|err| anyhow::anyhow!("Memory pool condvar wait failed: {}", err))?;
        }
    }
}

impl MemoryBlockItem {
    pub fn block(&self) -> &MemoryBlock {
        &self.block
    }
}

impl Drop for MemoryBlockItem {
    fn drop(&mut self) {
        match self.pool.available.lock() {
            Ok(mut available) => {
                available.push(self.index);
                self.pool.condvar.notify_one();
            }
            Err(err) => {
                tracing::error!("Failed to lock memory pool available: {}", err);
            }
        };
    }
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let ptr = ffi::cuda_malloc(size).context("cudaMalloc failed")?;
        Ok(Self { ptr })
    }

    pub fn ptr(&self) -> usize {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            log_ffi_error(
                ffi::cuda_free(self.ptr),
                "Failed to free CUDA device buffer during drop",
            );
        }
    }
}

impl HostBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let ptr = ffi::cuda_malloc_host(size).context("cudaMallocHost failed")?;
        Ok(Self { ptr })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            log_ffi_error(
                ffi::cuda_free_host(self.ptr),
                "Failed to free CUDA host buffer during drop",
            );
        }
    }
}
