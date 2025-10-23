use std::pin::Pin;
use std::slice;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use cxx::let_cxx_string;

use crate::{
    config::{GameConfig, NUM_PLAYERS},
    inference,
    inference::batcher::Executor,
    inference::softmax::softmax_inplace,
    tensorrt::bridge::ffi,
};

const BOARD_INPUT_NAME: &str = "board";
const VALUE_OUTPUT_NAME: &str = "value";
const POLICY_OUTPUT_NAME: &str = "policy";
const TENSORRT_FLOAT_DATATYPE: i32 = 0;

struct Engine {
    inner: cxx::UniquePtr<ffi::TrtEngine>,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    fn new(inner: cxx::UniquePtr<ffi::TrtEngine>) -> Self {
        Self { inner }
    }

    fn pin_mut(&mut self) -> Pin<&mut ffi::TrtEngine> {
        self.inner.pin_mut()
    }
}

pub struct TensorRtExecutor {
    game_config: &'static GameConfig,
    engine: Arc<Mutex<Engine>>,
    streams: Streams,
    memory_pool: MemoryPool,
    sizes: TensorShapes,
    max_batch_size: usize,
}

impl TensorRtExecutor {
    pub fn build(
        model_path: &std::path::Path,
        game_config: &'static GameConfig,
        max_batch_size: usize,
        pool_size: usize,
    ) -> Result<Self> {
        if max_batch_size == 0 {
            panic!("max_batch_size must be greater than zero");
        }
        if pool_size == 0 {
            panic!("pool_size must be greater than zero");
        }

        let model_path_str = model_path
            .to_str()
            .unwrap_or_else(|| panic!("Model path must be valid UTF-8: {}", model_path.display()));

        let engine_unique = {
            let_cxx_string!(model_path_cxx = model_path_str);
            ffi::create_engine(&model_path_cxx, max_batch_size)
        }
        .context("Failed to create TensorRT engine")?;

        let sizes = TensorShapes::from_engine(&engine_unique)?;

        let input_bytes = sizes.input_elements_per_item * std::mem::size_of::<f32>();
        let value_bytes = sizes.value_elements_per_item * std::mem::size_of::<f32>();
        let policy_bytes = sizes.policy_elements_per_item * std::mem::size_of::<f32>();

        let memory_pool = MemoryPool::new(
            pool_size,
            MemoryBlockSizes {
                input_device_bytes: input_bytes * max_batch_size,
                value_device_bytes: value_bytes * max_batch_size,
                policy_device_bytes: policy_bytes * max_batch_size,
                input_host_bytes: input_bytes * max_batch_size,
                value_host_bytes: value_bytes * max_batch_size,
                policy_host_bytes: policy_bytes * max_batch_size,
            },
        )?;

        let streams = Streams::new()?;

        Ok(Self {
            game_config,
            engine: Arc::new(Mutex::new(Engine::new(engine_unique))),
            streams,
            memory_pool,
            sizes,
            max_batch_size,
        })
    }
}

impl Executor for TensorRtExecutor {
    fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response> {
        if requests.is_empty() {
            return Vec::new();
        }

        if requests.len() > self.max_batch_size {
            panic!(
                "Batch size {} exceeds configured TensorRT max batch size {}",
                requests.len(),
                self.max_batch_size
            );
        }

        let batch_size = requests.len();
        let block_handle = self.memory_pool.acquire();
        let block = block_handle.block();

        let input_elements = self.sizes.input_elements_per_item;
        let value_elements = self.sizes.value_elements_per_item;
        let policy_elements = self.sizes.policy_elements_per_item;

        let input_floats = unsafe {
            slice::from_raw_parts_mut(
                block.input_host.as_mut_ptr() as *mut f32,
                input_elements * batch_size,
            )
        };

        input_floats.fill(0.0);
        let board_area = self.game_config.board_area();

        for (batch_index, request) in requests.iter().enumerate() {
            let offset = batch_index * input_elements;
            let sample_slice = &mut input_floats[offset..offset + input_elements];

            for player in 0..NUM_PLAYERS {
                let player_offset = player * board_area;
                let slice = request.board.slice(player);
                for x in 0..self.game_config.board_size {
                    for y in 0..self.game_config.board_size {
                        if slice.get((x, y)) {
                            let idx = player_offset + x * self.game_config.board_size + y;
                            sample_slice[idx] = 1.0;
                        }
                    }
                }
            }
        }

        let input_bytes = batch_size * input_elements * std::mem::size_of::<f32>();
        let value_bytes = batch_size * value_elements * std::mem::size_of::<f32>();
        let policy_bytes = batch_size * policy_elements * std::mem::size_of::<f32>();

        let h2d_event = CudaEvent::new(false).expect("Failed to create H2D event");
        let compute_event = CudaEvent::new(false).expect("Failed to create compute event");
        let d2h_event = CudaEvent::new(true).expect("Failed to create D2H event");

        unsafe {
            ffi::memcpy_h2d_async(
                block.input_device.ptr(),
                block.input_host.as_ptr(),
                input_bytes,
                self.streams.h2d_handle(),
            )
            .expect("cudaMemcpyAsync H2D failed");
        }
        ffi::event_record(h2d_event.handle(), self.streams.h2d_handle())
            .expect("Failed to record H2D event");

        {
            let mut engine_guard = self.engine.lock().unwrap();
            let mut engine = engine_guard.pin_mut();

            ffi::stream_wait_event(self.streams.compute_handle(), h2d_event.handle())
                .expect("Failed to wait for H2D event on compute stream");

            ffi::set_input_shape(engine.as_mut(), batch_size)
                .expect("Failed to set input shape on TensorRT context");

            let_cxx_string!(board_name = BOARD_INPUT_NAME);
            let_cxx_string!(value_name = VALUE_OUTPUT_NAME);
            let_cxx_string!(policy_name = POLICY_OUTPUT_NAME);

            ffi::set_tensor_address(engine.as_mut(), &board_name, block.input_device.ptr())
                .expect("Failed to set board tensor address");
            ffi::set_tensor_address(engine.as_mut(), &value_name, block.value_device.ptr())
                .expect("Failed to set value tensor address");
            ffi::set_tensor_address(engine.as_mut(), &policy_name, block.policy_device.ptr())
                .expect("Failed to set policy tensor address");

            ffi::enqueue(engine.as_mut(), self.streams.compute_handle())
                .expect("Failed to enqueue TensorRT inference");

            ffi::event_record(compute_event.handle(), self.streams.compute_handle())
                .expect("Failed to record compute event");
        }

        ffi::stream_wait_event(self.streams.d2h_handle(), compute_event.handle())
            .expect("Failed to wait for compute event on D2H stream");

        unsafe {
            ffi::memcpy_d2h_async(
                block.value_host.as_mut_ptr(),
                block.value_device.ptr(),
                value_bytes,
                self.streams.d2h_handle(),
            )
            .expect("cudaMemcpyAsync value D2H failed");
            ffi::memcpy_d2h_async(
                block.policy_host.as_mut_ptr(),
                block.policy_device.ptr(),
                policy_bytes,
                self.streams.d2h_handle(),
            )
            .expect("cudaMemcpyAsync policy D2H failed");
        }

        ffi::event_record(d2h_event.handle(), self.streams.d2h_handle())
            .expect("Failed to record D2H event");

        ffi::event_synchronize(d2h_event.handle()).expect("Failed to synchronize on D2H event");

        let values = unsafe {
            slice::from_raw_parts(
                block.value_host.as_ptr() as *const f32,
                batch_size * value_elements,
            )
        };
        let policies = unsafe {
            slice::from_raw_parts(
                block.policy_host.as_ptr() as *const f32,
                batch_size * policy_elements,
            )
        };

        let mut responses = Vec::with_capacity(batch_size);
        let policy_dims = &self.sizes.policy_shape_without_batch;
        let policy_orientation_stride = policy_dims[1] * policy_dims[2];
        let policy_row_stride = policy_dims[2];

        for (batch_index, request) in requests.into_iter().enumerate() {
            let value_offset = batch_index * value_elements;
            let mut value = [0.0f32; NUM_PLAYERS];
            value.copy_from_slice(&values[value_offset..value_offset + NUM_PLAYERS]);
            softmax_inplace(&mut value);

            let policy_offset = batch_index * policy_elements;
            let policy_slice = &policies[policy_offset..policy_offset + policy_elements];
            let mut policy_values = request
                .valid_move_indexes
                .iter()
                .map(|&move_index| {
                    let profile = self.game_config.move_profiles().get(move_index);
                    let orientation = profile.piece_orientation_index;
                    let row = profile.center.0;
                    let col = profile.center.1;
                    let index =
                        orientation * policy_orientation_stride + row * policy_row_stride + col;
                    *policy_slice.get(index).expect("Policy index out of bounds")
                })
                .collect::<Vec<f32>>();
            softmax_inplace(&mut policy_values);

            responses.push(inference::Response {
                value,
                policy: policy_values,
            });
        }

        responses
    }
}

struct TensorShapes {
    input_elements_per_item: usize,
    value_elements_per_item: usize,
    policy_elements_per_item: usize,
    policy_shape_without_batch: Vec<usize>,
}

impl TensorShapes {
    fn from_engine(engine: &cxx::UniquePtr<ffi::TrtEngine>) -> Result<Self> {
        let_cxx_string!(board_name = BOARD_INPUT_NAME);
        let_cxx_string!(value_name = VALUE_OUTPUT_NAME);
        let_cxx_string!(policy_name = POLICY_OUTPUT_NAME);

        let board_shape = ffi::get_tensor_shape(engine, &board_name);
        let value_shape = ffi::get_tensor_shape(engine, &value_name);
        let policy_shape = ffi::get_tensor_shape(engine, &policy_name);

        ensure_float_tensor(engine, &board_name);
        ensure_float_tensor(engine, &value_name);
        ensure_float_tensor(engine, &policy_name);

        let policy_shape_without_batch = shape_without_batch(&policy_shape);
        if policy_shape_without_batch.len() != 3 {
            panic!(
                "Expected policy tensor to have 3 dimensions (excluding batch), got {}",
                policy_shape_without_batch.len()
            );
        }

        Ok(Self {
            input_elements_per_item: product_without_batch(&board_shape),
            value_elements_per_item: product_without_batch(&value_shape),
            policy_elements_per_item: product_without_batch(&policy_shape),
            policy_shape_without_batch,
        })
    }
}

fn ensure_float_tensor(engine: &cxx::UniquePtr<ffi::TrtEngine>, tensor_name: &cxx::CxxString) {
    let dtype = ffi::get_tensor_dtype(engine, tensor_name);
    if dtype != TENSORRT_FLOAT_DATATYPE {
        panic!(
            "Expected tensor {} to have float dtype, got {}",
            tensor_name.to_string_lossy(),
            dtype
        );
    }
}

fn product_without_batch(shape: &[i32]) -> usize {
    if shape.is_empty() {
        panic!("Tensor shape must include batch dimension");
    }
    if shape[0] != -1 && shape[0] != 1 {
        panic!("Tensor batch dimension must be dynamic (-1) or 1");
    }
    shape[1..]
        .iter()
        .map(|&d| {
            if d <= 0 {
                panic!("Tensor dimensions must be positive");
            }
            d as usize
        })
        .product()
}

fn shape_without_batch(shape: &[i32]) -> Vec<usize> {
    if shape.is_empty() {
        panic!("Tensor shape must include batch dimension");
    }
    shape
        .iter()
        .skip(1)
        .map(|&d| {
            if d <= 0 {
                panic!("Tensor dimensions must be positive");
            }
            d as usize
        })
        .collect()
}

struct MemoryBlockSizes {
    input_device_bytes: usize,
    value_device_bytes: usize,
    policy_device_bytes: usize,
    input_host_bytes: usize,
    value_host_bytes: usize,
    policy_host_bytes: usize,
}

struct DeviceBuffer {
    ptr: usize,
}

impl DeviceBuffer {
    fn new(size: usize) -> Result<Self> {
        let ptr = ffi::cuda_malloc(size).context("cudaMalloc failed")?;
        Ok(Self { ptr })
    }

    fn ptr(&self) -> usize {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            ffi::cuda_free(self.ptr);
        }
    }
}

struct HostBuffer {
    ptr: usize,
}

impl HostBuffer {
    fn new(size: usize) -> Result<Self> {
        let ptr = ffi::cuda_malloc_host(size).context("cudaMallocHost failed")?;
        Ok(Self { ptr })
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            ffi::cuda_free_host(self.ptr);
        }
    }
}

struct MemoryBlock {
    input_device: DeviceBuffer,
    value_device: DeviceBuffer,
    policy_device: DeviceBuffer,
    input_host: HostBuffer,
    value_host: HostBuffer,
    policy_host: HostBuffer,
}

struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

struct MemoryPoolInner {
    blocks: Vec<Arc<MemoryBlock>>,
    available: Mutex<Vec<usize>>,
    condvar: Condvar,
}

impl MemoryPool {
    fn new(pool_size: usize, sizes: MemoryBlockSizes) -> Result<Self> {
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

    fn acquire(&self) -> MemoryBlockGuard {
        let mut available = self.inner.available.lock().unwrap();
        loop {
            if let Some(index) = available.pop() {
                let block = Arc::clone(&self.inner.blocks[index]);
                return MemoryBlockGuard {
                    pool: Arc::clone(&self.inner),
                    index,
                    block,
                };
            }
            available = self.inner.condvar.wait(available).unwrap();
        }
    }
}

struct MemoryBlockGuard {
    pool: Arc<MemoryPoolInner>,
    index: usize,
    block: Arc<MemoryBlock>,
}

impl MemoryBlockGuard {
    fn block(&self) -> &MemoryBlock {
        &self.block
    }
}

impl Drop for MemoryBlockGuard {
    fn drop(&mut self) {
        let mut available = self.pool.available.lock().unwrap();
        available.push(self.index);
        self.pool.condvar.notify_one();
    }
}

struct Streams {
    h2d: usize,
    compute: usize,
    d2h: usize,
}

impl Streams {
    fn new() -> Result<Self> {
        let h2d = ffi::create_stream().context("Failed to create H2D CUDA stream")?;
        let compute = match ffi::create_stream().context("Failed to create compute CUDA stream") {
            Ok(stream) => stream,
            Err(err) => {
                ffi::destroy_stream(h2d);
                return Err(err);
            }
        };
        let d2h = match ffi::create_stream().context("Failed to create D2H CUDA stream") {
            Ok(stream) => stream,
            Err(err) => {
                ffi::destroy_stream(compute);
                ffi::destroy_stream(h2d);
                return Err(err);
            }
        };
        Ok(Self { h2d, compute, d2h })
    }

    fn h2d_handle(&self) -> usize {
        self.h2d
    }

    fn compute_handle(&self) -> usize {
        self.compute
    }

    fn d2h_handle(&self) -> usize {
        self.d2h
    }
}

impl Drop for Streams {
    fn drop(&mut self) {
        ffi::destroy_stream(self.h2d);
        ffi::destroy_stream(self.compute);
        ffi::destroy_stream(self.d2h);
    }
}

struct CudaEvent {
    handle: usize,
}

impl CudaEvent {
    fn new(blocking: bool) -> Result<Self> {
        Ok(Self {
            handle: ffi::create_event(blocking).context("Failed to create CUDA event")?,
        })
    }

    fn handle(&self) -> usize {
        self.handle
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        ffi::destroy_event(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Board;
    use crate::inference::OrtExecutor;
    use crate::testing;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use std::path::Path;

    fn clone_request(request: &inference::Request) -> inference::Request {
        inference::Request {
            board: request.board.clone(),
            valid_move_indexes: request.valid_move_indexes.clone(),
        }
    }

    fn random_request(game_config: &'static GameConfig, rng: &mut StdRng) -> inference::Request {
        let mut board = Board::new(game_config);
        for player in 0..NUM_PLAYERS {
            for _ in 0..4 {
                let x = rng.random_range(0..game_config.board_size);
                let y = rng.random_range(0..game_config.board_size);
                board.slice_mut(player).set((x, y), true);
            }
        }

        let mut indexes: Vec<usize> = (0..game_config.num_moves).collect();
        indexes.shuffle(rng);
        let valid_move_indexes = indexes.into_iter().take(8).collect();

        inference::Request {
            board,
            valid_move_indexes,
        }
    }

    #[test]
    fn tensorrt_matches_ort_executor_outputs() {
        let game_config = testing::create_half_game_config();
        let model_path = Path::new("static/networks/trivial_net_half.onnx");

        match ffi::cuda_malloc(0) {
            Ok(ptr) => ffi::cuda_free(ptr),
            Err(err) => {
                eprintln!("Skipping TensorRT comparison test, CUDA unavailable: {err}");
                return;
            }
        }

        let ort =
            OrtExecutor::build(model_path, game_config).expect("failed to build ORT executor");
        let tensorrt = match TensorRtExecutor::build(model_path, game_config, 16, 4) {
            Ok(executor) => executor,
            Err(err) => {
                eprintln!("Skipping TensorRT comparison test: {err}");
                return;
            }
        };

        let mut rng = StdRng::seed_from_u64(0xDEC0BEEF);
        let requests: Vec<inference::Request> = (0..4)
            .map(|_| random_request(game_config, &mut rng))
            .collect();

        let ort_responses = ort.execute(requests.iter().map(clone_request).collect());
        let trt_responses = tensorrt.execute(requests);

        let value_tolerance = 1e-3;
        let policy_tolerance = 1e-3;

        for (ort_response, trt_response) in ort_responses.iter().zip(trt_responses.iter()) {
            for (ort_value, trt_value) in ort_response.value.iter().zip(trt_response.value.iter()) {
                assert!(
                    (ort_value - trt_value).abs() <= value_tolerance,
                    "value mismatch: {} vs {}",
                    ort_value,
                    trt_value
                );
            }

            assert_eq!(ort_response.policy.len(), trt_response.policy.len());
            for (ort_policy, trt_policy) in
                ort_response.policy.iter().zip(trt_response.policy.iter())
            {
                assert!(
                    (ort_policy - trt_policy).abs() <= policy_tolerance,
                    "policy mismatch: {} vs {}",
                    ort_policy,
                    trt_policy
                );
            }
        }
    }
}
