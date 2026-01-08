use std::pin::Pin;
use std::slice;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cxx::let_cxx_string;

use crate::{
    config::{GameConfig, NUM_PLAYERS},
    inference,
    inference::batcher::Executor,
    inference::softmax::softmax_inplace,
    tensorrt::bridge::ffi,
};

use super::{
    constants::{BOARD_INPUT_NAME, POLICY_OUTPUT_NAME, VALUE_OUTPUT_NAME},
    memory::{MemoryBlockSizes, MemoryPool},
    shapes::TensorShapes,
    streams::{CudaEvent, Streams},
};

struct Engine {
    inner: cxx::UniquePtr<ffi::TrtEngine>,
}

// We don't need the Engine to be Sync because it's wrapped in a mutex by the executor,
// and won't be concurrently referenced by multiple threads.
unsafe impl Send for Engine {}

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
            bail!("max_batch_size must be greater than zero");
        }
        if pool_size == 0 {
            bail!("pool_size must be greater than zero");
        }

        let model_path_str = model_path
            .to_str()
            .context("TensorRT model path cannot convert to string")?;

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
    fn execute(&self, requests: Vec<inference::Request>) -> Result<Vec<inference::Response>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        if requests.len() > self.max_batch_size {
            bail!(
                "Batch size {} exceeds configured TensorRT max batch size {}",
                requests.len(),
                self.max_batch_size
            );
        }

        let batch_size = requests.len();
        let block_handle = self.memory_pool.acquire()?;
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

        let h2d_event = CudaEvent::new(false).context("Failed to create H2D event")?;
        let compute_event = CudaEvent::new(false).context("Failed to create compute event")?;
        let d2h_event = CudaEvent::new(true).context("Failed to create D2H event")?;

        unsafe {
            ffi::memcpy_h2d_async(
                block.input_device.ptr(),
                block.input_host.as_ptr(),
                input_bytes,
                self.streams.h2d_handle(),
            )
            .context("cudaMemcpyAsync H2D failed")?;
        }
        ffi::event_record(h2d_event.handle(), self.streams.h2d_handle())
            .context("Failed to record H2D event")?;

        {
            let mut engine_guard = self
                .engine
                .lock()
                .map_err(|_| anyhow::anyhow!("Failed to lock TensorRT engine"))?;
            let mut engine = engine_guard.pin_mut();

            ffi::stream_wait_event(self.streams.compute_handle(), h2d_event.handle())
                .context("Failed to wait for H2D event on compute stream")?;

            ffi::set_input_shape(engine.as_mut(), batch_size)
                .context("Failed to set input shape on TensorRT context")?;

            let_cxx_string!(board_name = BOARD_INPUT_NAME);
            let_cxx_string!(value_name = VALUE_OUTPUT_NAME);
            let_cxx_string!(policy_name = POLICY_OUTPUT_NAME);

            ffi::set_tensor_address(engine.as_mut(), &board_name, block.input_device.ptr())
                .context("Failed to set board tensor address")?;
            ffi::set_tensor_address(engine.as_mut(), &value_name, block.value_device.ptr())
                .context("Failed to set value tensor address")?;
            ffi::set_tensor_address(engine.as_mut(), &policy_name, block.policy_device.ptr())
                .context("Failed to set policy tensor address")?;

            ffi::enqueue(engine.as_mut(), self.streams.compute_handle())
                .context("Failed to enqueue TensorRT inference")?;

            ffi::event_record(compute_event.handle(), self.streams.compute_handle())
                .context("Failed to record compute event")?;
        }

        ffi::stream_wait_event(self.streams.d2h_handle(), compute_event.handle())
            .context("Failed to wait for compute event on D2H stream")?;

        unsafe {
            ffi::memcpy_d2h_async(
                block.value_host.as_mut_ptr(),
                block.value_device.ptr(),
                value_bytes,
                self.streams.d2h_handle(),
            )
            .context("cudaMemcpyAsync value D2H failed")?;
            ffi::memcpy_d2h_async(
                block.policy_host.as_mut_ptr(),
                block.policy_device.ptr(),
                policy_bytes,
                self.streams.d2h_handle(),
            )
            .context("cudaMemcpyAsync policy D2H failed")?;
        }

        ffi::event_record(d2h_event.handle(), self.streams.d2h_handle())
            .context("Failed to record D2H event")?;

        ffi::event_synchronize(d2h_event.handle()).context("Failed to synchronize on D2H event")?;

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

        let move_profiles = self.game_config.move_profiles()?;
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
                    let profile = move_profiles.get(move_index);
                    let orientation = profile.piece_orientation_index;
                    let row = profile.center.0;
                    let col = profile.center.1;
                    let index =
                        orientation * policy_orientation_stride + row * policy_row_stride + col;
                    // Bit dangerous, because this'll panic if the executor gets an index out of bounds
                    // which will halt execution.
                    *policy_slice.get(index).unwrap()
                })
                .collect::<Vec<f32>>();
            softmax_inplace(&mut policy_values);

            responses.push(inference::Response {
                value,
                policy: policy_values,
            });
        }

        Ok(responses)
    }
}

#[cfg(all(test, cuda))]
mod tests {
    use super::*;
    use crate::game::Board;
    use crate::inference::OrtExecutor;
    use crate::testing;
    use rand::Rng;
    use rand::seq::SliceRandom;
    use std::path::Path;

    fn random_request(game_config: &'static GameConfig) -> inference::Request {
        let mut board = Board::new(game_config);
        for player in 0..NUM_PLAYERS {
            for _ in 0..4 {
                // Set 1/4th of the board to true for each player.
                for _ in 0..game_config.board_area() / 4 {
                    let x = rand::rng().random_range(0..game_config.board_size);
                    let y = rand::rng().random_range(0..game_config.board_size);
                    board.slice_mut(player).set((x, y), true);
                }
            }
        }

        let mut indexes: Vec<usize> = (0..game_config.num_moves).collect();
        indexes.shuffle(&mut rand::rng());
        let valid_move_indexes = indexes.into_iter().take(16).collect();

        inference::Request {
            board,
            valid_move_indexes,
        }
    }

    #[test]
    fn tensorrt_matches_ort_executor_outputs() {
        let game_config = testing::create_game_config();
        let model_path = Path::new("static/networks/trivial_net_tiny.onnx");

        let ort = OrtExecutor::build(
            model_path,
            &game_config,
            crate::config::OrtExecutionProvider::Cpu,
        )
        .expect("failed to build ORT executor");
        let tensorrt = TensorRtExecutor::build(model_path, &game_config, 4, 4)
            .expect("failed to build TensorRT executor");

        let requests: Vec<inference::Request> =
            (0..4).map(|_| random_request(&game_config)).collect();

        let ort_responses = ort.execute(requests.clone()).unwrap();
        let trt_responses = tensorrt.execute(requests.clone()).unwrap();

        let value_tolerance = 1e-4;
        let policy_tolerance = 1e-4;

        for (ort_response, trt_response) in ort_responses.iter().zip(trt_responses.iter()) {
            for (ort_value, trt_value) in ort_response.value.iter().zip(trt_response.value.iter()) {
                assert!(
                    (ort_value - trt_value).abs() <= value_tolerance,
                    "value mismatch: {} vs {}",
                    ort_value,
                    trt_value
                );
                println!("{}, {}", ort_value, trt_value);
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
                println!("{}, {}", ort_policy, trt_policy);
            }
        }
    }
}
