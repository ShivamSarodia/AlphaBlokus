import ray
import torch
import cuda.bindings.runtime as cuda_runtime
import tensorrt as trt
import numpy as np
from timer import Timer
from network import NeuralNet
import threading
import ctypes
import time
from queue import Queue


BATCH_SIZE = 128
ONNX_MODEL_PATH = "/tmp/neuralnet.onnx"
ENGINE_PATH = "/tmp/neuralnet.trt"
NUM_INFERENCE_THREADS = 24

torch_model = NeuralNet().eval()

def handleCudaError(tup):
    if tup[0]:
        raise RuntimeError(
            f"CUDA error: {cuda_runtime.cudaGetErrorString(tup[0])} ({tup[0]})"
        )
    if len(tup) == 2:
        return tup[1]
    else:
        return tup[1:]

def generate_onnx_file():
    print("Starting ONNX export...")
    input = torch.randn(BATCH_SIZE, 5, 20, 20)
    torch.onnx.export(
        torch_model,
        input,
        ONNX_MODEL_PATH,
        export_params=True,
        do_constant_folding=True,
        input_names=['boards'],
        output_names=['values', 'policy']
    )
    print("Saved ONNX model.")

def generate_engine_file():
    print("Generating engine...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX model and write it to `network`.
    success = parser.parse_from_file(ONNX_MODEL_PATH)
    if not success:
        print("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

    # Create an engine
    builder_config = builder.create_builder_config()
    engine = builder.build_engine_with_config(network, builder_config)

    # Save the engine to file
    serialized_engine = engine.serialize()
    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized_engine)

    print("Engine saved.")

def load_engine_from_file():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(ENGINE_PATH, "rb") as f:
        serialized_engine = f.read()
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to load engine from file")
    return engine

def get_tensor_info(engine, tensor_name):
    dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
    shape = engine.get_tensor_shape(tensor_name)
    return dtype, shape

def get_tensor_size(dtype, shape):
    return int(np.prod(shape) * np.dtype(dtype).itemsize)

def get_pinned_tensor_as_numpy(dtype, shape, pointer):
    nbytes = get_tensor_size(dtype, shape)
    buffer = (ctypes.c_uint8 * nbytes).from_address(pointer)
    return np.ndarray(shape, dtype=dtype, buffer=buffer)

generate_onnx_file()
generate_engine_file()

print("Starting main run")

class InferenceActor:
    def __init__(self) -> None:
        print("Starting InferenceActor...")

        self.time_in_lock = 0
        self.time_synchronizing = 0

        cuda_runtime.cudaDeviceSynchronize()

        self.lock = threading.Lock()

        self.engine = load_engine_from_file()
        self.context = self.engine.create_execution_context()

        self.boards_dtype, self.boards_shape = get_tensor_info(self.engine, "boards")
        self.values_dtype, self.values_shape = get_tensor_info(self.engine, "values")
        self.policy_dtype, self.policy_shape = get_tensor_info(self.engine, "policy")
        self.boards_size = get_tensor_size(self.boards_dtype, self.boards_shape)
        self.values_size = get_tensor_size(self.values_dtype, self.values_shape)
        self.policy_size = get_tensor_size(self.policy_dtype, self.policy_shape)

        self.spots = Queue()
        for _ in range(int(NUM_INFERENCE_THREADS * 1.1)):
            self.spots.put(
                {
                    "boards_device": handleCudaError(cuda_runtime.cudaMalloc(self.boards_size)),
                    "values_device": handleCudaError(cuda_runtime.cudaMalloc(self.values_size)),
                    "policy_device": handleCudaError(cuda_runtime.cudaMalloc(self.policy_size)),
                    "values_host": handleCudaError(cuda_runtime.cudaMallocHost(self.values_size)),
                    "policy_host": handleCudaError(cuda_runtime.cudaMallocHost(self.policy_size)),
                }
            )

        self.h_to_d_stream = handleCudaError(cuda_runtime.cudaStreamCreateWithFlags(cuda_runtime.cudaStreamNonBlocking))
        self.compute_stream = handleCudaError(cuda_runtime.cudaStreamCreateWithFlags(cuda_runtime.cudaStreamNonBlocking))
        self.d_to_h_stream = handleCudaError(cuda_runtime.cudaStreamCreateWithFlags(cuda_runtime.cudaStreamNonBlocking))
        print("Done starting InferenceActor.")

    def evaluate_batch(self, boards):
        # Create events.
        finished_h2d_event = handleCudaError(cuda_runtime.cudaEventCreateWithFlags(
            cuda_runtime.cudaEventDisableTiming
        ))
        finished_computation_event = handleCudaError(cuda_runtime.cudaEventCreateWithFlags(
            cuda_runtime.cudaEventDisableTiming
        ))
        finished_d2h_event = handleCudaError(cuda_runtime.cudaEventCreateWithFlags(
            # https://stackoverflow.com/questions/4822809/cuda-blocking-flags
            cuda_runtime.cudaEventBlockingSync
        ))

        with self.lock:
            start_time = time.perf_counter()

            # Get a spot. (We could maybe move this outside the lock?)
            spots = self.spots.get(block=False)

            # Copy data to device.
            handleCudaError(cuda_runtime.cudaMemcpyAsync(
                spots["boards_device"],
                boards,
                self.boards_size,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
                self.h_to_d_stream,
            ))
            handleCudaError(cuda_runtime.cudaEventRecord(finished_h2d_event, self.h_to_d_stream))

            # Wait for the host-to-device transfer to finish before we start computation.
            handleCudaError(cuda_runtime.cudaStreamWaitEvent(
                self.compute_stream,
                finished_h2d_event,
                0,
            ))
            self.context.set_tensor_address("boards", spots["boards_device"])
            self.context.set_tensor_address("values", spots["values_device"])
            self.context.set_tensor_address("policy", spots["policy_device"])

            # Execute the inference
            self.context.execute_async_v3(self.compute_stream)
            handleCudaError(cuda_runtime.cudaEventRecord(
                finished_computation_event,
                self.compute_stream,
            ))

            # Wait for the computation to finish
            handleCudaError(cuda_runtime.cudaStreamWaitEvent(
                self.d_to_h_stream,
                finished_computation_event,
                0,
            ))

            values_numpy_result = get_pinned_tensor_as_numpy(self.values_dtype, self.values_shape, spots["values_host"])
            policy_numpy_result = get_pinned_tensor_as_numpy(self.policy_dtype, self.policy_shape, spots["policy_host"])

            # Copy results back to host
            handleCudaError(cuda_runtime.cudaMemcpyAsync(
                values_numpy_result,
                spots["values_device"],
                self.values_size,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
                self.d_to_h_stream,
            ))
            handleCudaError(cuda_runtime.cudaMemcpyAsync(
                policy_numpy_result,
                spots["policy_device"],
                self.policy_size,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
                self.d_to_h_stream,
            ))
            handleCudaError(cuda_runtime.cudaEventRecord(finished_d2h_event, self.d_to_h_stream))

            self.time_in_lock += time.perf_counter() - start_time

        # Wait for all the processing to finish.
        handleCudaError(cuda_runtime.cudaEventSynchronize(finished_d2h_event))

        # Copy the Numpy arrays so that we can free the pinned host memory that currently
        # holds them to be used by another inference request.
        #
        # This isn't very efficient, but I don't think we care because the GPU is queued
        # up with other requests that it's asynchronously processing already.
        values = values_numpy_result.copy()
        policy = policy_numpy_result.copy()

        # Release the spots to be recycled elsewhere.
        self.spots.put(spots)

        # Free the events we created.
        handleCudaError(cuda_runtime.cudaEventDestroy(finished_h2d_event))
        handleCudaError(cuda_runtime.cudaEventDestroy(finished_computation_event))
        handleCudaError(cuda_runtime.cudaEventDestroy(finished_d2h_event))

        # Return the results.
        return values, policy

    def get_times(self):
        return {
            "in_lock": self.time_in_lock,
        }

inference_actor = InferenceActor()

def thread_function():
    print("Thread started.")
    start_time = time.perf_counter()
    while True:
        if time.perf_counter() - start_time > 10:
            print("Thread finished.")
            break
        boards = np.random.randn(BATCH_SIZE, 5, 20, 20).astype(np.float32)
        values, policy = inference_actor.evaluate_batch(boards)

threads = []
for i in range(10):
    t = threading.Thread(target=thread_function)
    t.daemon = True  # Optional: makes threads exit when the main program exits
    t.start()
    threads.append(t)

time.sleep(60)
