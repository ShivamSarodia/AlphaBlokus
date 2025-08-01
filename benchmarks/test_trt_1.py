import torch
import ctypes
import torch.nn as nn
import tensorrt as trt
import numpy as np
import cuda.bindings.runtime as cuda_runtime
from queue import Queue
import time
from contextlib import contextmanager
from timer import Timer

from network import NeuralNet

BATCH_SIZE = 128
ONNX_MODEL_PATH = "/tmp/neuralnet.onnx"
ENGINE_PATH = "/tmp/neuralnet.trt"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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

def get_tensor_size(engine, tensor_name):
    dtype, shape = get_tensor_info(engine, tensor_name)
    return int(np.prod(shape) * np.dtype(dtype).itemsize)

def get_pinned_tensor_as_numpy(engine, tensor_name, pointer):
    dtype, shape = get_tensor_info(engine, tensor_name)
    nbytes = get_tensor_size(engine, tensor_name)
    buffer = (ctypes.c_uint8 * nbytes).from_address(pointer)
    return np.ndarray(shape, dtype=dtype, buffer=buffer)

# Training actor's job
generate_onnx_file()

# Model downloading actor's job
generate_engine_file()

# Inference actor's job
cuda_runtime.cudaDeviceSynchronize()  # Ensure the device is synchronized before starting inference

engine = load_engine_from_file()
context = engine.create_execution_context()

# Confirm I got the order of tensors right.
assert [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)] == ['boards', 'values', 'policy']

# Inference actor's job per-iteration
print("Starting inference...")

boards_size = get_tensor_size(engine, "boards")
values_size = get_tensor_size(engine, "values")
policy_size = get_tensor_size(engine, "policy")

print("Policy size:", policy_size)

spots_size = 4

spots = Queue()
for _ in range(spots_size):
    spots.put(
        {
            "boards": handleCudaError(cuda_runtime.cudaMalloc(boards_size)),
            "values_device": handleCudaError(cuda_runtime.cudaMalloc(values_size)),
            "policy_device": handleCudaError(cuda_runtime.cudaMalloc(policy_size)),
            "values_host": handleCudaError(cuda_runtime.cudaMallocHost(values_size)),
            "policy_host": handleCudaError(cuda_runtime.cudaMallocHost(policy_size)),
        }
    )

h_to_d_stream = handleCudaError(cuda_runtime.cudaStreamCreateWithFlags(cuda_runtime.cudaStreamNonBlocking))
compute_stream = handleCudaError(cuda_runtime.cudaStreamCreateWithFlags(cuda_runtime.cudaStreamNonBlocking))
d_to_h_stream = handleCudaError(cuda_runtime.cudaStreamCreateWithFlags(cuda_runtime.cudaStreamNonBlocking))

timer = Timer(unit="ms")
for i in range(100):
    if i < 10:
        timer.disable()
    else:
        timer.enable()

    with timer.name("get_spots"):
        spotset = spots.get()
        boards_device_memory = spotset["boards"]
        values_device_memory = spotset["values_device"]
        policy_device_memory = spotset["policy_device"]
        values_host_as_numpy = get_pinned_tensor_as_numpy(engine, "values", spotset["values_host"])
        policy_host_as_numpy = get_pinned_tensor_as_numpy(engine, "policy", spotset["policy_host"])

    with timer.name("generate_data"):
        # Generate input data
        boards = np.random.randn(*engine.get_tensor_shape("boards")).astype(trt.nptype(engine.get_tensor_dtype("boards")))

    with timer.name("generate_events"):
        # Create events for synchronization.
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

    with timer.name("host_to_device"):
        # Move input data to device memory.
        handleCudaError(cuda_runtime.cudaMemcpyAsync(
            boards_device_memory,
            boards,
            boards_size,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
            h_to_d_stream,
        ))
        handleCudaError(cuda_runtime.cudaEventRecord(finished_h2d_event, h_to_d_stream))

    # Wait for the host-to-device transfer to finish before we start computation.
    with timer.name("wait_for_h2d"):
        handleCudaError(cuda_runtime.cudaStreamWaitEvent(compute_stream, finished_h2d_event, 0))

    with timer.name("set_addresses"):
        context.set_tensor_address("boards", boards_device_memory)
        context.set_tensor_address("values", values_device_memory)
        context.set_tensor_address("policy", policy_device_memory)

    with timer.name("execute"):
        # Execute the inference.
        context.execute_async_v3(compute_stream)
        handleCudaError(cuda_runtime.cudaEventRecord(finished_computation_event, compute_stream))

    with timer.name("wait_for_compute"):
        handleCudaError(cuda_runtime.cudaStreamWaitEvent(d_to_h_stream, finished_computation_event, 0))

    with timer.name("device_to_host"):
        # Move output data back to host memory.
        handleCudaError(cuda_runtime.cudaMemcpyAsync(
            values_host_as_numpy,
            values_device_memory,
            values_size,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
            d_to_h_stream,
        ))
        handleCudaError(cuda_runtime.cudaMemcpyAsync(
            policy_host_as_numpy,
            policy_device_memory,
            policy_size,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
            d_to_h_stream,
        ))
        handleCudaError(cuda_runtime.cudaEventRecord(finished_d2h_event, d_to_h_stream))

    with timer.name("synchronize"):
        # Synchronize to block the host until the D2H transfer is complete.
        handleCudaError(cuda_runtime.cudaEventSynchronize(finished_d2h_event))

    spots.put(spotset)

    with timer.name("clean_up_events"):
        handleCudaError(cuda_runtime.cudaEventDestroy(finished_h2d_event))
        handleCudaError(cuda_runtime.cudaEventDestroy(finished_computation_event))

    with timer.name("torch_equivalent"):
        # Confirm that TensorRT and PyTorch outputs are similar.
        boards_tensor = torch.from_numpy(boards)
        with torch.no_grad():
            torch_values, torch_policy = torch_model(boards_tensor)
        torch_values = torch_values.numpy()
        torch_policy = torch_policy.numpy()

    # print(values[53])
    # print(torch_values[53])
    # max_diff_values = np.max(np.abs(values - torch_values))
    # max_diff_policy = np.max(np.abs(policy - torch_policy))
    # print(f"Max diff (values): {max_diff_values:.3e}")
    # print(f"Max diff (policy): {max_diff_policy:.3e}")
    assert np.allclose(values_host_as_numpy, torch_values, atol=1e-3, rtol=1e-3), "Values outputs differ!"
    assert np.allclose(policy_host_as_numpy, torch_policy, atol=1e-3, rtol=1e-3), "Policy outputs differ!"

while not spots.empty():
    for k, v in spots.get().items():
        if k.endswith("_host"):
            handleCudaError(cuda_runtime.cudaFreeHost(v))
        else:
            handleCudaError(cuda_runtime.cudaFree(v))

timer.print_results()
