import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
import cuda.bindings.driver as cuda
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
            f"CUDA error: {cuda.cuGetErrorString(tup[0])} ({tup[0]})"
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

def get_tensor_size(engine, tensor_name):
    dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
    shape = engine.get_tensor_shape(tensor_name)
    return np.prod(shape) * np.dtype(dtype).itemsize

def get_tensor_np_empty(engine, tensor_name):
    dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
    shape = engine.get_tensor_shape(tensor_name)
    return np.empty(shape, dtype=dtype)

# Training actor's job
# generate_onnx_file()

# Model downloading actor's job
# generate_engine_file()

# Inference actor's job
cuda.cuInit(0)
engine = load_engine_from_file()
context = engine.create_execution_context()

# Confirm I got the order of tensors right.
assert [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)] == ['boards', 'values', 'policy']

# Inference actor's job per-iteration
print("Starting inference...")

boards_size = get_tensor_size(engine, "boards")
values_size = get_tensor_size(engine, "values")
policy_size = get_tensor_size(engine, "policy")

spots_size = 4

spots = Queue()
for _ in range(spots_size):
    spots.put(
        {
            "boards": handleCudaError(cuda.cuMemAlloc(boards_size)),
            "values": handleCudaError(cuda.cuMemAlloc(values_size)),
            "policy": handleCudaError(cuda.cuMemAlloc(policy_size)),
        }
    )

timer = Timer(unit="ms")
for _ in range(100):
    with timer.name("get_spots"):
        spotset = spots.get()
        boards_device_memory = spotset["boards"]
        values_device_memory = spotset["values"]
        policy_device_memory = spotset["policy"]

    with timer.name("generate_data"):
        # Generate input data
        boards = np.random.randn(*engine.get_tensor_shape("boards")).astype(trt.nptype(engine.get_tensor_dtype("boards")))

    with timer.name("host_to_device"):
        # Move input data to device memory.
        handleCudaError(cuda.cuMemcpyHtoD(boards_device_memory, boards, boards_size))

    with timer.name("execute"):
        # Execute the inference.
        success = context.execute_v2(
            bindings=[
                boards_device_memory,
                values_device_memory,
                policy_device_memory,
            ]
        )
    if not success:
        raise RuntimeError("Failed to execute inference")

    with timer.name("device_to_host"):
        # Move output data back to host memory.
        values = get_tensor_np_empty(engine, "values")
        policy = get_tensor_np_empty(engine, "policy")
        handleCudaError(cuda.cuMemcpyDtoH(values, values_device_memory, values_size))
        handleCudaError(cuda.cuMemcpyDtoH(policy, policy_device_memory, policy_size))

    with timer.name("synchronize"):
        # Synchronize to make sure everything is done.
        handleCudaError(cuda.cuCtxSynchronize())

    assert (not np.isnan(values).any()) and (not np.isnan(policy).any()), "NaN detected in outputs!"

    spots.put(spotset)

    # # Confirm that TensorRT and PyTorch outputs are similar.
    # boards_tensor = torch.from_numpy(boards)
    # with torch.no_grad():
    #     torch_values, torch_policy = torch_model(boards_tensor)
    # torch_values = torch_values.numpy()
    # torch_policy = torch_policy.numpy()

    # print(values[53])
    # print(torch_values[53])

    # max_diff_values = np.max(np.abs(values - torch_values))
    # max_diff_policy = np.max(np.abs(policy - torch_policy))
    # print(f"Max diff (values): {max_diff_values:.3e}")
    # print(f"Max diff (policy): {max_diff_policy:.3e}")
    # assert np.allclose(values, torch_values, atol=1e-3, rtol=1e-3), "Values outputs differ!"
    # assert np.allclose(policy, torch_policy, atol=1e-3, rtol=1e-3), "Policy outputs differ!"

while not spots.empty():
    for _, value in spots.get().items():
        handleCudaError(cuda.cuMemFree(value))

timer.print_results()
