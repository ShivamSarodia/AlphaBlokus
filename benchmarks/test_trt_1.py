import torch
import tensorrt as trt
import numpy as np
import cuda.bindings.driver as cuda

from network import NeuralNet

BATCH_SIZE = 128
ONNX_MODEL_PATH = "/tmp/neuralnet.onnx"
ENGINE_PATH = "/tmp/neuralnet.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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
    model = NeuralNet().eval()
    input = torch.randn(BATCH_SIZE, 5, 20, 20)
    torch.onnx.export(
        model,
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
generate_onnx_file()

# Model downloading actor's job
generate_engine_file()

# Inference actor's job
cuda.cuInit(0)
engine = load_engine_from_file()
context = engine.create_execution_context()

# Confirm I got the order of tensors right.
assert [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)] == ['boards', 'values', 'policy']

# Inference actor's job per-iteration
print("Starting inference...")
for _ in range(1):
    # Generate input data
    boards = np.random.randn(*engine.get_tensor_shape("boards")).astype(trt.nptype(engine.get_tensor_dtype("boards")))

    # Allocate memory for input and output tensors. (TODO: Share memory between calls!)
    boards_size = get_tensor_size(engine, "boards")
    values_size = get_tensor_size(engine, "values")
    policy_size = get_tensor_size(engine, "policy")
    boards_device_memory = handleCudaError(cuda.cuMemAlloc(boards_size))
    values_device_memory = handleCudaError(cuda.cuMemAlloc(values_size))
    policy_device_memory = handleCudaError(cuda.cuMemAlloc(policy_size))

    # Move input data to device memory.
    handleCudaError(cuda.cuMemcpyHtoD(boards_device_memory, boards, boards_size))

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

    # Move output data back to host memory.
    values = get_tensor_np_empty(engine, "values")
    policy = get_tensor_np_empty(engine, "policy")
    handleCudaError(cuda.cuMemcpyDtoH(values, values_device_memory, values_size))
    handleCudaError(cuda.cuMemcpyDtoH(policy, policy_device_memory, policy_size))

    # Free the allocated device memory.
    handleCudaError(cuda.cuMemFree(boards_device_memory))
    handleCudaError(cuda.cuMemFree(values_device_memory))
    handleCudaError(cuda.cuMemFree(policy_device_memory))

    # Synchronize to make sure everything is done.
    handleCudaError(cuda.cuCtxSynchronize())

    # Print the results.
    print("Values:", values[0])
    print("Policy:", policy[0][10:])
