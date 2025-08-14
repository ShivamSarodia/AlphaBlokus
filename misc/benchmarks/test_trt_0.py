import torch
import tensorrt as trt
from cuda.bindings import driver as
import numpy as np

from network import NeuralNet

model = NeuralNet().eval()
dummy_input = torch.randn(128, 5, 20, 20)
torch.onnx.export(
    model,
    dummy_input,
    "/tmp/neuralnet.onnx",
    export_params=True,        # include learned parameters
    do_constant_folding=True,  # enable constant folding
    input_names=['input'],
    output_names=['values', 'policy']
)
print("ONNX export complete: neuralnet.onnx")

driver.cuInit(0)
# device = cuda.Device(0)
# ctx = device.make_context()
#
#
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def handleCudaError(tup):
    if tup[0]:
        raise RuntimeError(
            f"CUDA error: {driver.cuGetErrorString(tup[0])} ({tup[0]})"
        )
    if len(tup) == 2:
        return tup[1]
    else:
        return tup[1:]

def build_engine_from_onnx(onnx_path: str,
                           engine_path: str) -> None:
    """
    Reads an ONNX file, builds a TensorRT engine, and saves it to disk.
    """
    # 2. Create builder/network/parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 3. Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            # ctx.pop()
            return

    # 4. Configure builder
    config = builder.create_builder_config()

    # 5. Build engine
    plan = builder.build_serialized_network(network, config)
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(plan)

    if engine is None:
        print("ERROR: Failed to create TensorRT engine")
        return

    return engine

# 2. Helper: load a serialized TensorRT engine
def load_engine(engine_path: str, logger: trt.Logger) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 3. Helper: allocate host/device buffers and collect binding pointers
def allocate_buffers(engine: trt.ICudaEngine):
    inputs, outputs, bindings = [], [], []
    for idx in range(engine.num_io_tensors):
        # get name, dtype, and shape
        name = engine.get_tensor_name(idx)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = engine.get_tensor_shape(name)
        size = int(np.prod(shape))

        # allocate host and device buffers
        host_mem = np.empty(size, dtype=dtype)
        device_mem = handleCudaError(driver.cuMemAlloc(host_mem.nbytes))
        bindings.append(int(device_mem))

        # sort into inputs vs. outputs
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({"name": name, "host": host_mem, "device": device_mem, "size": host_mem.nbytes})
        elif engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outputs.append({"name": name, "host": host_mem, "device": device_mem, "size": host_mem.nbytes})
        else:
            raise RuntimeError(f"Unknown tensor mode for {name}")

    return inputs, outputs, bindings

# 4. Helper: perform inference (synchronous)
def do_inference(context, inputs, outputs, bindings):
    # copy host → device
    for inp in inputs:
        driver.cuMemcpyHtoD(inp["device"], inp["host"], inp["size"])
    # execute (blocking)
    context.execute_v2(bindings)
    # copy device → host
    for out in outputs:
        driver.cuMemcpyDtoH(out["host"], out["device"], inp["size"])
    # return list of numpy arrays
    return [out["host"] for out in outputs]

def main():
    ENGINE_PATH = "/tmp/neuralnet.trt"

    engine = build_engine_from_onnx(
        onnx_path="/tmp/neuralnet.onnx",
        engine_path=ENGINE_PATH,
    )

    # Load engine & create execution context
    context = engine.create_execution_context()

    # Allocate I/O buffers
    inputs, outputs, bindings = allocate_buffers(engine)

    # Prepare a dummy input (e.g., batch=1, 3×224×224)
    dummy = np.random.random_sample(inputs[0]["host"].shape).astype(inputs[0]["host"].dtype)
    inputs[0]["host"] = dummy

    # Run inference
    output_buffers = do_inference(context, inputs, outputs, bindings)

    # Reshape and display
    for output_buffer, output in zip(output_buffers, outputs):
        print(output_buffer)
        out_shape = engine.get_tensor_shape(output["name"])
        result = output_buffer.reshape(out_shape)
        print("Name", output["name"])
        print("Result shape:", result.shape)
        print("First 10 values:", result.flatten()[:10])

if __name__ == "__main__":
    main()
