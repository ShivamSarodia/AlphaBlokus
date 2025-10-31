#include "tensorrt.h"

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alpha_blokus {
namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TensorRT] " << msg << "\n";
    }
  }
};

void check_cuda(cudaError_t status, const std::string& context) {
  if (status != cudaSuccess) {
    std::ostringstream oss;
    oss << context << ": " << cudaGetErrorString(status) << " (" << static_cast<int>(status)
        << ")";
    throw std::runtime_error(oss.str());
  }
}

void check_trt(bool ok, const std::string& message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

std::vector<int32_t> dims_to_vector(const nvinfer1::Dims& dims) {
  return std::vector<int32_t>(dims.d, dims.d + dims.nbDims);
}

}  // namespace

struct TrtEngine::Impl {
  Impl(const std::string& model_path, std::size_t max_batch_size);

  std::vector<int32_t> tensor_shape(const std::string& name) const;
  int32_t tensor_dtype(const std::string& name) const;
  void set_input_shape(std::size_t batch_size);
  void set_tensor_address(const std::string& tensor_name, std::uintptr_t device_ptr);
  void enqueue(std::uintptr_t stream);

  Logger logger_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};
  int32_t profile_index_{0};
  int32_t max_batch_size_{0};
  std::vector<std::string> input_names_;
  std::vector<nvinfer1::Dims> input_dims_;
};

TrtEngine::Impl::Impl(const std::string& model_path, std::size_t max_batch_size)
    : max_batch_size_(static_cast<int32_t>(max_batch_size)) {
  if (max_batch_size == 0) {
    throw std::runtime_error("max_batch_size must be greater than zero");
  }
  if (max_batch_size > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("max_batch_size exceeds supported limits");
  }

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
  check_trt(builder != nullptr, "Failed to create TensorRT builder");

  // TensorRT 10+ always operates in explicit batch mode so no flags are required.
  constexpr uint32_t network_flags = 0;
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(network_flags));
  check_trt(network != nullptr, "Failed to create TensorRT network");

  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));
  check_trt(parser != nullptr, "Failed to create ONNX parser");

  const int parser_severity =
      static_cast<int>(nvinfer1::ILogger::Severity::kERROR);
  if (!parser->parseFromFile(model_path.c_str(), parser_severity)) {
    std::ostringstream oss;
    oss << "Failed to parse ONNX model: " << model_path;
    const int nb_errors = parser->getNbErrors();
    for (int i = 0; i < nb_errors; ++i) {
      const nvonnxparser::IParserError* err = parser->getError(i);
      oss << "\n[" << static_cast<int>(err->code()) << "] " << err->file() << ":"
          << err->line() << " - " << err->desc();
    }
    throw std::runtime_error(oss.str());
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  check_trt(config != nullptr, "Failed to create TensorRT builder config");
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  check_trt(profile != nullptr, "Failed to create optimization profile");

  const int nb_inputs = network->getNbInputs();
  input_names_.reserve(nb_inputs);
  input_dims_.reserve(nb_inputs);
  for (int i = 0; i < nb_inputs; ++i) {
    nvinfer1::ITensor* tensor = network->getInput(i);
    input_names_.emplace_back(tensor->getName());
    const nvinfer1::Dims dims = tensor->getDimensions();
    input_dims_.push_back(dims);

    nvinfer1::Dims min_dims = dims;
    nvinfer1::Dims opt_dims = dims;
    nvinfer1::Dims max_dims = dims;

    if (dims.d[0] == -1) {
      min_dims.d[0] = 1;
      opt_dims.d[0] = std::max<int32_t>(1, max_batch_size_ / 2);
      max_dims.d[0] = max_batch_size_;
    } else {
      check_trt(dims.d[0] <= max_batch_size_, "Model batch size exceeds configured maximum");
      min_dims.d[0] = dims.d[0];
      opt_dims.d[0] = dims.d[0];
      max_dims.d[0] = dims.d[0];
    }

    check_trt(
        profile->setDimensions(tensor->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims),
        "Failed to set minimum optimization profile dimensions for " + input_names_.back());
    check_trt(
        profile->setDimensions(tensor->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims),
        "Failed to set optimum optimization profile dimensions for " + input_names_.back());
    check_trt(
        profile->setDimensions(tensor->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims),
        "Failed to set maximum optimization profile dimensions for " + input_names_.back());
  }

  profile_index_ = config->addOptimizationProfile(profile);
  check_trt(profile_index_ >= 0, "Failed to register optimization profile");

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
  check_trt(engine_ != nullptr, "Failed to build TensorRT engine");

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  check_trt(context_ != nullptr, "Failed to create TensorRT execution context");
}

std::vector<int32_t> TrtEngine::Impl::tensor_shape(const std::string& name) const {
  const nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
  return dims_to_vector(dims);
}

int32_t TrtEngine::Impl::tensor_dtype(const std::string& name) const {
  return static_cast<int32_t>(engine_->getTensorDataType(name.c_str()));
}

void TrtEngine::Impl::set_input_shape(std::size_t batch_size) {
  if (batch_size == 0 || batch_size > static_cast<std::size_t>(max_batch_size_)) {
    throw std::runtime_error("Batch size exceeds configured maximum");
  }

  const int32_t batch = static_cast<int32_t>(batch_size);
  for (std::size_t i = 0; i < input_names_.size(); ++i) {
    nvinfer1::Dims dims = input_dims_[i];
    if (dims.nbDims == 0) {
      throw std::runtime_error("Input tensor has invalid dimensions");
    }
    dims.d[0] = batch;
    check_trt(
        context_->setInputShape(input_names_[i].c_str(), dims),
        "Failed to set input shape for tensor " + input_names_[i]);
  }
}

void TrtEngine::Impl::set_tensor_address(const std::string& tensor_name, std::uintptr_t device_ptr) {
  void* address = reinterpret_cast<void*>(device_ptr);
  check_trt(
      context_->setTensorAddress(tensor_name.c_str(), address),
      "Failed to set tensor address for " + tensor_name);
}

void TrtEngine::Impl::enqueue(std::uintptr_t stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  check_trt(context_->enqueueV3(cuda_stream), "Failed to enqueue TensorRT inference");
}

TrtEngine::TrtEngine(const std::string& model_path, std::size_t max_batch_size)
    : impl_(std::make_unique<Impl>(model_path, max_batch_size)) {}

TrtEngine::~TrtEngine() = default;

std::vector<int32_t> TrtEngine::tensor_shape(const std::string& name) const {
  return impl_->tensor_shape(name);
}

int32_t TrtEngine::tensor_dtype(const std::string& name) const {
  return impl_->tensor_dtype(name);
}

void TrtEngine::set_input_shape(std::size_t batch_size) {
  impl_->set_input_shape(batch_size);
}

void TrtEngine::set_tensor_address(const std::string& tensor_name, std::uintptr_t device_ptr) {
  impl_->set_tensor_address(tensor_name, device_ptr);
}

void TrtEngine::enqueue(std::uintptr_t stream) {
  impl_->enqueue(stream);
}

std::uintptr_t to_uintptr(void* ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr);
}

void* from_uintptr(std::uintptr_t ptr) {
  return reinterpret_cast<void*>(ptr);
}

cudaStream_t to_stream(std::uintptr_t stream) {
  return reinterpret_cast<cudaStream_t>(stream);
}

cudaEvent_t to_event(std::uintptr_t event) {
  return reinterpret_cast<cudaEvent_t>(event);
}

std::unique_ptr<TrtEngine> create_engine(
    const std::string& model_path,
    std::size_t max_batch_size) {
  return std::make_unique<TrtEngine>(model_path, max_batch_size);
}

::rust::Vec<int32_t> get_tensor_shape(
    const TrtEngine& engine,
    const std::string& tensor_name) {
  const std::vector<int32_t> dims = engine.tensor_shape(tensor_name);
  ::rust::Vec<int32_t> result;
  result.reserve(dims.size());
  for (int32_t value : dims) {
    result.push_back(value);
  }
  return result;
}

int32_t get_tensor_dtype(
    const TrtEngine& engine,
    const std::string& tensor_name) {
  return engine.tensor_dtype(tensor_name);
}

void set_input_shape(
    TrtEngine& engine,
    std::size_t batch_size) {
  engine.set_input_shape(batch_size);
}

void set_tensor_address(
    TrtEngine& engine,
    const std::string& tensor_name,
    std::uintptr_t device_ptr) {
  engine.set_tensor_address(tensor_name, device_ptr);
}

void enqueue(
    TrtEngine& engine,
    std::uintptr_t stream) {
  engine.enqueue(stream);
}

std::uintptr_t cuda_malloc(std::size_t size) {
  void* ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, size), "cudaMalloc");
  return to_uintptr(ptr);
}

void cuda_free(std::uintptr_t ptr) {
  if (ptr != 0) {
    check_cuda(cudaFree(from_uintptr(ptr)), "cudaFree");
  }
}

std::uintptr_t cuda_malloc_host(std::size_t size) {
  void* ptr = nullptr;
  check_cuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
  return to_uintptr(ptr);
}

void cuda_free_host(std::uintptr_t ptr) {
  if (ptr != 0) {
    check_cuda(cudaFreeHost(from_uintptr(ptr)), "cudaFreeHost");
  }
}

std::uintptr_t create_stream() {
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  return to_uintptr(stream);
}

void destroy_stream(std::uintptr_t stream) {
  if (stream != 0) {
    check_cuda(cudaStreamDestroy(to_stream(stream)), "cudaStreamDestroy");
  }
}

std::uintptr_t create_event(bool blocking) {
  unsigned int flags = cudaEventDisableTiming;
  if (blocking) {
    flags |= cudaEventBlockingSync;
  }
  cudaEvent_t event = nullptr;
  check_cuda(cudaEventCreateWithFlags(&event, flags), "cudaEventCreateWithFlags");
  return to_uintptr(event);
}

void destroy_event(std::uintptr_t event) {
  if (event != 0) {
    check_cuda(cudaEventDestroy(to_event(event)), "cudaEventDestroy");
  }
}

void memcpy_h2d_async(
    std::uintptr_t dst_device,
    const std::uint8_t* src_host,
    std::size_t size,
    std::uintptr_t stream) {
  check_cuda(
      cudaMemcpyAsync(
          from_uintptr(dst_device),
          reinterpret_cast<void const*>(src_host),
          size,
          cudaMemcpyHostToDevice,
          to_stream(stream)),
      "cudaMemcpyAsync H2D");
}

void memcpy_d2h_async(
    std::uint8_t* dst_host,
    std::uintptr_t src_device,
    std::size_t size,
    std::uintptr_t stream) {
  check_cuda(
      cudaMemcpyAsync(
          reinterpret_cast<void*>(dst_host),
          from_uintptr(src_device),
          size,
          cudaMemcpyDeviceToHost,
          to_stream(stream)),
      "cudaMemcpyAsync D2H");
}

void event_record(std::uintptr_t event, std::uintptr_t stream) {
  check_cuda(cudaEventRecord(to_event(event), to_stream(stream)), "cudaEventRecord");
}

void stream_wait_event(std::uintptr_t stream, std::uintptr_t event) {
  check_cuda(cudaStreamWaitEvent(to_stream(stream), to_event(event), 0), "cudaStreamWaitEvent");
}

void event_synchronize(std::uintptr_t event) {
  check_cuda(cudaEventSynchronize(to_event(event)), "cudaEventSynchronize");
}

}  // namespace alpha_blokus
