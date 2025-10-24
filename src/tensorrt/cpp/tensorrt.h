#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rust/cxx.h"

namespace alpha_blokus {

class TrtEngine {
public:
    TrtEngine(const std::string& model_path, std::size_t max_batch_size);
    ~TrtEngine();

    std::vector<int32_t> tensor_shape(const std::string& tensor_name) const;
    int32_t tensor_dtype(const std::string& tensor_name) const;
    void set_input_shape(std::size_t batch_size);
    void set_tensor_address(const std::string& tensor_name, std::uintptr_t device_ptr);
    void enqueue(std::uintptr_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<TrtEngine> create_engine(
    const std::string& model_path,
    std::size_t max_batch_size);

::rust::Vec<int32_t> get_tensor_shape(
    const TrtEngine& engine,
    const std::string& tensor_name);

int32_t get_tensor_dtype(
    const TrtEngine& engine,
    const std::string& tensor_name);

void set_input_shape(
    TrtEngine& engine,
    std::size_t batch_size);

void set_tensor_address(
    TrtEngine& engine,
    const std::string& tensor_name,
    std::uintptr_t device_ptr);

void enqueue(
    TrtEngine& engine,
    std::uintptr_t stream);

std::uintptr_t cuda_malloc(std::size_t size);
void cuda_free(std::uintptr_t ptr);

std::uintptr_t cuda_malloc_host(std::size_t size);
void cuda_free_host(std::uintptr_t ptr);

std::uintptr_t create_stream();
void destroy_stream(std::uintptr_t stream);

std::uintptr_t create_event(bool blocking);
void destroy_event(std::uintptr_t event);

void memcpy_h2d_async(
    std::uintptr_t dst_device,
    const std::uint8_t* src_host,
    std::size_t size,
    std::uintptr_t stream);

void memcpy_d2h_async(
    std::uint8_t* dst_host,
    std::uintptr_t src_device,
    std::size_t size,
    std::uintptr_t stream);

void event_record(std::uintptr_t event, std::uintptr_t stream);
void stream_wait_event(std::uintptr_t stream, std::uintptr_t event);
void event_synchronize(std::uintptr_t event);

}  // namespace alpha_blokus
