#include <cuda_runtime.h>
#include <iostream>

int print_hello() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    constexpr size_t N = 16;
    int h_data[N];
    for (size_t i = 0; i < N; ++i) h_data[i] = static_cast<int>(i);

    int* d_data = nullptr;
    err = cudaMalloc(&d_data, N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    err = cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy (H->D) failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_data);
        return 1;
    }

    int back = -1;
    err = cudaMemcpy(&back, d_data + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy (D->H) failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_data);
        return 1;
    }

    std::cout << "Transferred " << (N * sizeof(int))
              << " bytes to device. Last element round-trip = "
              << back << "\n";

    cudaFree(d_data);
    return 0;
}