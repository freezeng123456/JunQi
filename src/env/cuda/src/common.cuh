/*
 * common.cuh
 * Shared CUDA utilities and device functions.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

namespace junqi_cuda {

#define CUDA_CHECK(err) \
  do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1); \
    } \
  } while (0)

#define KERNEL_CHECK() \
  do { \
    CUDA_CHECK(cudaPeekAtLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (0)

__device__ inline int16_t xy_to_flat(int8_t x, int8_t y) {
  return y * 17 + x;
}

__device__ inline void flat_to_xy(int16_t flat, int8_t& x, int8_t& y) {
  x = flat % 17;
  y = flat / 17;
}

__device__ inline bool is_valid_cell(int16_t flat) {
  return flat >= 0 && flat < 289;
}

__device__ inline bool is_valid_pos(int8_t x, int8_t y) {
  return x >= 0 && x < 17 && y >= 0 && y < 17;
}

inline void* cuda_malloc(size_t size) {
  void* ptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

inline void cuda_free(void* ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

inline void* cuda_malloc_host(size_t size) {
  void* ptr;
  CUDA_CHECK(cudaMallocHost(&ptr, size));
  return ptr;
}

inline void cuda_free_host(void* ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
}

}  // namespace junqi_cuda
