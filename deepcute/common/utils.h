#pragma once

#include <cstdarg>
#include <iostream>
#include <stdlib.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

#define round_up(x, y) ceil_div(x, y) * (y)
#define round_down(x, y) ((x) / (y)) * (y)
#define DEVICE __device__ __forceinline__
#define PRINT(x)                                                               \
  print(#x ":\n");                                                             \
  print(x);                                                                    \
  print("\n");
#define PRINT_TENSOR(x)                                                        \
  print(#x ":\n");                                                             \
  print_tensor(x);                                                             \
  print("\n");

using namespace cute;

template <typename T> inline auto make_cutlass_rowmajor_tensor(int m, int n) {
  cutlass::HostTensor<T, cutlass::layout::RowMajor> tensor(
      cutlass::MatrixCoord({m, n}));
  return tensor;
}

template <typename T> inline auto make_cutlass_colmajor_tensor(int m, int n) {
  cutlass::HostTensor<T, cutlass::layout::ColumnMajor> tensor(
      cutlass::MatrixCoord({m, n}));
  return tensor;
}

inline int get_max_smem_size() {
  int max_shared_mem;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  return max_shared_mem;
}

template <typename Kernel> void config_smem(Kernel kernel, int smem_size) {
  if (smem_size >= 32 * 1024) {
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size) != cudaSuccess) {
      int max_shared_mem = get_max_smem_size();
      cudaError_t err = cudaGetLastError();
      std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err)
                << std::endl;
      std::cerr
          << "Kernel required " << smem_size
          << " shared memory but the max shared memory per block optin is: "
          << max_shared_mem << std::endl;
    }
  }
}

int get_device_sm() {
  int current_device;
  int device_sms;
  CUDA_CHECK(cudaGetDevice(&current_device));
  CUDA_CHECK(cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount,
                                    current_device));
  return device_sms;
}

template <typename Kernel>
int get_sm_occupancy(Kernel kernel, int block_size, int smem_size) {
  int sm_occupancy;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &sm_occupancy, kernel, block_size, smem_size,
      cudaOccupancyDisableCachingOverride));
  return sm_occupancy;
}

void printf_fail(const char *fmt, ...) {
  int red = 31;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

void printf_pass(const char *fmt, ...) {
  int red = 32;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

inline
float compute_tflops(float flop, float ms) {
  float tflops = flop * 1e-9 / ms;
  return tflops;
}
