#include "gemm_multistage.h"
#include <utils.h>
#include <reference.h>

template <typename Kernel>
__global__ void kernel_wrapper(void* __restrict__ Aptr,
                               void* __restrict__ Bptr,
                               void* __restrict__ Cptr,
                               int M, int N, int K) {
    Kernel gemm;
    gemm(Aptr, Bptr, Cptr, M, N, K);
}


int main() {
    int M = 4096;
    int N = 4096;
    int K = 1024;
    using T = half_t;
    
    auto A = make_cutlass_rowmajor_tensor<T>(M, K);
    auto B = make_cutlass_colmajor_tensor<T>(K, N);
    auto C = make_cutlass_rowmajor_tensor<T>(M, N);
    auto C_ref = make_cutlass_rowmajor_tensor<T>(M, N);

    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 0, 2, -2);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 0, 2, -2);
    
    A.sync_device();
    B.sync_device();
    
    using CTATile = decltype(make_shape(_128{}, _128{}, _32{}));
    using GemmKernel = GemmMultiStageSM80<CTATile, 3>;
    auto launch_config = GemmKernel::get_launch_config(M, N, K);

    // a lambda function to run kernel
    auto run_kernel = [&](int repeat) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < repeat; ++i) {
            CUDA_CHECK(cudaLaunchKernelEx(&launch_config, 
                                          kernel_wrapper<GemmKernel>,
                                          A.device_data(),
                                          B.device_data(),
                                          C.device_data(),
                                          M, N, K));
        }
        timer.stop();
        print(timer.elapsed_millis() / repeat); print("\n");
    };

    run_kernel(1000);
    C.sync_host();

    // reference 
    cutlass_gemmTN_ref(A, B, C_ref);
    C_ref.sync_host();
    cpu_cosine_similarity(C.host_data(), C_ref.host_data(), C_ref.capacity());

    // print tensor
    // auto C_tensor = make_tensor(C.host_data(), make_layout(make_shape(128, 128), LayoutRight{}));
    // print("C tensor \n"); print_tensor(C_tensor);
}