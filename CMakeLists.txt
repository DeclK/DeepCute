cmake_minimum_required(VERSION 3.23.1)
project(cute_learning LANGUAGES CUDA CXX)

# CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set include directories, for simplicity just include for all targets
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include)
set(CUTLASS_UTIL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include)
set(MY_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deepcute/common)
include_directories(
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_UTIL_DIR}
    ${MY_INCLUDE_DIR}
)

# set flag var
set(CUTE_LEARNING_FLAGS 
    "-arch=sm_80"
    "-std=c++17"
    "--expt-relaxed-constexpr"
    "--use_fast_math"
    "-w"    # disable warning
)

# Create executables
add_executable(gemm "deepcute/sm80/naive_gemm/gemm_multistage.cu")

# Set CUDA compilation flags
target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUTE_LEARNING_FLAGS}>)