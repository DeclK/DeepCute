# DeepCute

Cute solutions to high performace CUDA kernels.

This repo is designed for following purposes:

- A readable repo to learn cuda programming
- Understand engineer tricks and hardware properties in precise docs
- Build SOTA low-bit gemm kernels in clean structure
- Provide kernel support to quantization experiment in torch
- Easy to compile and manage, and most importantly to use

## Benchmarks

## Structure

## Usage

## TODO

- [x] sm80 fp16 gemm
- [ ] sm90 fp16 gemm
- [ ] sm90 fp8 deepgemm
- [ ] fused layernorm + convert fp8 kernel
- [ ] fused silu_matmul + convert fp8 kernel
- [ ] sm100 fp8 deepgemm
- [ ] sm100 nvfp4 gemm
- [ ] predicated kernels

## Acknowledgments

This project is inspired by [Awesome-Cute](https://github.com/CalebDu/Awesome-Cute), [DeepGemm](https://github.com/deepseek-ai/DeepGEMM), [Cute-Learning](https://github.com/DD-DuDa/Cute-Learning), and of course [cute-gemm](https://github.com/reed-lau/cute-gemm) from reed