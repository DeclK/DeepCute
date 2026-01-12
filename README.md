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
- [ ] sm80 int8 gemm
- [ ] sm80 int4 gemm
- [x] sm90 fp16 gemm
- [ ] sm90 fp8 deepgemm
- [ ] sm90 int8 gemm
- [ ] fused layernorm + convert low-bit kernel
- [ ] fused silu_matmul + convert low-bit kernel
- [ ] sm120 fp16 gemm
- [ ] sm120 fp8 deepgemm
- [ ] sm120 nvfp4 gemm
- [ ] JiT in python
- [ ] benchmark of all kernels
- [ ] grouped gemm kernels
- [ ] predicated kernels
- [ ] mixed precision kernels
- [ ] attention kernels

If I have finished previous 13 items, then this project can go public, and utilze the wisdom of the community.

## Acknowledgments

This project is inspired by [Awesome-Cute](https://github.com/CalebDu/Awesome-Cute), [DeepGemm](https://github.com/deepseek-ai/DeepGEMM), [Cute-Learning](https://github.com/DD-DuDa/Cute-Learning), and of course [cute-gemm](https://github.com/reed-lau/cute-gemm) from reed