# Mini Tensor Library (C++)

A small, fast, header-only tensor library written from scratch in C++17 for the Diwali Challenge.

## Features
- N-dimensional float tensors in contiguous, 64-byte aligned memory
- Safe element access (`t.at()`) and fast raw pointer access (`t[i]`)
- Element-wise addition, subtraction, multiplication, scalar operations, and in-place `add_`
- Reshape, transpose, sum, and ReLU
- Fast matrix multiplication via loop reordering (`i-k-j`), cache blocking, and OpenMP multi-threading

## Build Instructions
git clone [https://github.com/mahaj/mini-tensor.git](https://github.com/mahaj/mini-tensor.git)
cd mini-tensor && mkdir build && cd build
cmake .. && make && ctest --output-on-failure

## Benchmarks
Measured on Ubuntu (WSL) with OpenMP:
- 256 x 256: 3.0x speedup
- 512 x 512: 25.1x speedup
- 1024 x 1024: 53.0x speedup

## License
MIT
