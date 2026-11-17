# Mini Tensor Library (C++)

A small, fast, header-only tensor library written from scratch in C++17 for the Diwali 2026 Challenge.

## Features
- N-dimensional float tensors in contiguous, 64-byte aligned memory
- Safe element access `t.at({i, j})` and fast flat access `t[i]`
- Element-wise add, subtract, multiply, scalar ops, in-place `add_`
- reshape, transpose, sum, relu
- Fast matrix multiplication: i-k-j loop order, cache blocking, OpenMP
- Clear exceptions for shape mismatch and out-of-range access

## Build (Ubuntu 22.04+)
```bash
sudo apt install build-essential cmake git
git clone https://github.com/mahajanparth591-ship-it/mini_tensor-.git
cd mini_tensor-
mkdir build && cd build
cmake .. && make
ctest --output-on-failure
./bench
```
For a portable build (not tied to your CPU): `cmake .. -DMT_NATIVE=OFF`

## Quick example
```cpp
#include "tensor.hpp"
Tensor A = Tensor::random({512, 512});
Tensor B = Tensor::random({512, 512});
Tensor C = Tensor::matmul(A, B);
C.at({0, 0}) = 1.0f;
```

## Benchmarks
Measured on: 13th Gen Intel Core i5-13420H, 2 CPUs visible in WSL, Ubuntu (WSL), g++ 15.2 with -O3 -march=native -fopenmp. Best of several runs (the 1024 naive case runs once).

| Size | Naive (ms) | Fast (ms) | Speed-up |
|---|---|---|---|
| 256 x 256 | 9.23 | 0.81 | 11.4x |
| 512 x 512 | 114.81 | 7.69 | 14.9x |
| 1024 x 1024 | 5402.80 | 75.81 | 71.3x |

## Design notes
- Row-major layout; offset = sum(index x stride).
- i-k-j order makes the inner loop walk neighbouring memory (cache friendly).
- 64x64 blocking keeps working tiles in cache.
- OpenMP splits row-blocks of the result across threads; each thread writes different rows, so there are no conflicts. Small inputs stay single-threaded.
- `__restrict` and flat loops let the compiler vectorise (SIMD).

## License
MIT