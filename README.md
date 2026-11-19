# Mini Tensor Library (C++)

A small, fast, header-only tensor library written from scratch in C++17 (no external tensor code).
Built for the Diwali 2026 Challenge and tested with the **Common Test Specification v1.0** (TC01-TC50, PB01-PB12).

## Features

* N-dimensional `float` tensors (rank 0 scalars up to any rank) in one contiguous, 64-byte aligned block, row-major
* Create from shape (zero-initialised) or from data: `Tensor t({2,3}, {1,2,3,4,5,6})`
* Safe element access `t.at({i,j})` (throws `std::out\_of\_range`) and fast flat access `t\[k]`
* `+`, `-`, element-wise `mul`, scalar ops, in-place `add\_`  (shape mismatch throws `std::invalid\_argument`)
* `reshape` (copying), `reshape\_` (in-place, zero-copy), `transpose`, `sum`, `relu`
* Fast `matmul`: cache blocking + 4x16 register-blocked micro-kernel + OpenMP; `matmul\_naive` kept as reference
* Overflow-checked allocation: impossible sizes throw `std::length\_error` / `std::bad\_alloc`
* Optional `Tensor::enable\_memory\_reuse()` (glibc): keeps freed memory in the process to avoid page-fault cost

## Build (Ubuntu 22.04+)

```bash
sudo apt install build-essential cmake git python3-numpy
git clone https://github.com/mahajanparth591-ship-it/mini\_tensor-.git
cd mini\_tensor-
cmake -S . -B build \&\& cmake --build build -j
ctest --test-dir build --output-on-failure
```

Options: `-DMT\_NATIVE=OFF` (portable binary, not tied to your CPU), `-DMT\_SANITIZE=ON` (ASan + UBSan).

## Quick example

```cpp
#include "tensor.hpp"
Tensor A = Tensor::random({512, 512});
Tensor B = Tensor::random({512, 512});
Tensor C = Tensor::matmul(A, B);
C.at({0, 0}) = 1.0f;
```

Compile your own file: `g++ -std=c++17 -O3 -march=native -fopenmp -Iinclude main.cpp`

## Run the common test specification (one command)

```bash
bash tools/run\_all.sh "<Team\_ID>"
```

Creates `submission/` with: `functional\_results.csv`, `perf\_results.csv`, `perf\_results\_detail.csv`,
`reference\_results.csv` (NumPy, single thread), `results.csv` (mandatory columns, merged), `summary.md`,
`machine\_info.txt`, `sanitizer\_log.txt`. Sources: `tests/common\_test.cpp`, `benchmarks/spec\_bench.cpp`,
`benchmarks/reference\_numpy.py`, `tools/merge\_results.py`.

## Benchmarks

## Benchmarks
Measured on: Intel Core i5-13420H, 2 CPUs visible in WSL2 (Ubuntu), g++ 15.2, -O3 -DNDEBUG -march=native -fopenmp.
Reference: NumPy 2.3.5 with OpenBLAS, single thread, same machine. Protocol: 5 warm-ups, 30 timed runs, 3 repetitions (Common Test Specification v1.0).

| Case | Student median (ms) | NumPy median (ms) | Ref / student |
|---|---|---|---|
| PB11 matmul 512x512 | 2.4425 | 2.0967 | 0.86 |
| PB10 matmul 256x256 | 0.2686 | 0.2675 | 1.00 |
| PB06 add 1000x1000 | 0.6496 | 0.6337 | 0.98 |
| PB03 create 4000x4000 | 7.6698 | 9.7782 | 1.27 |
| PB05 random reads | 0.2188 | 0.4901 | 2.24 |

A ratio above 1 means this library is faster; below 1 means NumPy is faster. Matrix multiplication is close to tuned OpenBLAS but not faster. Full results: `submission/results.csv`, `submission/summary.md` and the report PDF.

## Design notes

* Row-major layout; offset = sum(index x stride).
* `matmul`: 64x256x256 cache blocks; inside, a 4x16 tile of C is held in vector registers while looping over k.
OpenMP splits row-blocks across threads (each thread writes different rows: no races); small inputs stay single-threaded.
* Element-wise ops are one flat loop over raw pointers with `\_\_restrict` + `omp simd`; outputs skip the zero-fill pass.
* `reshape()` copies (deep-copy contract); `reshape\_()` changes shape/strides in O(1). Reported separately in PB12.
* Negative indices: the API takes unsigned indices, so `-1` wraps to a huge value and is rejected by the bounds check.

## Known limits

No broadcasting, N-D transpose/permute, slicing views, or other dtypes. Allocator tuning is glibc-only.
See `docs/TEST\_NOTES.md` for the TC35 specification discrepancy.

## License

MIT

