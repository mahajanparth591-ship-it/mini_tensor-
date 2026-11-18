# Mini Tensor Library (C++)

A small, fast, header-only tensor library written from scratch in C++17 (no external tensor code).
Built for the Diwali 2026 Challenge and tested with the **Common Test Specification v1.0** (TC01-TC50, PB01-PB12).

## Features
- N-dimensional `float` tensors (rank 0 scalars up to any rank) in one contiguous, 64-byte aligned block, row-major
- Create from shape (zero-initialised) or from data: `Tensor t({2,3}, {1,2,3,4,5,6})`
- Safe element access `t.at({i,j})` (throws `std::out_of_range`) and fast flat access `t[k]`
- `+`, `-`, element-wise `mul`, scalar ops, in-place `add_`  (shape mismatch throws `std::invalid_argument`)
- `reshape` (copying), `reshape_` (in-place, zero-copy), `transpose`, `sum`, `relu`
- Fast `matmul`: cache blocking + 4x16 register-blocked micro-kernel + OpenMP; `matmul_naive` kept as reference
- Overflow-checked allocation: impossible sizes throw `std::length_error` / `std::bad_alloc`
- Optional `Tensor::enable_memory_reuse()` (glibc): keeps freed memory in the process to avoid page-fault cost

## Build (Ubuntu 22.04+)
```bash
sudo apt install build-essential cmake git python3-numpy
git clone https://github.com/mahajanparth591-ship-it/mini_tensor-.git
cd mini_tensor-
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```
Options: `-DMT_NATIVE=OFF` (portable binary, not tied to your CPU), `-DMT_SANITIZE=ON` (ASan + UBSan).

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
bash tools/run_all.sh "<Team_ID>"
```
Creates `submission/` with: `functional_results.csv`, `perf_results.csv`, `perf_results_detail.csv`,
`reference_results.csv` (NumPy, single thread), `results.csv` (mandatory columns, merged), `summary.md`,
`machine_info.txt`, `sanitizer_log.txt`. Sources: `tests/common_test.cpp`, `benchmarks/spec_bench.cpp`,
`benchmarks/reference_numpy.py`, `tools/merge_results.py`.

## Benchmarks
Fill this in from `submission/summary.md` after running on your machine
(CPU model, cores, compiler, and `machine_info.txt` go here):

| Case | Student median (ms) | NumPy median (ms) | Ref / student |
|---|---|---|---|
| PB10 matmul 256x256 | TODO | TODO | TODO |
| PB11 matmul 512x512 | TODO | TODO | TODO |
| PB06 add 1000x1000 | TODO | TODO | TODO |

## Design notes
- Row-major layout; offset = sum(index x stride).
- `matmul`: 64x256x256 cache blocks; inside, a 4x16 tile of C is held in vector registers while looping over k.
  OpenMP splits row-blocks across threads (each thread writes different rows: no races); small inputs stay single-threaded.
- Element-wise ops are one flat loop over raw pointers with `__restrict` + `omp simd`; outputs skip the zero-fill pass.
- `reshape()` copies (deep-copy contract); `reshape_()` changes shape/strides in O(1). Reported separately in PB12.
- Negative indices: the API takes unsigned indices, so `-1` wraps to a huge value and is rejected by the bounds check.

## Known limits
No broadcasting, N-D transpose/permute, slicing views, or other dtypes. Allocator tuning is glibc-only.
See `docs/TEST_NOTES.md` for the TC35 specification discrepancy.

## License
MIT
