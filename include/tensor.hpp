#pragma once
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <new>
#include <cstdint>
#include <limits>
#include <initializer_list>
#ifdef __GLIBC__
#include <malloc.h>
#endif

class Tensor {
    std::vector<size_t> shape_, strides_;
    size_t size_=0;
    float* data_ = nullptr;

    void computeStrides() {
        strides_.assign(shape_.size(), 1);
        for (int i = (int)shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    void allocate() {
        size_t bytes = ((size_ * sizeof(float) + 63) / 64) * 64;
        if (bytes == 0) bytes = 64;
        data_ = static_cast<float*>(std::aligned_alloc(64, bytes));
        if (!data_) throw std::bad_alloc();
    }


    // Computes element count with overflow checks, then strides, then allocates.
    void init() {
        size_ = 1;
        for (auto d : shape_) {
            if (d != 0 && size_ > std::numeric_limits<size_t>::max() / d)
                throw std::length_error("tensor size overflows size_t");
            size_ *= d;
        }
        if (size_ > (std::numeric_limits<size_t>::max() / sizeof(float)) - 128)
            throw std::length_error("tensor byte size overflows size_t");
        computeStrides();
        allocate();
    }

    // 4x16 micro-kernel written with GCC/Clang vector types (8 floats per register) so that the C tile
    // stays in registers; works with SSE/AVX/AVX-512/NEON, the compiler picks the instructions.
    typedef float v8f __attribute__((vector_size(32), aligned(4), may_alias));
    static inline void micro4x16(const float* __restrict a, size_t lda, const float* __restrict b, size_t ldb,
                                 float* __restrict c, size_t ldc, size_t kc) {
        v8f c00 = *(const v8f*)(c),           c01 = *(const v8f*)(c + 8);
        v8f c10 = *(const v8f*)(c + ldc),     c11 = *(const v8f*)(c + ldc + 8);
        v8f c20 = *(const v8f*)(c + 2 * ldc), c21 = *(const v8f*)(c + 2 * ldc + 8);
        v8f c30 = *(const v8f*)(c + 3 * ldc), c31 = *(const v8f*)(c + 3 * ldc + 8);
        for (size_t k = 0; k < kc; ++k) {
            const v8f b0 = *(const v8f*)(b + k * ldb), b1 = *(const v8f*)(b + k * ldb + 8);
            float av = a[k];            c00 += av * b0; c01 += av * b1;
            av = a[lda + k];            c10 += av * b0; c11 += av * b1;
            av = a[2 * lda + k];        c20 += av * b0; c21 += av * b1;
            av = a[3 * lda + k];        c30 += av * b0; c31 += av * b1;
        }
        *(v8f*)(c) = c00;           *(v8f*)(c + 8) = c01;
        *(v8f*)(c + ldc) = c10;     *(v8f*)(c + ldc + 8) = c11;
        *(v8f*)(c + 2 * ldc) = c20; *(v8f*)(c + 2 * ldc + 8) = c21;
        *(v8f*)(c + 3 * ldc) = c30; *(v8f*)(c + 3 * ldc + 8) = c31;
    }

    void checkSame(const Tensor& o) const {
        if (shape_ != o.shape_) throw std::invalid_argument("shape mismatch");
    }

    static void checkMatmul(const Tensor& A, const Tensor& B) {
        if (A.shape_.size() != 2 || B.shape_.size() != 2 || A.shape_[1] != B.shape_[0])
            throw std::invalid_argument("matmul: need 2D tensors with A.cols == B.rows");
    }

    Tensor binary(const Tensor& o, int op) const {
        checkSame(o);
        Tensor r = empty(shape_);
        const float* __restrict__ a = data_;
        const float* __restrict__ b = o.data_;
        float* __restrict__ c = r.data_;
        const size_t n = size_;

        if (op == 0) {
            #pragma omp parallel for simd if (n > 200000)
            for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
        } else if (op == 1) {
            #pragma omp parallel for simd if (n > 200000)
            for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
        } else {
            #pragma omp parallel for simd if (n > 200000)
            for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
        }
        return r;
    }

public:
    // OPTIONAL performance switch (glibc only, process-wide): keep freed memory in the process instead of
    // returning it to the OS, so repeated create/destroy of big tensors avoids page-fault cost.
    // Returns true if the allocator was tuned. Call once at program start; trades RSS for speed.
    static bool enable_memory_reuse() {
#ifdef __GLIBC__
        bool ok = mallopt(M_MMAP_THRESHOLD, 1 << 30) != 0;
        ok = (mallopt(M_TRIM_THRESHOLD, 1 << 30) != 0) && ok;
        mallopt(M_TOP_PAD, 64 << 20);
        return ok;
#else
        return false;
#endif
    }

    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
        init();
        std::memset(data_, 0, size_ * sizeof(float));
    }
    // Create from explicit row-major data. Shape {} makes a rank-0 scalar (1 element).
    Tensor(std::vector<size_t> shape, const std::vector<float>& values) : shape_(std::move(shape)) {
        init();
        if (values.size() != size_) throw std::invalid_argument("data size does not match shape");
        if (size_) std::memcpy(data_, values.data(), size_ * sizeof(float));
    }
    // Uninitialised tensor (fast path; caller must write every element).
    static Tensor empty(std::vector<size_t> shape) {
        Tensor t; t.shape_ = std::move(shape); t.init(); return t;
    }
    static Tensor full(std::vector<size_t> shape, float v) {
        Tensor t = empty(std::move(shape)); t.fill(v); return t;
    }
    ~Tensor() {
        std::free(data_);
    }

    Tensor(const Tensor& o) : shape_(o.shape_), strides_(o.strides_), size_(o.size_) {
        allocate();
        if (size_) std::memcpy(data_, o.data_, size_ * sizeof(float));
    }

    Tensor(Tensor&& o) noexcept
        : shape_(std::move(o.shape_)), strides_(std::move(o.strides_)),
          size_(o.size_), data_(o.data_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }

    void swap(Tensor& o) noexcept {
        std::swap(shape_, o.shape_);
        std::swap(strides_, o.strides_);
        std::swap(size_, o.size_);
        std::swap(data_, o.data_);
    }

    Tensor& operator=(Tensor o) noexcept {
        swap(o);
        return *this;
    }

    // Static factories
    static Tensor zeros(std::vector<size_t> s) { return Tensor(std::move(s)); }
    static Tensor ones(std::vector<size_t> s) { return full(std::move(s), 1.0f); }
    static Tensor random(std::vector<size_t> s, unsigned seed = 42) {
        Tensor t(std::move(s));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> d(0.0f, 1.0f);
        for (size_t i = 0; i < t.size_; ++i) t.data_[i] = d(gen);
        return t;
    }

    void fill(float v) {
        #pragma omp parallel for simd if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) data_[i] = v;
    }

    // Getters & Indexing
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    float* data() { return data_; }
    const float* data() const { return data_; }

    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }

    
    float& at(std::initializer_list<size_t> idx) { return data_[offset(idx)]; }
    const float& at(std::initializer_list<size_t> idx) const { return data_[offset(idx)]; }

    size_t offset(std::initializer_list<size_t> idx) const {
        if (idx.size() != shape_.size()) throw std::invalid_argument("wrong number of indices");
        size_t off = 0, d = 0;
        for (auto i : idx) {
            if (i >= shape_[d]) throw std::out_of_range("index out of range");
            off += i * strides_[d++];
        }
        return off;
    }

    // Math operations
    Tensor operator+(const Tensor& o) const { return binary(o, 0); }
    Tensor operator-(const Tensor& o) const { return binary(o, 1); }
    Tensor mul(const Tensor& o) const { return binary(o, 2); }

    Tensor operator*(float s) const {
        Tensor r = empty(shape_);
        #pragma omp parallel for simd if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] * s;
        return r;
    }

    Tensor operator+(float s) const {
        Tensor r = empty(shape_);
        #pragma omp parallel for simd if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] + s;
        return r;
    }

    void add_(const Tensor& o) {
        checkSame(o);
        #pragma omp parallel for simd if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) data_[i] += o.data_[i];
    }

    // Phase 4: Reshape, Transpose, Sum, ReLU
    Tensor reshape(std::vector<size_t> ns) const {
        size_t n = 1; for (auto s : ns) n *= s;
        if (n != size_) throw std::invalid_argument("reshape: total size must stay the same");
        Tensor t(*this);
        t.shape_ = std::move(ns);
        t.computeStrides();
        return t;
    }

    // In-place reshape: no data is copied or moved (O(1)). Element order is preserved.
    void reshape_(std::vector<size_t> ns) {
        size_t n = 1; for (auto d : ns) n *= d;
        if (n != size_) throw std::invalid_argument("reshape: total size must stay the same");
        shape_ = std::move(ns);
        computeStrides();
    }

    Tensor transpose() const {
        if (shape_.size() != 2) throw std::invalid_argument("transpose needs a 2D tensor");
        size_t R = shape_[0], C = shape_[1];
        Tensor t = empty({C, R});
        const size_t B = 32;
        for (size_t ii = 0; ii < R; ii += B) {
            for (size_t jj = 0; jj < C; jj += B) {
                for (size_t i = ii; i < std::min(ii + B, R); ++i) {
                    for (size_t j = jj; j < std::min(jj + B, C); ++j) {
                        t.data_[j * R + i] = data_[i * C + j];
                    }
                }
            }
        }
        return t;
    }

    float sum() const {
        double s = 0;
        #pragma omp parallel for reduction(+:s) if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) s += data_[i];
        return (float)s;
    }

    Tensor relu() const {
        Tensor r = empty(shape_);
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] > 0 ? data_[i] : 0;
        return r;
    }

    static Tensor matmul_naive(const Tensor& A, const Tensor& B) {
        checkMatmul(A, B);
        size_t M = A.shape_[0], K = A.shape_[1], N = B.shape_[1];
        Tensor C({M, N});
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float s = 0;
                for (size_t k = 0; k < K; ++k) s += A.data_[i * K + k] * B.data_[k * N + j];
                C.data_[i * N + j] = s;
            }
        }
        return C;
    }

    // Fast version: cache blocking + 4x16 register-blocked micro-kernel + OpenMP over row blocks.
    // The micro-kernel keeps a 4x16 tile of C in registers while looping over k, so C is read/written
    // once per tile instead of once per k (this is what makes it several times faster than plain i-k-j).
    static Tensor matmul(const Tensor& A, const Tensor& B) {
        checkMatmul(A, B);
        const size_t M = A.shape_[0], K = A.shape_[1], N = B.shape_[1];
        Tensor C({M, N});                                   // zero-initialised: the kernel accumulates
        const float* __restrict a = A.data_;
        const float* __restrict b = B.data_;
        float* __restrict c = C.data_;
        const size_t MC = 64, KC = 256, NC = 256;
        #pragma omp parallel for schedule(static) if((double)M * N * K >= 1e6)
        for (size_t ii = 0; ii < M; ii += MC) {
            const size_t iMax = std::min(ii + MC, M);
            for (size_t kk = 0; kk < K; kk += KC) {
                const size_t kMax = std::min(kk + KC, K), kc = kMax - kk;
                for (size_t jj = 0; jj < N; jj += NC) {
                    const size_t jMax = std::min(jj + NC, N);
                    size_t i = ii;
                    for (; i + 4 <= iMax; i += 4) {
                        size_t j = jj;
                        for (; j + 16 <= jMax; j += 16)
                            micro4x16(a + i * K + kk, K, b + kk * N + j, N, c + i * N + j, N, kc);
                        for (; j < jMax; ++j)               // leftover columns
                            for (size_t r = 0; r < 4; ++r) {
                                float s = 0;
                                for (size_t k = kk; k < kMax; ++k) s += a[(i + r) * K + k] * b[k * N + j];
                                c[(i + r) * N + j] += s;
                            }
                    }
                    for (; i < iMax; ++i)                   // leftover rows
                        for (size_t k = kk; k < kMax; ++k) {
                            const float av = a[i * K + k];
                            for (size_t j = jj; j < jMax; ++j) c[i * N + j] += av * b[k * N + j];
                        }
                }
            }
        }
        return C;
    }

    void print() const {
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) std::cout << shape_[i] << (i + 1 < shape_.size() ? "," : "");
        std::cout << "]) data=[";
        for (size_t i = 0; i < std::min<size_t>(size_, 20); ++i) std::cout << data_[i] << " ";
        std::cout << (size_ > 20 ? "..." : "") << "]\n";
    }
};
