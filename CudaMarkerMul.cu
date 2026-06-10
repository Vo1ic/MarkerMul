// ============================================================================
//  CudaMarkerMul.cu  —  GPU-based MarkerMul multiplication (Implementation)
//  Compilation: nvcc -O3 -arch=sm_120 CudaMarkerMul.cu main.cu -o gpu_benchmark
// ============================================================================

#include "CudaMarkerMul.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <vector>

// 128-bit addition: dst += val
__device__ __forceinline__
void add128(uint128_t& dst, uint128_t val) {
    dst.lo += val.lo;
    dst.hi += val.hi + (dst.lo < val.lo ? 1ULL : 0ULL); // carry detection
}

// 64-bit x 64-bit -> 128-bit multiplication.
// Uses native multiplication for low 64 bits and __umul64hi for high 64 bits.
__device__ __forceinline__
uint128_t mul64(uint64_t a, uint64_t b) {
    return {
        a * b,
        __umul64hi(a, b) // PTX mul.hi.u64 instruction
    };
}

// Phase 1: GPU convolution kernel.
// Each thread k computes the raw convolution coefficient raw[k].
// Loop bounds are precalculated mathematically to avoid warp divergence.
__global__ void convKernel(
    const uint64_t* __restrict__ d_A,
    const uint64_t* __restrict__ d_B,
    size_t n,
    size_t m,
    uint128_t*      d_raw,
    size_t          result_size)
{
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= result_size) return;

    const size_t i_start = (k + 1 > m) ? (k + 1 - m) : 0;
    const size_t i_end   = (k < n - 1) ? k : (n - 1);

    uint128_t sum = {0ULL, 0ULL};

    for (size_t i = i_start; i <= i_end; ++i) {
        add128(sum, mul64(d_A[i], d_B[k - i]));
    }

    d_raw[k] = sum;
}

// Host wrapper for cudaMultiply.
std::vector<uint64_t> cudaMultiply(
    const std::vector<uint64_t>& A,
    const std::vector<uint64_t>& B,
    uint64_t base)
{
    const size_t n = A.size(), m = B.size();
    const size_t result_size = n + m - 1;
    const size_t final_size  = n + m;

    // 1. Allocate GPU memory
    uint64_t  *d_A, *d_B;
    uint128_t *d_raw;

    cudaMalloc(&d_A,  n * sizeof(uint64_t));
    cudaMalloc(&d_B,  m * sizeof(uint64_t));
    cudaMalloc(&d_raw, result_size * sizeof(uint128_t));

    // 2. Host-to-Device copy
    cudaMemcpy(d_A, A.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), m * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 3. Grid configuration and kernel launch
    const int THREADS = 256;
    const int BLOCKS  = (int)((result_size + THREADS - 1) / THREADS);

    convKernel<<<BLOCKS, THREADS>>>(d_A, d_B, n, m, d_raw, result_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[cudaMultiply] kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_raw);
        return {};
    }

    // 4. Device-to-Host copy
    std::vector<uint128_t> raw(result_size);
    cudaMemcpy(raw.data(), d_raw, result_size * sizeof(uint128_t), cudaMemcpyDeviceToHost);

    // 5. Host carry propagation
    std::vector<uint64_t> result(final_size, 0ULL);
    uint64_t carry_lo = 0, carry_hi = 0;

    for (size_t k = 0; k < result_size; ++k) {
        uint64_t sum_lo = raw[k].lo + carry_lo;
        uint64_t sum_hi = raw[k].hi + carry_hi + (sum_lo < carry_lo ? 1ULL : 0ULL);

        if (base == (1ULL << 32)) {
            // Optimized bitwise operations for standard 2^32 base
            result[k] = sum_lo & 0xFFFFFFFFULL;
            carry_lo = (sum_lo >> 32) | (sum_hi << 32);
            carry_hi = sum_hi >> 32;
        } else {
            // General case for custom base
            unsigned __int128 s = ((unsigned __int128)sum_hi << 64) | sum_lo;
            result[k] = (uint64_t)(s % (unsigned __int128)base);
            s /= (unsigned __int128)base;
            carry_lo = (uint64_t)s;
            carry_hi = (uint64_t)(s >> 64);
        }
    }

    result[result_size] = carry_lo;

    while (result.size() > 1 && result.back() == 0)
        result.pop_back();

    // 6. Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_raw);

    return result;
}
