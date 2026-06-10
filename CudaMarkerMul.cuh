// ============================================================================
//  CudaMarkerMul.cuh  —  GPU-based MarkerMul multiplication (Interface)
//  (CUDA C++, Header-only declarations)
//
//  Design: Two-phase multiplication:
//    - Phase 1: Parallel GPU convolution. Each thread k calculates the
//               raw convolution sum raw_sum[k] = sum_{i} A[i]*B[k-i] in 128-bit.
//    - Phase 2: Sequential carry propagation on CPU.
//
//  Technical notes:
//    - Standard native __int128 is not hardware-supported on GPU by NVCC.
//      We use a custom uint128_t struct with carry detection.
// ============================================================================
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// 128-bit unsigned integer representation for GPU/CPU accumulator.
struct uint128_t {
    uint64_t lo; // Low 64 bits
    uint64_t hi; // High 64 bits
};

// Phase 1 CUDA kernel: Computes raw convolution sum: d_raw[k] = sum_{i} d_A[i]*d_B[k-i].
// Bounds are calculated mathematically to avoid branching/warp divergence.
__global__ void convKernel(
    const uint64_t* __restrict__ d_A,
    const uint64_t* __restrict__ d_B,
    size_t n,
    size_t m,
    uint128_t*      d_raw,
    size_t          result_size
);

// Host wrapper for full GPU-based multiplication A * B.
// Manages device allocations, host-to-device/device-to-host copies, grid
// configuration, kernel execution, and host-side carry propagation.
std::vector<uint64_t> cudaMultiply(
    const std::vector<uint64_t>& A,
    const std::vector<uint64_t>& B,
    uint64_t base = (1ULL << 32)
);
