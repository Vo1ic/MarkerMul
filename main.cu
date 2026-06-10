// ============================================================================
//  main.cu  —  Benchmark: CPU FastMul vs GPU cudaMultiply
//  Compilation: nvcc -O3 -std=c++20 -arch=sm_120 CudaMarkerMul.cu main.cu \
//                     -o gpu_benchmark
// ============================================================================

#include "MarkerMul.h"
#include "CudaMarkerMul.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using FastMul = MarkerMul<uint64_t, (1ULL << 32)>;

// Generates a random BigNum with non-zero MSB.
static FastMul::BigNum randomBigNum(std::size_t n, uint64_t base, std::mt19937_64& rng) {
    FastMul::BigNum result(n);
    std::uniform_int_distribution<uint64_t> dist(0, base - 1);
    for (auto& limb : result) limb = dist(rng);
    result.back() = std::max<uint64_t>(1, result.back());
    return result;
}

// Timer helper.
template<typename Fn>
static double timeMs(Fn&& fn) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    std::size_t N = (argc > 1) ? (std::size_t)std::atoll(argv[1]) : 5000;
    constexpr uint64_t BASE = (1ULL << 32);

    std::cout << "============================================\n";
    std::cout << "  MarkerMul: CPU vs GPU Benchmark\n";
    std::cout << "  N = " << N << " limbs (" << N * 32 << " bits)\n";
    std::cout << "  Base: 2^32\n";
    std::cout << "============================================\n\n";

    std::mt19937_64 rng(42);
    auto A = randomBigNum(N, BASE, rng);
    auto B = randomBigNum(N, BASE, rng);

    // 1. CPU multiplication
    std::vector<uint64_t> cpu_result;
    double cpu_ms = timeMs([&] {
        cpu_result = FastMul::multiply(A, B);
    });

    std::cout << "[CPU] FastMul<uint64_t, 2^32>::multiply\n";
    std::cout << "  Time:      " << cpu_ms << " ms\n";
    std::cout << "  Limbs:     " << cpu_result.size() << "\n\n";

    // 2. GPU multiplication (including device allocations/transfers)
    {
        std::vector<uint64_t> warmup_A = {1}, warmup_B = {1};
        cudaMultiply(warmup_A, warmup_B, BASE);
    }

    std::vector<uint64_t> gpu_result;
    double gpu_ms = timeMs([&] {
        gpu_result = cudaMultiply(A, B, BASE);
    });

    std::cout << "[GPU] cudaMultiply (convKernel + CPU carry prop)\n";
    std::cout << "  Time:      " << gpu_ms << " ms\n";
    std::cout << "  Limbs:     " << gpu_result.size() << "\n\n";

    // 3. Verification
    bool match = (cpu_result.size() == gpu_result.size());
    if (match) {
        for (std::size_t i = 0; i < cpu_result.size(); ++i) {
            if (cpu_result[i] != gpu_result[i]) {
                std::cerr << "[MISMATCH] Limb " << i << ": CPU=" << cpu_result[i]
                          << " GPU=" << gpu_result[i] << "\n";
                match = false;
                break;
            }
        }
    } else {
        std::cerr << "[MISMATCH] Limb count mismatch: CPU=" << cpu_result.size()
                  << " GPU=" << gpu_result.size() << "\n";
    }

    if (match) {
        std::cout << "✅ CPU and GPU results match!\n\n";
    } else {
        std::cerr << "❌ Results mismatch!\n\n";
        return 1;
    }

    // 4. Statistics
    std::cout << "============================================\n";
    std::cout << "  GPU vs CPU Speedup: " << cpu_ms / gpu_ms << "x\n";
    std::cout << "============================================\n";

    return 0;
}
