# MarkerMul: Carry-Free Big-Integer Operations & Partial Product Extraction 🎯

[![Language](https://img.shields.io/badge/Language-C++20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![GPU](https://img.shields.io/badge/GPU-CUDA-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`MarkerMul` is a high-performance C++20 header-only library designed for arbitrary-precision arithmetic. It specializes in **targeted digit extraction** (single digit or range of digits) of the product of two large integers *without* computing the entire product. 

By bounding carry propagation mathematically, `MarkerMul` decouples carry dependencies, enabling high parallelization on GPUs/FPGAs and highly efficient "sniper" digit checking.

---

## 🛑 The Core Problem: Carry Chains
In standard arbitrary-precision multiplication ($O(N^2)$ schoolbook or $O(N \log N)$ FFT), extracting a specific digit at index $k$ of a product $C = A \times B$ traditionally requires computing the entire carry-dependency chain from the least significant digit (LSD) up to $k$. This sequential carry propagation acts as a bottleneck for parallel hardware and introduces high memory overhead when only a small portion of the product is needed.

## 🟢 The Solution: Carry-Depth Bounding
`MarkerMul` utilizes the mathematical guarantee that carry propagation from lower-order digits stabilizes within a predictable window. The maximum depth of carry propagation is bounded by:

$$\text{depth} = \left\lceil \log_{\text{base}} \left( (\text{base} - 1)^2 \cdot L \right) \right\rceil + 1$$

where $L$ is the length of the shorter operand. For binary representation ($\text{base} = 2$), this reduces to:

$$\text{depth} = \lceil \log_2(L) \rceil + 1$$

Using this window, the exact $k$-th digit can be calculated in $O(k \cdot \min(k, n, m))$ time, avoiding the compute and memory overhead of the remainder of the multiplication.

---

## 🚀 Key Features & APIs

### 1. CPU API (`MarkerMul.h`)
The header provides three pre-configured aliases for different bases:
* `FastMul` — Limb type `uint64_t`, base $2^{32}$ (maximum speed, word-based).
* `DecMul`  — Limb type `uint8_t`, base $10$ (for human-readable input/output).
* `BinMul`  — Limb type `uint8_t`, base $2$ (equivalent to former `BinaryMarkerMul`).

#### Core Methods:
* **`digitAt(A, B, k)`**: Returns the exact $k$-th digit of $A \times B$. No `depth` parameter is needed as the library executes a single-pass 0 to $k$ traversal, guaranteeing correctness.
* **`digitsInRange(A, B, first, last)`**: Extracts a contiguous range $[first, last]$ of digits in a single pass. This is significantly faster than calling `digitAt` repeatedly.
* **`checkDigits(A, B, sortedPositions, expected)`**: Efficiently verifies whether the digits at the specified `sortedPositions` match the `expected` values. It features **early-exit** optimization, returning `false` as soon as the first mismatch is encountered. Highly suited for candidate screening algorithms.

### 2. GPU API (`CudaMarkerMul.cuh` / `CudaMarkerMul.cu`)
For full product calculation, the CUDA implementation divides multiplication into a two-phase pipeline:
* **Phase 1 (GPU)**: A parallel convolution kernel (`convKernel`) computes the raw coefficients $raw[k] = \sum A[i] \cdot B[k - i]$ into 128-bit accumulators. Loop bounds are calculated analytically to prevent branching (**warp divergence**).
* **Phase 2 (CPU)**: A sequential carry-propagation pass computes the final limbs. This avoids complex inter-thread dependencies on the GPU.

---

## 💻 Code Examples

### CPU: Extracting and Checking Digits
```cpp
#include "MarkerMul.h"
#include <iostream>
#include <vector>

int main() {
    // Large numbers represented as vectors of limbs (LSB -> MSB)
    FastMul::BigNum A = FastMul::fromInt(123456789ULL);
    FastMul::BigNum B = FastMul::fromInt(987654321ULL);

    // Extract the exact 2nd limb of A * B
    uint64_t limb2 = FastMul::digitAt(A, B, 2);
    std::cout << "Limb #2 of A * B: " << limb2 << std::endl;

    // Verify multiple limbs at once with early-exit
    std::vector<std::size_t> positions = {0, 1, 2};
    std::vector<uint64_t> expected = {1650893339ULL, 27361664ULL, 28ULL}; // known limbs of A*B
    bool matches = FastMul::checkDigits(A, B, positions, expected);
    std::cout << "Verification: " << (matches ? "PASS" : "FAIL") << std::endl;
}
```

### GPU: Full Multiplication
```cpp
#include "CudaMarkerMul.cuh"
#include <iostream>
#include <vector>

int main() {
    std::vector<uint64_t> A = { 4294967295ULL, 1000ULL }; // 32-bit limbs (base 2^32)
    std::vector<uint64_t> B = { 123456ULL,     987654ULL };

    // Execute full multiplication on GPU
    std::vector<uint64_t> result = cudaMultiply(A, B);

    std::cout << "Result limb 0: " << result[0] << std::endl;
}
```

---

## 📊 Benchmarks & Performance

### 1. CPU vs GPU (Full Multiplication, Base $2^{32}$)
Tested on an NVIDIA RTX 5070 GPU (sm_120) comparing CUDA-accelerated `cudaMultiply` against CPU-based `FastMul::multiply` for varying operand lengths ($N$ limbs):

| $N$ Limbs | Equivalent Bits | CPU Time (ms) | GPU Time (ms) | Speedup |
|-----------|-----------------|---------------|---------------|---------|
| 1,000     | 32,000          | 1.34 ms       | 0.11 ms       | ~12x    |
| 5,000     | 160,000         | 33.25 ms      | 0.38 ms       | ~87x    |
| 10,000    | 320,000         | 134.12 ms     | 1.05 ms       | ~127x   |
| 20,000    | 640,000         | 538.90 ms     | 3.52 ms       | ~153x   |

### 2. Sniper Screening (candidateCheck / digitAt)
In search-based applications (e.g., cryptographic sieves, factoring candidates), checking the correctness of a multiplier candidate $A'$ via `candidateCheck` is extremely fast. Since a single word in `FastMul` is 32 bits, checking the lowest-order digit filters out $99.99999997\%$ of incorrect candidates with $O(1)$ cost (1 multiplication of limb 0), bypassing full $O(N^2)$ multiplication.

---

## 🛠️ Build and Verification

### Requirements
* C++20 compiler (`g++` 10+ or `clang` 12+)
* CUDA Toolkit 12.0+ (for GPU benchmark)
* GMP library (optional, for comparing against `benchmark_multiply.cpp`)

### 1. Compile and Run CPU Tests
```bash
g++ -std=c++20 -O3 -Wall -Wextra test_markermul.cpp -o test_markermul
./test_markermul
```

### 2. Compile and Run GPU Benchmarks
```bash
nvcc -O3 -std=c++20 -arch=sm_120 CudaMarkerMul.cu main.cu -o gpu_benchmark
./gpu_benchmark 10000
```

### 3. Compile and Run Algorithmic Sniper Benchmarks
```bash
g++ -std=c++20 -O3 -Wall -Wextra benchmark_sniper.cpp -o bench_sniper
./bench_sniper
```
