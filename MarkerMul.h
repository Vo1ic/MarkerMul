#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace markermul_detail {

// Selects the appropriate accumulator type to avoid overflow during convolution sum.
template<typename LimbT>
struct AccumType {
    using type = unsigned __int128;
};
template<> struct AccumType<uint8_t>  { using type = uint64_t; };
template<> struct AccumType<uint16_t> { using type = uint64_t; };
template<> struct AccumType<uint32_t> { using type = uint64_t; };

} // namespace markermul_detail

// Template class for arbitrary-precision arithmetic operations.
template<typename LimbT = uint64_t, uint64_t BASE = (1ULL << 32)>
class MarkerMul {
    static_assert(BASE >= 2, "BASE must be at least 2");

public:
    using Limb   = LimbT;
    using BigNum = std::vector<Limb>;
    using Accum  = typename markermul_detail::AccumType<LimbT>::type;

    static constexpr uint64_t kBase = BASE;

    // Calculates the maximum carry propagation depth for the given operand length L.
    // depth = ceil( log_base( (base - 1)^2 * L ) ) + 1.
    // Uses fast bitwise operations if the base is a power of two.
    static int calculateDepth(std::size_t L, uint64_t base = kBase) noexcept {
        if (L == 0) return 1;

        unsigned __int128 max_sum =
            (unsigned __int128)(base - 1) * (base - 1) * (unsigned __int128)L;

        if (max_sum == 0) return 1;

        if ((base & (base - 1)) == 0) {
            int bits_per_limb = std::bit_width(base) - 1; // log2(base)
            int total_bits = bit_width128(max_sum);
            return (total_bits + bits_per_limb - 1) / bits_per_limb + 1;
        }

        int depth = 0;
        while (max_sum > 0) {
            max_sum /= base;
            ++depth;
        }
        return depth + 1;
    }

    // Computes full product of A and B using O(N^2) schoolbook multiplication.
    // Loop bounds are pre-calculated to avoid branching.
    static BigNum multiply(const BigNum& A, const BigNum& B,
                           uint64_t base = kBase) {
        const std::size_t n = A.size(), m = B.size();
        BigNum result(n + m, Limb{0});

        for (std::size_t k = 0; k < n + m - 1; ++k) {
            const std::size_t i_start = (k + 1 > m) ? (k + 1 - m) : 0;
            const std::size_t i_end   = std::min(k, n - 1);

            Accum sum = static_cast<Accum>(result[k]);
            for (std::size_t i = i_start; i <= i_end; ++i) {
                sum += static_cast<Accum>(A[i]) * static_cast<Accum>(B[k - i]);
            }
            result[k]     = static_cast<Limb>(sum % base);
            result[k + 1] += static_cast<Limb>(sum / base);
        }
        return trim(result);
    }

    // Computes the exact k-th digit of A * B using a single pass carry propagation.
    // Complexity: O(k * min(k, n, m)). No carry depth parameter required.
    static Limb digitAt(const BigNum& A, const BigNum& B,
                        std::size_t k,
                        uint64_t base = kBase) noexcept {
        const std::size_t n = A.size(), m = B.size();
        if (n == 0 || m == 0 || k >= n + m) return Limb{0};

        Accum carry = 0;
        Limb  digit = 0;

        for (std::size_t pos = 0; pos <= k; ++pos) {
            const std::size_t i_start = (pos + 1 > m) ? (pos + 1 - m) : 0;
            const std::size_t i_end   = std::min(pos, n - 1);

            Accum sum = carry;
            if (i_start <= i_end) {
                for (std::size_t i = i_start; i <= i_end; ++i)
                    sum += static_cast<Accum>(A[i]) * static_cast<Accum>(B[pos - i]);
            }
            digit = static_cast<Limb>(sum % base);
            carry = sum / base;
        }
        return digit;
    }

    // Deprecated overload kept for backward compatibility. Parameter depth is ignored.
    [[deprecated("Use digitAt(A, B, k, base) instead")]]
    static Limb digitAt(const BigNum& A, const BigNum& B,
                        std::size_t k, int /*depth*/,
                        uint64_t base = kBase) noexcept {
        return digitAt(A, B, k, base);
    }

    // Extracts digits of A * B in range [first, last] inclusive.
    // More efficient than multiple digitAt calls due to single-pass carry propagation.
    static BigNum digitsInRange(const BigNum& A, const BigNum& B,
                                std::size_t first, std::size_t last,
                                uint64_t base = kBase) {
        if (first > last) return {};
        const std::size_t n = A.size(), m = B.size();
        const std::size_t result_limit = (n + m > 0) ? (n + m - 1) : 0;
        last = std::min(last, result_limit);

        BigNum result;
        result.reserve(last - first + 1);

        Accum carry = 0;
        for (std::size_t pos = 0; pos <= last; ++pos) {
            const std::size_t i_start = (pos + 1 > m) ? (pos + 1 - m) : 0;
            const std::size_t i_end   = std::min(pos, n - 1);

            Accum sum = carry;
            if (i_start <= i_end) {
                for (std::size_t i = i_start; i <= i_end; ++i)
                    sum += static_cast<Accum>(A[i]) * static_cast<Accum>(B[pos - i]);
            }
            if (pos >= first)
                result.push_back(static_cast<Limb>(sum % base));
            carry = sum / base;
        }
        return result;
    }

    // Checks if digits at sortedPositions equal the expected values.
    // Early-exit optimization: returns false on the first mismatch.
    // sortedPositions MUST be sorted in ascending order.
    static bool checkDigits(const BigNum&                   A,
                            const BigNum&                   B,
                            const std::vector<std::size_t>& sortedPositions,
                            const std::vector<Limb>&        expected,
                            uint64_t base = kBase) {
        if (sortedPositions.empty()) return true;
        const std::size_t n   = A.size(), m = B.size();
        const std::size_t end = sortedPositions.back();

        Accum       carry   = 0;
        std::size_t chk_idx = 0;

        for (std::size_t pos = 0; pos <= end; ++pos) {
            const std::size_t i_start = (pos + 1 > m) ? (pos + 1 - m) : 0;
            const std::size_t i_end   = std::min(pos, n - 1);

            Accum sum = carry;
            if (i_start <= i_end) {
                for (std::size_t i = i_start; i <= i_end; ++i)
                    sum += static_cast<Accum>(A[i]) * static_cast<Accum>(B[pos - i]);
            }
            const Limb digit = static_cast<Limb>(sum % base);
            carry = sum / base;

            if (sortedPositions[chk_idx] == pos) {
                if (digit != expected[chk_idx]) return false;
                if (++chk_idx == sortedPositions.size()) return true;
            }
        }
        return true;
    }

    // Backward compatibility aliases.
    static bool candidateCheck(const BigNum&                   A,
                               const BigNum&                   B,
                               const std::vector<std::size_t>& positions,
                               const std::vector<Limb>&        values,
                               uint64_t base = kBase) {
        const std::size_t K = positions.size();
        std::vector<std::size_t> idx(K);
        for (std::size_t i = 0; i < K; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b){ return positions[a] < positions[b]; });

        std::vector<std::size_t> sorted_pos(K);
        std::vector<Limb>        sorted_val(K);
        for (std::size_t i = 0; i < K; ++i) {
            sorted_pos[i] = positions[idx[i]];
            sorted_val[i] = values[idx[i]];
        }
        return checkDigits(A, B, sorted_pos, sorted_val, base);
    }

    static bool multiDigitCheck(const BigNum&                   A,
                                const BigNum&                   B,
                                const std::vector<std::size_t>& sortedPositions,
                                const std::vector<Limb>&        knownValues,
                                uint64_t base = kBase) {
        return checkDigits(A, B, sortedPositions, knownValues, base);
    }

    static bool orderedCandidateCheck(const BigNum&                   A,
                                      const BigNum&                   B,
                                      const std::vector<std::size_t>& positions,
                                      const std::vector<Limb>&        values,
                                      uint64_t base = kBase) {
        return candidateCheck(A, B, positions, values, base);
    }

    // Prints the BigNum from MSB to LSB.
    static void print(const BigNum& A) {
        for (int i = static_cast<int>(A.size()) - 1; i >= 0; --i) {
            std::cout << static_cast<uint64_t>(A[i]);
            if (i > 0) std::cout << ' ';
        }
        std::cout << '\n';
    }

    // Constructs a BigNum from a 64-bit integer.
    static BigNum fromInt(uint64_t x, uint64_t base = kBase) {
        BigNum A;
        if (x == 0) return {Limb{0}};
        while (x > 0) {
            A.push_back(static_cast<Limb>(x % base));
            x /= base;
        }
        return A;
    }

    // Checks if two BigNum values are equal (ignores leading zeros).
    static bool equals(const BigNum& A, const BigNum& B) noexcept {
        return trim(A) == trim(B);
    }

private:
    // Trims leading zero limbs.
    static BigNum trim(BigNum A) {
        while (A.size() > 1 && A.back() == Limb{0})
            A.pop_back();
        return A;
    }

    // Custom bit width implementation for 128-bit unsigned integers.
    static int bit_width128(unsigned __int128 x) noexcept {
        if (x == 0) return 0;
        const uint64_t hi = static_cast<uint64_t>(x >> 64);
        const uint64_t lo = static_cast<uint64_t>(x);
        if (hi) return 64 + (64 - __builtin_clzll(hi));
        return          64 - __builtin_clzll(lo);
    }
};

// Fast implementation: 64-bit limbs with base 2^32.
using FastMul = MarkerMul<uint64_t, (1ULL << 32)>;

// Decimal base implementation (primarily for verification and readable output).
using DecMul  = MarkerMul<uint8_t, 10>;

// Binary base implementation.
using BinMul  = MarkerMul<uint8_t, 2>;
