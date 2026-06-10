// Тест: MarkerMul (шаблонна версія)
// Компіляція: g++ -std=c++20 -O3 -Wall -Wextra -o test_markermul test_markermul.cpp

#include "MarkerMul.h"
#include "BinaryMarkerMul.h"
#include <cassert>
#include <iostream>
#include <string>

// ----------------------------------------------------------------
// Утиліта: перевіряємо що multiply і digitAt дають однаковий результат
// ----------------------------------------------------------------
template<typename MM>
void verifyConsistency(const char* label,
                       const typename MM::BigNum& A,
                       const typename MM::BigNum& B,
                       uint64_t base = MM::kBase)
{
    auto full = MM::multiply(A, B, base);
    std::size_t L          = std::min(A.size(), B.size());
    int         depth_base = MM::calculateDepth(L, base);

    bool ok = true;
    for (std::size_t k = 0; k < full.size(); ++k) {
        // Для позиції k: depth має бути >= k+1, щоб carry-ланцюг
        // стартував від pos=0 (carry=0) і доходив до pos=k.
        int depth = std::max(depth_base, static_cast<int>(k) + 1);
        auto d = MM::digitAt(A, B, k, depth, base);
        if (d != full[k]) {
            std::cerr << "[" << label << "] MISMATCH at k=" << k
                      << "  full=" << (uint64_t)full[k]
                      << "  digitAt=" << (uint64_t)d << "\n";
            ok = false;
        }
    }
    if (ok)
        std::cout << "[" << label << "] OK  (digits=" << full.size() << ")\n";
}

int main() {
    // ============================================================
    //  1. FastMul (база 2^32)
    // ============================================================
    {
        // 12345678 * 87654321
        auto A = FastMul::fromInt(12345678ULL);
        auto B = FastMul::fromInt(87654321ULL);
        auto C = FastMul::multiply(A, B);
        auto expected = FastMul::fromInt(12345678ULL * 87654321ULL);
        assert(FastMul::equals(C, expected));
        std::cout << "[FastMul] 12345678 * 87654321 = ";
        FastMul::print(C);
        verifyConsistency<FastMul>("FastMul", A, B);
    }

    // ============================================================
    //  2. DecMul (база 10, десяткові цифри)
    // ============================================================
    {
        // 999 * 999 = 998001
        auto A = DecMul::fromInt(999, 10);
        auto B = DecMul::fromInt(999, 10);
        auto C = DecMul::multiply(A, B, 10);
        auto expected = DecMul::fromInt(998001, 10);
        assert(DecMul::equals(C, expected));
        std::cout << "[DecMul] 999 * 999 = ";
        DecMul::print(C);   // 1 0 0 8 9 9  (LSB→MSB у векторі, print→MSB→LSB)
        verifyConsistency<DecMul>("DecMul", A, B, 10);
    }

    // ============================================================
    //  3. BinMul / BinaryMarkerMul (база 2)
    // ============================================================
    {
        // 13 (1101₂) * 11 (1011₂) = 143 (10001111₂)
        auto A = BinMul::fromInt(13, 2);
        auto B = BinMul::fromInt(11, 2);
        auto C = BinMul::multiply(A, B, 2);
        auto expected = BinMul::fromInt(143, 2);
        assert(BinMul::equals(C, expected));
        std::cout << "[BinMul] 13 * 11 = ";
        BinMul::print(C);
        verifyConsistency<BinMul>("BinMul", A, B, 2);

        // Перевіряємо, що BinaryMarkerMul — це той самий тип
        static_assert(std::is_same_v<BinMul, BinaryMarkerMul>,
                      "BinaryMarkerMul must equal BinMul");
    }

    // ============================================================
    //  4. candidateCheck
    // ============================================================
    {
        // DecMul: 12 * 34 = 408
        // Перевіряємо розряди: [0]=8, [1]=0, [2]=4
        auto A = DecMul::fromInt(12, 10);
        auto B = DecMul::fromInt(34, 10);
        std::vector<std::size_t> pos    = {0, 1, 2};
        std::vector<DecMul::Limb> vals  = {8, 0, 4};
        bool ok = DecMul::candidateCheck(A, B, pos, vals, 10);
        assert(ok);
        std::cout << "[candidateCheck] 12*34 digits check: OK\n";
    }

    // ============================================================
    //  5. Великий тест: FastMul з великими числами
    // ============================================================
    {
        uint64_t x = (1ULL << 31) - 1;
        auto A = FastMul::fromInt(x);
        auto B = FastMul::fromInt(x);
        auto C = FastMul::multiply(A, B);
        auto expected = FastMul::fromInt(x * x);
        assert(FastMul::equals(C, expected));
        std::cout << "[FastMul] (2^31-1)^2 = ";
        FastMul::print(C);
        verifyConsistency<FastMul>("FastMul-large", A, B);
    }

    std::cout << "\nAll tests passed ✓\n";
    return 0;
}
