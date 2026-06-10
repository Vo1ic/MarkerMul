// ================================================================
//  benchmark_multiply.cpp  —  Оцінка MarkerMul::multiply
//
//  Порівнюємо:
//    1. MarkerMul<uint64_t, 2^32>::multiply  — наш алгоритм
//    2. GMP mpz_mul                          — промисловий стандарт
//       (schoolbook → Karatsuba → Toom-Cook → FFT залежно від n)
//
//  Мета: чесно зрозуміти, де MarkerMul виграє/програє і чому.
//
//  Компіляція:
//    g++ -std=c++20 -O3 -Wall -o bench_mul benchmark_multiply.cpp -lgmp
// ================================================================

#include "MarkerMul.h"
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <string>

using FastMul = MarkerMul<uint64_t, (1ULL << 32)>;
constexpr uint64_t BASE = (1ULL << 32);

// ----------------------------------------------------------------
// Таймер (мінімум з кількох запусків для стабільності)
// ----------------------------------------------------------------
template<typename Fn>
static double timeMinMs(Fn&& fn, int repeats = 3) {
    double best = 1e18;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

// ----------------------------------------------------------------
// Генерація випадкового BigNum
// ----------------------------------------------------------------
static FastMul::BigNum randomBigNum(size_t n, std::mt19937_64& rng) {
    FastMul::BigNum v(n);
    for (auto& x : v) x = rng();
    // Маскуємо до 32 біт — щоб відповідало базі 2^32
    for (auto& x : v) x &= 0xFFFFFFFFULL;
    v.back() = std::max<uint64_t>(1, v.back());
    return v;
}

// ----------------------------------------------------------------
// Конвертація FastMul::BigNum → mpz_t (база 2^32)
// ----------------------------------------------------------------
static void bignum_to_mpz(const FastMul::BigNum& v, mpz_t out) {
    mpz_set_ui(out, 0);
    for (int i = (int)v.size() - 1; i >= 0; --i) {
        mpz_mul_2exp(out, out, 32);  // << 32
        mpz_add_ui(out, out, v[i]);
    }
}

// ----------------------------------------------------------------
// Конвертація mpz_t → FastMul::BigNum (база 2^32)
// ----------------------------------------------------------------
static FastMul::BigNum mpz_to_bignum(mpz_t x) {
    FastMul::BigNum result;
    mpz_t tmp, mod;
    mpz_init_set(tmp, x);
    mpz_init(mod);
    mpz_t base_mpz;
    mpz_init(base_mpz);
    mpz_set_str(base_mpz, "4294967296", 10); // 2^32

    while (mpz_sgn(tmp) > 0) {
        mpz_tdiv_qr(tmp, mod, tmp, base_mpz);
        result.push_back((uint64_t)mpz_get_ui(mod));
    }
    if (result.empty()) result.push_back(0);

    mpz_clear(tmp); mpz_clear(mod); mpz_clear(base_mpz);
    return result;
}

// ================================================================
//  Основний бенчмарк
// ================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║    MarkerMul::multiply  vs  GMP mpz_mul                  ║\n";
    std::cout << "║    (честний порівняльний аналіз)                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    std::mt19937_64 rng(2025);

    // Заголовок таблиці
    std::cout << std::left
              << std::setw(8)  << "N limbs"
              << std::setw(10) << "N bits"
              << std::setw(14) << "MarkerMul ms"
              << std::setw(12) << "GMP ms"
              << std::setw(12) << "ratio"
              << "GMP виграє?"
              << "\n" << std::string(70, '─') << "\n";

    // Перевіряємо коректність на малому прикладі
    {
        auto A = FastMul::fromInt(123456789ULL);
        auto B = FastMul::fromInt(987654321ULL);
        auto C_mm = FastMul::multiply(A, B);

        mpz_t ga, gb, gc;
        mpz_inits(ga, gb, gc, nullptr);
        mpz_set_ui(ga, 123456789ULL);
        mpz_set_ui(gb, 987654321ULL);
        mpz_mul(gc, ga, gb);
        auto C_gmp = mpz_to_bignum(gc);
        mpz_clears(ga, gb, gc, nullptr);

        bool ok = FastMul::equals(C_mm, C_gmp);
        std::cout << (ok ? "✅ Коректність підтверджена: 123456789×987654321 збігається\n\n"
                         : "❌ ПОМИЛКА КОРЕКТНОСТІ!\n\n");
    }

    // Розміри від маленьких до великих
    for (size_t N : {10, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000}) {
        auto A = randomBigNum(N, rng);
        auto B = randomBigNum(N, rng);

        // --- MarkerMul ---
        FastMul::BigNum mm_result;
        double mm_ms = timeMinMs([&] {
            mm_result = FastMul::multiply(A, B);
        });

        // --- GMP ---
        mpz_t ga, gb, gc;
        mpz_inits(ga, gb, gc, nullptr);
        bignum_to_mpz(A, ga);
        bignum_to_mpz(B, gb);

        double gmp_ms = timeMinMs([&] {
            mpz_mul(gc, ga, gb);
        });

        // Перевірка збігу результатів
        auto gmp_result = mpz_to_bignum(gc);
        bool match = FastMul::equals(mm_result, gmp_result);
        mpz_clears(ga, gb, gc, nullptr);

        double ratio = mm_ms / gmp_ms;
        size_t bits   = N * 32;

        std::cout << std::fixed << std::setprecision(4)
                  << std::left
                  << std::setw(8)  << N
                  << std::setw(10) << bits
                  << std::setw(14) << mm_ms
                  << std::setw(12) << gmp_ms
                  << std::setw(12) << ratio
                  << (match ? "" : "❌MISMATCH") // завжди має бути пусто
                  << (ratio < 1.2 ? "⚖️  Рівно"
                    : ratio < 3.0 ? "🟡 Трохи"
                    : ratio < 10  ? "🟠 Суттєво"
                                  : "🔴 Сильно")
                  << "\n";
    }

    std::cout << "\n";

    // ----------------------------------------------------------------
    // Аналіз: складність
    // ----------------------------------------------------------------
    std::cout << "─── Аналіз асимптотики (log-log нахил) ─────────────────────\n";
    std::cout << "  MarkerMul: O(n²) — schoolbook convolution\n";
    std::cout << "  GMP переключається:\n";
    std::cout << "    n < ~70 limbs  → asm-оптимізований schoolbook (теж O(n²))\n";
    std::cout << "    n < ~300       → Karatsuba  O(n^1.585)\n";
    std::cout << "    n < ~20000     → Toom-Cook  O(n^1.465)\n";
    std::cout << "    n ≥ ~20000     → Schönhage–Strassen / Harvey-Hoeven  O(n log n log log n)\n\n";

    // Перевіряємо масштабування MarkerMul: ratio між 2N і N
    std::cout << "─── Перевірка O(n²): час(2N)/час(N) ≈ 4? ──────────────────\n";
    for (size_t N : {100, 500, 1000, 2000}) {
        auto A1 = randomBigNum(N, rng);
        auto B1 = randomBigNum(N, rng);
        auto A2 = randomBigNum(2*N, rng);
        auto B2 = randomBigNum(2*N, rng);
        double t1 = timeMinMs([&] { FastMul::multiply(A1, B1); });
        double t2 = timeMinMs([&] { FastMul::multiply(A2, B2); });
        std::cout << "  N=" << std::setw(5) << N
                  << "  t(N)=" << std::setw(10) << t1
                  << "  t(2N)=" << std::setw(10) << t2
                  << "  ratio=" << std::setprecision(2) << t2/t1
                  << (std::abs(t2/t1 - 4.0) < 1.0 ? "  ← ~4x ✓ O(n²)" : "  ← відхилення")
                  << "\n";
    }

    std::cout << "\n─── Висновок ────────────────────────────────────────────────\n";

    return 0;
}
