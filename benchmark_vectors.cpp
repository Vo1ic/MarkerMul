// benchmark_vectors.cpp — вимір Вектор #1 та #2
// g++ -std=c++20 -O3 -o bench_vec benchmark_vectors.cpp

#include "MarkerMul.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using DM = MarkerMul<uint8_t, 10>;  // base=10 — цікавіша для ordered

template<typename Fn>
static double timeMs(Fn&& fn, int rep = 5) {
    double best = 1e18;
    for (int r = 0; r < rep; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    std::mt19937_64 rng(42);
    const size_t N = 400;  // лімбів (цифр) в кожному числі
    const size_t N_CANDS = 20000;

    // Генеруємо A, B у base=10
    DM::BigNum A(N), B(N);
    for (auto& x : A) x = rng() % 10;
    for (auto& x : B) x = rng() % 10;
    A.back() = std::max<uint8_t>(1, A.back());
    B.back() = std::max<uint8_t>(1, B.back());

    auto C_ref = DM::multiply(A, B, 10);

    // Тестуємо K розрядів від 1 до 32
    for (size_t K : {1UL, 2UL, 4UL, 8UL, 16UL, 32UL}) {

        // Позиції: рівномірно по результату (не тільки перші)
        std::vector<size_t>       pos(K);
        std::vector<DM::Limb>     vals(K);
        // Беремо перші K позицій — де carry-chain найдешевша
        for (size_t i = 0; i < K; ++i) {
            pos[i]  = i;
            vals[i] = C_ref[i];
        }

        // Кандидати: A з випадковим першим лімбом (більшість відсіється)
        std::vector<DM::BigNum> cands(N_CANDS, A);
        for (auto& c : cands) c[0] = rng() % 10;

        // --- Метод A: candidateCheck (K окремих digitAt) ---
        double t_a = timeMs([&] {
            volatile size_t s = 0;
            for (const auto& c : cands)
                s += DM::candidateCheck(c, B, pos, vals, 10) ? 1 : 0;
        });

        // --- Метод B: multiDigitCheck (один прохід) ---
        double t_b = timeMs([&] {
            volatile size_t s = 0;
            for (const auto& c : cands)
                s += DM::multiDigitCheck(c, B, pos, vals, 10) ? 1 : 0;
        });

        // --- Метод C: orderedCandidateCheck ---
        double t_c = timeMs([&] {
            volatile size_t s = 0;
            for (const auto& c : cands)
                s += DM::orderedCandidateCheck(c, B, pos, vals, 10) ? 1 : 0;
        });

        std::cout << "K=" << std::setw(3) << K
                  << "  candidateCheck: " << std::setw(8) << t_a << " ms"
                  << "  multiDigit: "     << std::setw(8) << t_b << " ms"
                  << " (x" << std::setw(5) << std::setprecision(1)
                  << std::fixed << t_a/t_b << ")"
                  << "  ordered: "        << std::setw(8) << t_c << " ms"
                  << " (x" << std::setw(5) << t_a/t_c << ")"
                  << "\n";
    }

    std::cout << "\n--- Перевірка коректності ---\n";
    std::vector<size_t> pos4 = {0,1,2,3};
    std::vector<DM::Limb> v4 = {C_ref[0], C_ref[1], C_ref[2], C_ref[3]};
    bool r1 = DM::candidateCheck(A, B, pos4, v4, 10);
    bool r2 = DM::multiDigitCheck(A, B, pos4, v4, 10);
    bool r3 = DM::orderedCandidateCheck(A, B, pos4, v4, 10);
    assert(r1 && r2 && r3);

    // Хибний кандидат
    DM::BigNum A_bad = A; A_bad[0] = (A[0] + 5) % 10;
    bool f1 = DM::candidateCheck(A_bad, B, pos4, v4, 10);
    bool f2 = DM::multiDigitCheck(A_bad, B, pos4, v4, 10);
    bool f3 = DM::orderedCandidateCheck(A_bad, B, pos4, v4, 10);
    assert(!f1 && !f2 && !f3);
    std::cout << "Всі методи дають однаковий результат ✓\n";
    return 0;
}
