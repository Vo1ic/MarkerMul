// ================================================================
//  benchmark_sniper.cpp  —  Оцінка практичної цінності digitAt
//
//  Сценарій: "Криптографічне просіювання кандидатів"
//
//  Задача: маємо число T = A × B (відоме — наприклад, публічний модуль RSA).
//          Шукаємо множники серед кандидатів A' (невеликі варіації A).
//          Нам відомо, що правильний A' дасть T, тобто A' × B = T.
//
//  Питання: чи вигідніше для кожного кандидата:
//    [Метод 1] Обчислити повний добуток A'×B і порівняти з T?
//    [Метод 2] Перевірити лише K розрядів через candidateCheck(digitAt)?
//
//  Ключовий ефект: при великій базі (2^32) перший розряд відсіює
//  майже всіх кандидатів з вартістю O(1) — просто один добуток лімбів.
//  Тільки 1 кандидат з ~4 мільярдів пройде далі.
//
//  Компіляція:
//    g++ -std=c++20 -O3 -Wall -Wextra benchmark_sniper.cpp -o bench_sniper
// ================================================================

#include "MarkerMul.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using FastMul = MarkerMul<uint64_t, (1ULL << 32)>;
constexpr uint64_t BASE = (1ULL << 32);

// ----------------------------------------------------------------
// Утиліта: точний таймер
// ----------------------------------------------------------------
template<typename Fn>
static double timeMs(Fn&& fn) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ----------------------------------------------------------------
// Спрощений candidateCheck: ручна реалізація з early-exit
// та підрахунком реальних операцій (для статистики)
// ----------------------------------------------------------------
struct CheckResult {
    bool   matched;
    size_t ops_done;   // скільки доданків A[i]*B[j] реально обчислено
};

CheckResult candidateCheckCounted(
    const FastMul::BigNum& A,
    const FastMul::BigNum& B,
    const std::vector<size_t>&         knownPos,
    const std::vector<FastMul::Limb>&  knownVals)
{
    const size_t n = A.size(), m = B.size();
    const size_t L = std::min(n, m);
    const int    depth_base = FastMul::calculateDepth(L, BASE);
    size_t total_ops = 0;

    for (size_t idx = 0; idx < knownPos.size(); ++idx) {
        const size_t k     = knownPos[idx];
        const int    d_eff = std::max(depth_base, (int)k + 1);

        // digitAt inline з підрахунком операцій
        unsigned __int128 carry = 0;
        uint64_t digit = 0;

        const int pos_start = std::max(0, (int)k - d_eff + 1);
        for (int pos = pos_start; pos <= (int)k; ++pos) {
            auto p = (size_t)pos;
            size_t i_start = (p + 1 > m) ? (p + 1 - m) : 0;
            size_t i_end   = std::min(p, n - 1);

            unsigned __int128 sum = carry;
            for (size_t i = i_start; i <= i_end; ++i) {
                sum += (unsigned __int128)A[i] * B[p - i];
                ++total_ops;
            }
            digit = (uint64_t)(sum % BASE);
            carry = sum / BASE;
        }

        if (digit != knownVals[idx])
            return {false, total_ops}; // ← early exit!
    }
    return {true, total_ops};
}

// ================================================================
//  ТЕСТ 1: Базовий сценарій — великі числа, багато кандидатів
// ================================================================
void test_candidate_screening(size_t N_LIMBS, size_t N_CANDS, size_t K_DIGITS) {
    std::cout << "\n=== Тест: Просіювання кандидатів ===\n";
    std::cout << "  Лімбів: " << N_LIMBS
              << "  Кандидатів: " << N_CANDS
              << "  Перевіряємо K=" << K_DIGITS << " розрядів\n\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, BASE - 1);

    // Генеруємо A, B
    FastMul::BigNum A(N_LIMBS), B(N_LIMBS);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    A.back() = std::max<uint64_t>(1, A.back());
    B.back() = std::max<uint64_t>(1, B.back());

    // "Правильний" добуток — еталон
    auto C_ref = FastMul::multiply(A, B);

    // Перші K розрядів добутку — те, що ми перевіряємо
    std::vector<size_t>         positions(K_DIGITS);
    std::vector<FastMul::Limb>  targets(K_DIGITS);
    for (size_t i = 0; i < K_DIGITS; ++i) {
        positions[i] = i;
        targets[i]   = C_ref[i];
    }

    // Кандидати: A з різними значеннями першого (молодшого) лімба
    // A[0] змінюється → вплив на всі розряди добутку
    // Серед N_CANDS кандидатів рівно 1 збігається (delta=0)
    std::vector<FastMul::BigNum> candidates(N_CANDS, A);
    {
        // Перемішуємо порядок, щоб "правильний" не завжди був першим
        size_t correct_idx = rng() % N_CANDS;
        for (size_t i = 0; i < N_CANDS; ++i) {
            if (i != correct_idx)
                candidates[i][0] = dist(rng); // випадковий лімб
        }
        // candidates[correct_idx] = A (без змін) — правильний
    }

    // ----------------------------------------------------------
    //  Метод 1: Повне множення для кожного кандидата
    // ----------------------------------------------------------
    size_t full_found = 0;
    double full_ms = timeMs([&] {
        for (const auto& cand : candidates) {
            auto prod = FastMul::multiply(cand, B);
            bool match = (prod.size() >= K_DIGITS);
            for (size_t i = 0; i < K_DIGITS && match; ++i)
                if (prod[i] != targets[i]) match = false;
            if (match) ++full_found;
        }
    });

    // ----------------------------------------------------------
    //  Метод 2: candidateCheck (digitAt з early-exit)
    // ----------------------------------------------------------
    size_t sniper_found = 0;
    size_t total_ops    = 0;
    size_t early_exits  = 0;   // скільки кандидатів відсіялось до K-го розряду
    double sniper_ms = timeMs([&] {
        for (const auto& cand : candidates) {
            auto res = candidateCheckCounted(cand, B, positions, targets);
            total_ops += res.ops_done;
            if (res.matched) ++sniper_found;
            // Підраховуємо early-exit: якщо ops < теоретичного максимуму
            // (1+3+6+...+K*(K+1)/2) → кандидат відсіявся раніше
            size_t max_ops = 0;
            for (size_t k = 0; k < K_DIGITS; ++k)
                max_ops += std::min(k + 1, N_LIMBS);
            if (res.ops_done < max_ops) ++early_exits;
        }
    });

    // Теоретичні ops для повного множення (один кандидат)
    const size_t ops_full_one    = N_LIMBS * N_LIMBS;
    const size_t ops_sniper_avg  = (total_ops + N_CANDS - 1) / N_CANDS;

    assert(full_found == sniper_found);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  [Full Multiply]\n";
    std::cout << "    Час: " << full_ms << " мс\n";
    std::cout << "    Ops на кандидата: " << ops_full_one << "\n";
    std::cout << "    Знайдено: " << full_found << "\n\n";

    std::cout << "  [Sniper digitAt]\n";
    std::cout << "    Час: " << sniper_ms << " мс\n";
    std::cout << "    Ops на кандидата (середнє): " << ops_sniper_avg << "\n";
    std::cout << "    Early exits: " << early_exits << "/" << N_CANDS
              << " (" << 100.0*early_exits/N_CANDS << "%)\n";
    std::cout << "    Знайдено: " << sniper_found << "\n\n";

    std::cout << "  ⚡ Speedup: " << full_ms / sniper_ms << "x\n";
    std::cout << "  🎯 Ops reduction: "
              << (double)ops_full_one / ops_sniper_avg << "x менше\n";
}

// ================================================================
//  ТЕСТ 2: Incremental update — зміна одного лімба, оновлення K розрядів
//  Сценарій: ітеративна оптимізація (gradient descent по великих числах?)
//  Порівнюємо: перерахунок усього добутку vs оновлення K змінених розрядів
// ================================================================
void test_incremental_update(size_t N_LIMBS, size_t N_ITER, size_t K_CHECK) {
    std::cout << "\n=== Тест: Incremental Update ===\n";
    std::cout << "  Лімбів: " << N_LIMBS
              << "  Ітерацій: " << N_ITER
              << "  Перевіряємо K=" << K_CHECK << " розрядів після кожної зміни\n\n";

    std::mt19937_64 rng(99);
    std::uniform_int_distribution<uint64_t> dist(0, BASE - 1);

    FastMul::BigNum A(N_LIMBS), B(N_LIMBS);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Перші K позицій — те, що ми хочемо "моніторити"
    auto C_ref = FastMul::multiply(A, B);
    std::vector<size_t>        positions(K_CHECK);
    std::vector<FastMul::Limb> targets(K_CHECK);
    for (size_t i = 0; i < K_CHECK; ++i) {
        positions[i] = i;
        targets[i]   = C_ref[i];
    }

    // Симулюємо N_ITER ітерацій, де A[0] змінюється (hill-climb по одному лімбу)
    // Мета: знайти A[0] такий, щоб перші K розрядів добутку = targets.

    // Метод 1: повний перерахунок на кожній ітерації
    FastMul::BigNum A1 = A;
    size_t full_checks = 0;
    double full_ms = timeMs([&] {
        for (size_t iter = 0; iter < N_ITER; ++iter) {
            A1[0] = dist(rng); // змінюємо один лімб
            auto prod = FastMul::multiply(A1, B);
            bool ok = true;
            for (size_t i = 0; i < K_CHECK && ok; ++i)
                if (prod[i] != targets[i]) ok = false;
            if (ok) ++full_checks;
        }
    });

    // Метод 2: digitAt тільки для K позицій
    FastMul::BigNum A2 = A;
    size_t sniper_checks = 0;
    double sniper_ms = timeMs([&] {
        for (size_t iter = 0; iter < N_ITER; ++iter) {
            A2[0] = dist(rng);
            bool ok = FastMul::candidateCheck(A2, B, positions, targets, BASE);
            if (ok) ++sniper_checks;
        }
    });

    assert(full_checks == sniper_checks);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  [Full Recompute per iteration]\n";
    std::cout << "    Час: " << full_ms << " мс\n";
    std::cout << "    Знайдено: " << full_checks << "\n\n";

    std::cout << "  [Sniper digitAt per iteration]\n";
    std::cout << "    Час: " << sniper_ms << " мс\n";
    std::cout << "    Знайдено: " << sniper_checks << "\n\n";

    std::cout << "  ⚡ Speedup: " << full_ms / sniper_ms << "x\n";
}

// ================================================================
//  ТЕСТ 3: Вплив розміру K (кількості перевіряємих розрядів)
//  на ефективність. Де знаходиться "точка беззбитковості"?
// ================================================================
void test_breakeven_k(size_t N_LIMBS, size_t N_CANDS) {
    std::cout << "\n=== Тест: Break-even аналіз по K ===\n";
    std::cout << "  Лімбів: " << N_LIMBS
              << ", Кандидатів: " << N_CANDS << "\n\n";

    std::mt19937_64 rng(7);
    std::uniform_int_distribution<uint64_t> dist(0, BASE - 1);

    FastMul::BigNum A(N_LIMBS), B(N_LIMBS);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    auto C_ref = FastMul::multiply(A, B);

    // Час одного повного множення (базова вартість)
    double full_one_ms;
    {
        FastMul::BigNum dummy;
        full_one_ms = timeMs([&] { dummy = FastMul::multiply(A, B); });
    }
    double full_total_ms = full_one_ms * N_CANDS;

    std::cout << "  Повне множення (1 раз): " << full_one_ms << " мс\n";
    std::cout << "  Повне множення (" << N_CANDS << " разів): "
              << full_total_ms << " мс\n\n";

    std::cout << std::left
              << std::setw(6)  << "K"
              << std::setw(18) << "sniper_ms"
              << std::setw(14) << "speedup"
              << std::setw(14) << "вигідний?"
              << "\n"
              << std::string(52, '-') << "\n";

    // Генеруємо кандидати де A[0] випадковий → більшість відсіється на K=1
    // але деякі пройдуть далі (для чесного виміру великих K)
    // Щоб примусити кандидатів "виживати" довше, робимо їх схожими на A
    std::vector<FastMul::BigNum> cands(N_CANDS, A);
    for (size_t i = 0; i < N_CANDS; ++i) {
        // Лімб 0 = правильний (тому k=0 завжди проходить)
        // Лімб 1 = випадковий (тому k=1 відсіює більшість)
        cands[i][1] = dist(rng);
    }

    // Перераховуємо targets щоб lімб 0 завжди збігався
    std::vector<size_t>        pos_all(N_LIMBS * 2);
    std::vector<FastMul::Limb> vals_all(N_LIMBS * 2);
    for (size_t i = 0; i < std::min(C_ref.size(), N_LIMBS * 2); ++i) {
        pos_all[i]  = i;
        vals_all[i] = C_ref[i];
    }

    const int REPEATS = 5; // повторюємо для стабільності

    for (size_t K : {1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL, 256UL,
                     500UL, 750UL, 1000UL}) {
        if (K > C_ref.size()) break;

        std::vector<size_t>        pos (pos_all.begin(),  pos_all.begin()  + K);
        std::vector<FastMul::Limb> vals(vals_all.begin(), vals_all.begin() + K);

        double sniper_ms = 0;
        for (int r = 0; r < REPEATS; ++r) {
            sniper_ms += timeMs([&] {
                volatile size_t dummy = 0;
                for (const auto& cand : cands)
                    dummy += FastMul::candidateCheck(cand, B, pos, vals, BASE) ? 1 : 0;
            });
        }
        sniper_ms /= REPEATS;

        double speedup = full_total_ms / sniper_ms;
        std::cout << std::setw(6)  << K
                  << std::setw(18) << sniper_ms
                  << std::setw(14) << speedup
                  << std::setw(14) << (speedup > 1.0 ? "✅ ТАК" : "❌ НІ")
                  << "\n";
    }
}

int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║   MarkerMul «Sniper Mode» — Практична оцінка ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n";

    // Тест 1: типовий сценарій — велика N, багато кандидатів, K малий
    test_candidate_screening(/*N_LIMBS=*/500, /*N_CANDS=*/10000, /*K=*/4);

    // Тест 2: incremental update (оптимізаційний loop)
    test_incremental_update(/*N_LIMBS=*/500, /*N_ITER=*/5000, /*K=*/4);

    // Тест 3: де знаходиться точка беззбитковості по K?
    test_breakeven_k(/*N_LIMBS=*/500, /*N_CANDS=*/5000);

    return 0;
}
