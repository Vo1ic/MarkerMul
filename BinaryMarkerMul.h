#pragma once
#include <vector>
#include <iostream>
#include <cmath>

class BinaryMarkerMul {
public:
    using Bits = std::vector<int>; // LSB -> MSB

    // 1. �������� �������� � ������� ������
    static Bits multiply(const Bits& A, const Bits& B) {
        int n = A.size(), m = B.size();
        Bits result(n + m, 0);

        for (int k = 0; k < n + m - 1; k++) {
            int sum = result[k];
            for (int i = 0; i <= k; i++) {
                int j = k - i;
                if (i < n && j < m) {
                    sum += A[i] & B[j]; // ����� �������� = AND
                }
            }
            result[k] = sum % 2;
            result[k + 1] += sum / 2;
        }
        return trim(result);
    }

    // 2. ������� ���������� ����������� ���
    static int bitAt(const Bits& A, const Bits& B, int k) {
        int n = A.size(), m = B.size();
        int L = std::min(n, m);
        int depth = (int)(std::log2((2 - 1) * (2 - 1) * L)) + 1; // ������� ��� b=2

        int carry = 0, bit = 0;
        for (int pos = k - depth + 1; pos <= k; pos++) {
            if (pos < 0) continue;
            int sum = carry;
            for (int i = 0; i <= pos; i++) {
                int j = pos - i;
                if (i < n && j < m) sum += A[i] & B[j];
            }
            bit = sum % 2;
            carry = sum / 2;
        }
        return bit;
    }

    // 3. Գ���� ���������
    static bool candidateCheck(const Bits& A, const Bits& B,
        const std::vector<int>& knownPositions,
        const std::vector<int>& knownBits)
    {
        for (size_t idx = 0; idx < knownPositions.size(); idx++) {
            int pos = knownPositions[idx];
            int expected = knownBits[idx];
            if (bitAt(A, B, pos) != expected) return false;
        }
        return true;
    }

    // ��������: ������� ����� �����
    static void print(const Bits& A) {
        for (int i = A.size() - 1; i >= 0; i--) std::cout << A[i];
        std::cout << "\n";
    }

    // ������������ � int � ������ ������
    static Bits fromInt(unsigned long long x) {
        Bits A;
        if (x == 0) return { 0 };
        while (x > 0) {
            A.push_back(x & 1ULL);
            x >>= 1ULL;
        }
        return A;
    }

private:
    static Bits trim(Bits A) {
        while (A.size() > 1 && A.back() == 0) A.pop_back();
        return A;
    }
};
