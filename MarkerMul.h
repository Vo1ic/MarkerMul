#pragma once
#include <vector>
#include <iostream>
#include <cmath>

class MarkerMul {
public:
    using BigInt = std::vector<int>; // �������� ����� �� ����� ���� (������� ������)

    // 1. �������� �������� (������ ���������)
    static BigInt multiply(const BigInt& A, const BigInt& B, int base = 10) {
        int n = A.size(), m = B.size();
        BigInt result(n + m, 0);

        for (int k = 0; k < n + m - 1; k++) {
            int sum = result[k]; // ������� � ������������ �����
            for (int i = 0; i <= k; i++) {
                int j = k - i;
                if (i < n && j < m) {
                    sum += A[i] * B[j];
                }
            }
            result[k] = sum % base;
            result[k + 1] += sum / base;
        }
        return trim(result);
    }

    // 2. ������� ���������� ����������� �������
    static int digitAt(const BigInt& A, const BigInt& B, int k, int base = 10) {
        int n = A.size(), m = B.size();
        int L = std::min(n, m);
        int depth = (int)(std::log((base - 1) * (base - 1) * L) / std::log(base)) + 1;

        int carry = 0, digit = 0;
        for (int pos = k - depth + 1; pos <= k; pos++) {
            if (pos < 0) continue;
            int sum = carry;
            for (int i = 0; i <= pos; i++) {
                int j = pos - i;
                if (i < n && j < m) sum += A[i] * B[j];
            }
            digit = sum % base;
            carry = sum / base;
        }
        return digit;
    }

    // 3. Գ���� ��������� (�������� �� ����� ������ ������� ��������)
    static bool candidateCheck(const BigInt& A, const BigInt& B,
        const std::vector<int>& knownPositions,
        const std::vector<int>& knownDigits,
        int base = 10)
    {
        for (size_t idx = 0; idx < knownPositions.size(); idx++) {
            int pos = knownPositions[idx];
            int expected = knownDigits[idx];
            if (digitAt(A, B, pos, base) != expected) {
                return false; // ������������� � �������� ���������
            }
        }
        return true; // �� �������
    }

    // ��������: ������� BigInt
    static void print(const BigInt& A) {
        for (int i = A.size() - 1; i >= 0; i--) std::cout << A[i];
        std::cout << "\n";
    }

    // ������������ � int
    static BigInt fromInt(long long x, int base = 10) {
        BigInt A;
        if (x == 0) return { 0 };
        while (x > 0) {
            A.push_back(x % base);
            x /= base;
        }
        return A;
    }

private:
    static BigInt trim(BigInt A) {
        while (A.size() > 1 && A.back() == 0) A.pop_back();
        return A;
    }
};
