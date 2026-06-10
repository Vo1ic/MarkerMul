#pragma once
// ================================================================
//  BinaryMarkerMul.h  —  псевдонім для зворотної сумісності
//
//  Клас BinaryMarkerMul більше не дублює логіку — він є простим
//  псевдонімом MarkerMul<uint8_t, 2>.
//
//  Весь функціонал доступний через:
//    BinaryMarkerMul::multiply(A, B)
//    BinaryMarkerMul::bitAt(A, B, k, depth)   ← нова назва: digitAt
//    BinaryMarkerMul::candidateCheck(...)
//    BinaryMarkerMul::fromInt(x)
//    BinaryMarkerMul::print(A)
//    BinaryMarkerMul::calculateDepth(L)
// ================================================================

#include "MarkerMul.h"

/// Повний псевдонім: MarkerMul<uint8_t, 2>
using BinaryMarkerMul = BinMul;
