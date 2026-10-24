// Generated twiddle factors for a forward 64-point radix-4 FFT.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

alignas(aie::vector_decl_align) static const cint16 kStage0Tw2[] = {
    {32767, 0}};
alignas(aie::vector_decl_align) static const cint16 kStage0Tw1[] = {
    {32767, 0}};
alignas(aie::vector_decl_align) static const cint16 kStage0Tw3[] = {
    {32767, 0}};

alignas(aie::vector_decl_align) static const cint16 kStage1Tw2[] = {
    {32767, 0}, {23170, -23170}, {0, -32768}, {-23170, -23170}};
alignas(aie::vector_decl_align) static const cint16 kStage1Tw1[] = {
    {32767, 0}, {30273, -12539}, {23170, -23170}, {12539, -30273}};
alignas(aie::vector_decl_align) static const cint16 kStage1Tw3[] = {
    {32767, 0}, {12539, -30273}, {-23170, -23170}, {-30273, 12539}};

alignas(aie::vector_decl_align) static const cint16 kStage2Tw2[] = {
    {32767, 0}, {32138, -6392}, {30273, -12539}, {27245, -18204},
    {23170, -23170}, {18204, -27245}, {12539, -30273}, {6392, -32138},
    {0, -32768}, {-6392, -32138}, {-12539, -30273}, {-18204, -27245},
    {-23170, -23170}, {-27245, -18204}, {-30273, -12539}, {-32138, -6392}};
alignas(aie::vector_decl_align) static const cint16 kStage2Tw1[] = {
    {32767, 0}, {32610, -3211}, {32138, -6392}, {31357, -9512},
    {30273, -12539}, {28898, -15446}, {27245, -18204}, {25330, -20787},
    {23170, -23170}, {20787, -25330}, {18204, -27245}, {15446, -28898},
    {12539, -30273}, {9512, -31357}, {6392, -32138}, {3211, -32610}};
alignas(aie::vector_decl_align) static const cint16 kStage2Tw3[] = {
    {32767, 0}, {31357, -9512}, {27245, -18204}, {20787, -25330},
    {12539, -30273}, {3211, -32610}, {-6392, -32138}, {-15446, -28898},
    {-23170, -23170}, {-28898, -15446}, {-32138, -6392}, {-32610, 3211},
    {-30273, 12539}, {-25330, 20787}, {-18204, 27245}, {-9512, 31357}};
