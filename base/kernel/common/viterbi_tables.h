#ifndef VITERBI_TABLES_H
#define VITERBI_TABLES_H

#include <stdint.h>
#include <aie_api/aie.hpp>

// K=7, polynomials 0155/0117. Entry s omits the hypothesis bit; XOR that
// bit into both parities. Shared by SIGNAL and streaming DATA decoders.
alignas(aie::vector_decl_align) static constexpr uint8_t kViterbiParityA[64] = {
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
};
alignas(aie::vector_decl_align) static constexpr uint8_t kViterbiParityB[64] = {
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
    1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
};
alignas(aie::vector_decl_align) static constexpr uint8_t kViterbiStates[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
};

#endif
