#ifndef FRAME_EQUALIZER_EQUALIZER_CONSTELLATION_H
#define FRAME_EQUALIZER_EQUALIZER_CONSTELLATION_H

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

// Support code for the decision-directed estimators (LMS and STA). Both turn a
// received sample R and the ideal constellation point S of a hard decision into
// a channel observation R / S.

// 1/S for every constellation point in signed Q24, indexed by the decision code
// produced by hard_decision(). The code ordering follows the constellation
// point ordering used by the reference (BPSK, QPSK, 16-QAM, 64-QAM), so one
// complex multiply R * (1/S) yields the channel observation exactly.
alignas(aie::vector_decl_align) const cint32 kInversePointsBpsk[2] = {
    { -16777216, 0 }, { 16777216, 0 },
};
alignas(aie::vector_decl_align) const cint32 kInversePointsQpsk[4] = {
    { -11863283, 11863283 }, { 11863283, 11863283 },
    { -11863283, -11863283 }, { 11863283, -11863283 },
};
alignas(aie::vector_decl_align) const cint32 kInversePoints16Qam[16] = {
    { -8842369, 8842369 }, { 8842369, 8842369 },
    { -5305422, 15916265 }, { 5305422, 15916265 },
    { -8842369, -8842369 }, { 8842369, -8842369 },
    { -5305422, -15916265 }, { 5305422, -15916265 },
    { -15916265, 5305422 }, { 15916265, 5305422 },
    { -26527108, 26527108 }, { 26527108, 26527108 },
    { -15916265, -5305422 }, { 15916265, -5305422 },
    { -26527108, -26527108 }, { 26527108, -26527108 },
};
alignas(aie::vector_decl_align) const cint32 kInversePoints64Qam[64] = {
    { -7766342, 7766342 }, { 7766342, 7766342 },
    { -2174576, 15222030 }, { 2174576, 15222030 },
    { -7346540, 10285155 }, { 7346540, 10285155 },
    { -5623903, 13122440 }, { 5623903, 13122440 },
    { -7766342, -7766342 }, { 7766342, -7766342 },
    { -2174576, -15222030 }, { 2174576, -15222030 },
    { -7346540, -10285155 }, { 7346540, -10285155 },
    { -5623903, -13122440 }, { 5623903, -13122440 },
    { -15222030, 2174576 }, { 15222030, 2174576 },
    { -54364393, 54364393 }, { 54364393, 54364393 },
    { -20909382, 4181876 }, { 20909382, 4181876 },
    { -32618636, 10872879 }, { 32618636, 10872879 },
    { -15222030, -2174576 }, { 15222030, -2174576 },
    { -54364393, -54364393 }, { 54364393, -54364393 },
    { -20909382, -4181876 }, { 20909382, -4181876 },
    { -32618636, -10872879 }, { 32618636, -10872879 },
    { -10285155, 7346540 }, { 10285155, 7346540 },
    { -4181876, 20909382 }, { 4181876, 20909382 },
    { -10872879, 10872879 }, { 10872879, 10872879 },
    { -9593716, 15989527 }, { 9593716, 15989527 },
    { -10285155, -7346540 }, { 10285155, -7346540 },
    { -4181876, -20909382 }, { 4181876, -20909382 },
    { -10872879, -10872879 }, { 10872879, -10872879 },
    { -9593716, -15989527 }, { 9593716, -15989527 },
    { -13122440, 5623903 }, { 13122440, 5623903 },
    { -10872879, 32618636 }, { 10872879, 32618636 },
    { -15989527, 9593716 }, { 15989527, 9593716 },
    { -18121464, 18121464 }, { 18121464, 18121464 },
    { -13122440, -5623903 }, { 13122440, -5623903 },
    { -10872879, -32618636 }, { 10872879, -32618636 },
    { -15989527, -9593716 }, { 15989527, -9593716 },
    { -18121464, -18121464 }, { 18121464, -18121464 },
};

/** @brief Selects the Q24 inverse-constellation table for a modulation order.
 *
 *  @param modulation 2 (BPSK), 4 (QPSK), or 16 (16-QAM); any other value maps
 *         to 64-QAM, matching the demapper's else branch.
 *  @return Table indexed by the decision codes produced by hard_decision().
 */
static inline const cint32 *inverse_points_for_modulation(int modulation) {
    if (modulation == 2) return kInversePointsBpsk;
    if (modulation == 4) return kInversePointsQpsk;
    if (modulation == 16) return kInversePoints16Qam;
    return kInversePoints64Qam;
}

/** @brief Rounds a signed Q24 product to the nearest integer.
 *
 *  @param value Q24 product.
 *  @return Rounded int32 value.
 */
static inline int32_t round_q24(int64_t value) {
    const int64_t half = (int64_t)1 << 23;
    return (int32_t)(value >= 0 ? (value + half) >> 24
                                : -(((-value) + half) >> 24));
}

/** @brief history = 0.5 * history + 0.5 * observation.
 *
 *  @param history Previous long-term estimate.
 *  @param observation New per-symbol estimate.
 *  @return Blended estimate.
 */
static inline cint32 blend_half(cint32 history, cint32 observation) {
    return {
        (int32_t)(history.real +
                  (((int64_t)observation.real - history.real) >> 1)),
        (int32_t)(history.imag +
                  (((int64_t)observation.imag - history.imag) >> 1)),
    };
}

/** @brief Channel observation R / S from the Q24 inverse of the ideal point.
 *
 *  @param sample Received frequency-domain sample.
 *  @param inverse_point Q24 1/S value of the hard decision's ideal point.
 *  @return Complex channel observation.
 */
static inline cint32 channel_observation(cint32 sample, cint32 inverse_point) {
    const int64_t real = (int64_t)sample.real * inverse_point.real -
                         (int64_t)sample.imag * inverse_point.imag;
    const int64_t imag = (int64_t)sample.real * inverse_point.imag +
                         (int64_t)sample.imag * inverse_point.real;
    return { round_q24(real), round_q24(imag) };
}

#endif
