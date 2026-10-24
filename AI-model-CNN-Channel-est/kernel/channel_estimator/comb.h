#ifndef FRAME_EQUALIZER_EQUALIZER_COMB_H
#define FRAME_EQUALIZER_EQUALIZER_COMB_H

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#include "common.h"

// Comb channel estimator, ported from stage34/original/equalizer/comb.cc.
//
// Every symbol is tracked from its four pilot subcarriers, interpolated across
// five spans and smoothed over time. The public interface matches the other
// algorithms:
//   update_channel_estimate(symbol, number, channel, snr)
//   reset_channel_estimate(snr)
// kPolarity comes from common.h.

// Q1.15 position within each of comb's five interpolation spans.
alignas(aie::vector_decl_align) const uint16_t kCombInterpolationWeightQ15[64] = {
    0, 2979, 5958, 8937, 11916, 14895, 17873, 20852,
    23831, 26810, 29789, 32768, 2341, 4681, 7022, 9362,
    11703, 14043, 16384, 18725, 21065, 23406, 25746, 28087,
    30427, 32768, 2341, 4681, 7022, 9362, 11703, 14043,
    16384, 18725, 21065, 23406, 25746, 28087, 30427, 32768,
    2341, 4681, 7022, 9362, 11703, 14043, 16384, 18725,
    21065, 23406, 25746, 28087, 30427, 32768, 2979, 5958,
    8937, 11916, 14895, 17873, 20852, 23831, 26810, 29789,
};

using comb_vector = aie::vector<cint32, 8>;
using comb_weight_vector = aie::vector<int32_t, 8>;

/** @brief Linearly interpolates one 8-bin chunk between two span anchors.
 *
 *  @param left Left anchor per lane.
 *  @param right Right anchor per lane.
 *  @param weight Q15 position within the span.
 *  @return Interpolated channel estimate vector.
 */
static inline comb_vector interpolate_vector(
        const comb_vector &left, const comb_vector &right,
        const comb_weight_vector &weight) {
    return aie::add(
        left,
        aie::mul<cacc64>(aie::sub(right, left), weight)
            .to_vector<cint32>(15));
}

/** @brief Interpolates one 8-bin chunk and smooths it over time.
 *
 *  Advances the stored estimate one fifth of the way toward the interpolated
 *  value, or replaces it outright on the frame's first symbol.
 *
 *  @param left Left anchor per lane.
 *  @param right Right anchor per lane.
 *  @param weight Q15 position within the span.
 *  @param channel In/out channel estimate row for this chunk.
 *  @param initialize All-ones mask on the first symbol, all-zeros later.
 */
static inline void update_channel_vector(
        const comb_vector &left, const comb_vector &right,
        const comb_weight_vector &weight, cint32 *channel,
        const aie::mask<8> &initialize) {
    const auto estimate = interpolate_vector(left, right, weight);
    const auto history = aie::load_v<8>(channel);
    // ceil(2^31 / 5) implements the scalar nearest-integer division by five.
    const auto correction = aie::mul<cacc64>(
        aie::sub(estimate, history), int32_t(429496730))
        .to_vector<cint32>(31);
    const auto updated = aie::add(history, correction);
    // select() returns the second value for set mask bits and the first for
    // clear bits, so initialize (all ones on number == 0, all zeros otherwise)
    // picks estimate for the frame's first symbol, matching the scalar
    // channel[bin] = estimate, and the smoothed value for every later symbol.
    aie::store_v(channel, aie::select(updated, estimate, initialize));
}

/** @brief Pilot-derived comb channel update for one symbol.
 *
 *  Removes the known pilot signs, averages the four pilot observations as the
 *  extrapolation anchor, and fills all 64 bins by piecewise-linear
 *  interpolation with 1/5-weight temporal smoothing.
 *
 *  @param symbol Prepared frequency-domain symbol, FFT-shifted order.
 *  @param number Frame-relative symbol index selecting the pilot polarity.
 *  @param channel In/out 64-bin channel estimate.
 *  @param snr Unused; reset_channel_estimate() sets the legacy placeholder.
 */
static NOINLINE void update_channel_estimate(const cint32 *symbol, int number,
                                             cint32 channel[FFT_SIZE],
                                             uint64_t &snr) {
    (void)snr;
    cint32 pilot[4];
    if (number < 2) {
        pilot[0] = symbol[11];
        pilot[1] = -symbol[25];
        pilot[2] = symbol[39];
        pilot[3] = symbol[53];
    } else {
        const int polarity = kPolarity[(number - 2) % 127];
        pilot[0] = polarity * symbol[11];
        pilot[1] = polarity * symbol[25];
        pilot[2] = polarity * symbol[39];
        pilot[3] = -polarity * symbol[53];
    }
    // The FFT input is cint16 Q0.15, so every bin is bounded by
    // 64 * 2^15 = 2^21 and the four-pilot sum by 2^23. int32 arithmetic is
    // exact here; the 64-bit path is only needed in interpolate_vector(), where
    // a Q15 weight scales the per-bin difference. This average is just the
    // extrapolation anchor, so a 2-bit rounding shift is accurate enough.
    const cint32 average =
        (pilot[0] + pilot[1] + pilot[2] + pilot[3] + 2) >> 2;

    // symmetric_inf reproduces round_shift() and divide_by_five_nearest().
    aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
    // First symbol of a frame: overwrite the channel estimate instead of
    // smoothing it. aie::mask broadcasts the bool, so this is all ones for
    // number == 0 and all zeros for every later symbol.
    const auto initialize = aie::mask<8>(number == 0);
    const auto average_v = aie::broadcast<cint32, 8>(average);
    const auto pilot0_v = aie::broadcast<cint32, 8>(pilot[0]);
    const auto pilot1_v = aie::broadcast<cint32, 8>(pilot[1]);
    const auto pilot2_v = aie::broadcast<cint32, 8>(pilot[2]);
    const auto pilot3_v = aie::broadcast<cint32, 8>(pilot[3]);

    // The unsigned Q15 weights are zero-extended once per 16-bin chunk. The
    // complex-by-uint16 AIE API path is unavailable on this toolchain.
    const auto weights0 = aie::vector_cast<int32_t>(
        aie::unpack(aie::load_v<16>(kCombInterpolationWeightQ15)));
    const auto weights16 = aie::vector_cast<int32_t>(
        aie::unpack(aie::load_v<16>(kCombInterpolationWeightQ15 + 16)));
    const auto weights32 = aie::vector_cast<int32_t>(
        aie::unpack(aie::load_v<16>(kCombInterpolationWeightQ15 + 32)));
    const auto weights48 = aie::vector_cast<int32_t>(
        aie::unpack(aie::load_v<16>(kCombInterpolationWeightQ15 + 48)));

    update_channel_vector(average_v, pilot0_v, weights0.extract<8>(0),
                          channel, initialize);
    const comb_vector left8(average, average, average, average,
                            pilot[0], pilot[0], pilot[0], pilot[0]);
    const comb_vector right8(pilot[0], pilot[0], pilot[0], pilot[0],
                             pilot[1], pilot[1], pilot[1], pilot[1]);
    update_channel_vector(left8, right8, weights0.extract<8>(1),
                          channel + 8, initialize);

    update_channel_vector(pilot0_v, pilot1_v, weights16.extract<8>(0),
                          channel + 16, initialize);
    const comb_vector left24(pilot[0], pilot[0], pilot[1], pilot[1],
                             pilot[1], pilot[1], pilot[1], pilot[1]);
    const comb_vector right24(pilot[1], pilot[1], pilot[2], pilot[2],
                              pilot[2], pilot[2], pilot[2], pilot[2]);
    update_channel_vector(left24, right24, weights16.extract<8>(1),
                          channel + 24, initialize);

    update_channel_vector(pilot1_v, pilot2_v, weights32.extract<8>(0),
                          channel + 32, initialize);
    update_channel_vector(pilot2_v, pilot3_v, weights32.extract<8>(1),
                          channel + 40, initialize);

    const comb_vector left48(pilot[2], pilot[2], pilot[2], pilot[2],
                             pilot[2], pilot[2], pilot[3], pilot[3]);
    const comb_vector right48(pilot[3], pilot[3], pilot[3], pilot[3],
                              pilot[3], pilot[3], average, average);
    update_channel_vector(left48, right48, weights48.extract<8>(0),
                          channel + 48, initialize);
    update_channel_vector(pilot3_v, average_v, weights48.extract<8>(1),
                          channel + 56, initialize);
}

// Comb updates the estimate from the pilots before equalization, so there is
// nothing to do after the hard decisions.
static inline void track_channel_estimate(const cint32 *, const int8_t *, int,
                                          int, cint32 [FFT_SIZE]) {}

static inline void reset_channel_estimate(uint64_t &snr) {
    // comb::get_snr() returns its legacy 42 dB placeholder. The metadata stores
    // twice the linear ratio, scaled by 16, matching the host's LS conversion.
    snr = 507166;
}

#endif
