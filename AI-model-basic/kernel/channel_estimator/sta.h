// Warning: a largely unoptimized scalar implementation; only LS equalizer has been optimized at the moment.
// (not recommended for actual use, mostly for testing purposes for now)

#ifndef FRAME_EQUALIZER_EQUALIZER_STA_H
#define FRAME_EQUALIZER_EQUALIZER_STA_H

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#include "common.h"
#include "constellation.h"

// Spatial-temporal averaging estimator, ported from original/equalizer/sta.cc.
// Every symbol after the LTS pair forms an instantaneous channel observation,
// smooths it over beta = 2 neighboring occupied bins, then blends the result
// into the long-term estimate with alpha = 0.5. Both the spatial and the
// temporal stage run after the hard decisions, so the symbol was equalized with
// the previous estimate, exactly as in the reference.

// Q24 reciprocals of the smoothing window sizes 1..5. Index by the number of
// occupied bins in the window; DC and the band edges reduce that count.
alignas(aie::vector_decl_align) const int32_t kStaReciprocalQ24[6] = {
    0, 16777216, 8388608, 5592405, 4194304, 3355443,
};

static inline void update_channel_estimate(const cint32 *symbol, int number,
                                             cint32 channel[FFT_SIZE],
                                             uint64_t &snr) {
    init_channel_estimate_from_lts(symbol, number, channel, snr);
}

static NOINLINE void track_channel_estimate(const cint32 *symbol,
                                            const int8_t *decisions,
                                            int modulation, int number,
                                            cint32 channel[FFT_SIZE]) {
    alignas(aie::vector_decl_align) cint32 observation[FFT_SIZE];
    const cint32 *inverse_points = inverse_points_for_modulation(modulation);

    // Known pilot values give direct observations at the pilot bins; the
    // transmitted signs are p, p, p, -p for bins 11, 25, 39, 53.
    const int polarity = kPolarity[(number - 2) % 127];
    observation[11] = polarity * symbol[11];
    observation[25] = polarity * symbol[25];
    observation[39] = polarity * symbol[39];
    observation[53] = -polarity * symbol[53];

    // Data bins use the ideal constellation point of the hard decision.
    int data = 0;
    AIE_LOOP_NO_UNROLL
    for (int bin = 0; bin < FFT_SIZE; ++bin) {
        if (bin == 11 || bin == 25 || bin == 32 || bin == 39 || bin == 53 ||
            bin < 6 || bin > 58) {
            continue;
        }
        observation[bin] = channel_observation(
            symbol[bin], inverse_points[decisions[data++]]);
    }

    // Spatial smoothing: average the occupied neighbors within beta bins, then
    // temporally blend the result with alpha = 0.5. DC and guards are skipped.
    AIE_LOOP_NO_UNROLL
    for (int bin = 6; bin <= 58; ++bin) {
        const int first = bin - 2 < 6 ? 6 : bin - 2;
        const int last = bin + 2 > 58 ? 58 : bin + 2;
        int count = 0;
        int64_t real = 0, imag = 0;
        for (int k = first; k <= last; ++k) {
            if (k == 32) continue;
            ++count;
            real += observation[k].real;
            imag += observation[k].imag;
        }
        const int32_t reciprocal = kStaReciprocalQ24[count];
        channel[bin] = blend_half(channel[bin],
                                  { round_q24(real * reciprocal),
                                    round_q24(imag * reciprocal) });
    }
}

static inline void reset_channel_estimate(uint64_t &snr) { (void)snr; }

#endif
