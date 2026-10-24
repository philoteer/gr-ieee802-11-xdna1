// Warning: a largely unoptimized scalar implementation; only LS equalizer has been optimized at the moment.
// (not recommended for actual use, mostly for testing purposes for now)

#ifndef FRAME_EQUALIZER_EQUALIZER_LMS_H
#define FRAME_EQUALIZER_EQUALIZER_LMS_H

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#include "common.h"
#include "constellation.h"

static inline void update_channel_estimate(const cint32 *symbol, int number,
                                             cint32 channel[FFT_SIZE],
                                             uint64_t &snr) {
    init_channel_estimate_from_lts(symbol, number, channel, snr);
}

static NOINLINE void track_channel_estimate(const cint32 *symbol,
                                            const int8_t *decisions,
                                            int modulation, int,
                                            cint32 channel[FFT_SIZE]) {
    const cint32 *inverse_points = inverse_points_for_modulation(modulation);

    // Data carriers only, in the same order as the reference loop: pilots, DC,
    // and guard bins keep their long-training estimate.
    int data = 0;
    AIE_LOOP_NO_UNROLL
    for (int bin = 0; bin < FFT_SIZE; ++bin) {
        if (bin == 11 || bin == 25 || bin == 32 || bin == 39 || bin == 53 ||
            bin < 6 || bin > 58) {
            continue;
        }
        channel[bin] = blend_half(
            channel[bin],
            channel_observation(symbol[bin],
                                inverse_points[decisions[data++]]));
    }
}

static inline void reset_channel_estimate(uint64_t &snr) { (void)snr; }

#endif
