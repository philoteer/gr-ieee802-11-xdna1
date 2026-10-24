#ifndef FRAME_EQUALIZER_ENGINE_CLASSIC_H
#define FRAME_EQUALIZER_ENGINE_CLASSIC_H

#include "frame_equalizer_engine_common.h"
#include "channel_estimator/select.h"

// Classic reference PHY pipeline for one prepared symbol. This is the behavior
// that FRAME_EQUALIZER_ENGINE=CLASSIC keeps, and the baseline the AI engine
// replaces: estimator update, equalization, hard demapping, and channel
// tracking. SIGNAL decoding and payload/tag emission stay in the scheduler.
//
// Inputs:
//   symbol     one prepared 64-bin frequency-domain symbol
//   number     frame-relative symbol index (0/1 LTS, 2 SIGNAL, 3+ payload)
//   modulation 2 for SIGNAL, otherwise the rate selected by parse_signal()
// Output:
//   decisions  one decision code per data carrier, filled for number >= 2
static inline void equalize_symbol(
        const cint32 *symbol, int number, int modulation,
        int8_t decisions[64]) {
    update_channel_estimate(symbol, number, d_channel, d_snr_linear);
    if (number >= 2) {
        // SIGNAL is always BPSK. Later symbols use the modulation selected
        // by SIGNAL; pilots, guards, and DC never enter decisions.
        {
            // Equalization plus QAM channel-power generation measured about
            // 1.51k cycles/tile (6.72% of equalize) in the full CSI profile.
            // Equalize every bin in contiguous 8-lane batches.
            aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
            aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
            for (int bin = 0; bin < 64; bin += 8) {
                const auto sample_v = aie::load_v<8>(symbol + bin);
                const auto channel_v = aie::load_v<8>(d_channel + bin);
                aie::store_v(
                    d_equalized + bin,
                    aie::mul<cacc64>(sample_v, aie::op_conj(channel_v))
                        .to_vector<cint32>(12));
            }
            // BPSK and QPSK need only signs; avoid power calculation for
            // those modes and keep the QAM path branch-free per batch.
            if (modulation > 4 || FRAME_EQUALIZER_OUTPUT_SYMBOLS) {
                for (int bin = 0; bin < 64; bin += 8) {
                    const auto channel_v = aie::load_v<8>(d_channel + bin);
                    aie::store_v(
                        d_channel_power + bin,
                        aie::abs_square(channel_v, 12));
                }
            }
        }
        hard_decision(d_equalized, d_channel_power, decisions, modulation);
        // Decision-directed estimators update the channel after demapping,
        // so this symbol was equalized with the previous estimate. LS and
        // comb use a no-op here.
        track_channel_estimate(symbol, decisions, modulation, number, d_channel);
    }
}

#endif
