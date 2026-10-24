#ifndef FRAME_EQUALIZER_ENGINE_AI_H
#define FRAME_EQUALIZER_ENGINE_AI_H

#include "frame_equalizer_algorithm.h"
#include "frame_equalizer_engine_common.h"
#include "model/frame_equalizer_model.h"

#if FRAME_EQUALIZER_ALGORITHM != FRAME_EQUALIZER_ALGORITHM_LS
#error "The initial AI equalizer engine requires EQUALIZER_ALGORITHM=LS"
#endif

namespace {

/**
 * Maps an IEEE 802.11 constellation size to its exported network index.
 *
 * @param modulation Number of constellation points: 2, 4, 16, or 64.
 * @return Model index 0 for BPSK, 1 for QPSK, 2 for 16-QAM, or 3 for 64-QAM.
 */
static inline unsigned ai_model_for_modulation(int modulation) {
    if (modulation == 2) return 0;
    if (modulation == 4) return 1;
    if (modulation == 16) return 2;
    return 3;
}

/**
 * @brief Runs the learned demapper for one prepared symbol.
 *
 * Symbols 0 and 1 are the LTS pair and produce no output. The channel estimate
 * comes from the scheduler's LTS-based LS update; this engine only forms the
 * matched-filter numerator, the channel power, and the model decisions, so no
 * decision-directed tracking is applied yet.
 *
 * @param symbol One prepared 64-bin frequency-domain symbol.
 * @param number Frame-relative symbol index (0/1 LTS, 2 SIGNAL, 3+ payload).
 * @param modulation Constellation size selected by SIGNAL for this symbol.
 * @param decisions Output storage for packed data-subcarrier decisions.
 */
static inline void equalize_symbol(
        const cint32 *symbol, const cint32 *channel, int number, int modulation,
        int8_t decisions[64]) {
    if (number < 2) return;

    {
        aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
        aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
        for (int bin = 0; bin < FFT_SIZE; bin += 8) {
            const auto sample_v = aie::load_v<8>(symbol + bin);
            const auto channel_v = aie::load_v<8>(channel + bin);
            aie::store_v(
                d_equalized + bin,
                aie::mul<cacc64>(sample_v, aie::op_conj(channel_v))
                    .to_vector<cint32>(12));
        }
        for (int bin = 0; bin < FFT_SIZE; bin += 8) {
            const auto channel_v = aie::load_v<8>(channel + bin);
            aie::store_v(
                d_channel_power + bin,
                aie::abs_square(channel_v, 12));
        }
    }

    AiMlp::forward(
        d_equalized, d_channel_power, decisions,
        ai_model_for_modulation(modulation));
    // Version one is stateless after the LTS-derived LS estimate. Learned
    // decision-directed tracking will be added as a separate stage.
    
    //track_channel_estimate(symbol, decisions, modulation, number, d_channel); will be also implemented in the future, just not now
}

} // namespace

#endif
