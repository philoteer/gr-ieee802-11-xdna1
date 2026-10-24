#ifndef FRAME_EQUALIZER_MODEL_MLP_16X16_H
#define FRAME_EQUALIZER_MODEL_MLP_16X16_H

// Two-hidden-layer bias-free demapper: 3 -> 16 -> 16 -> bits, with ReLU after
// both hidden layers. Weights come from utils/frame_equalizer_ai's "mlp_16x16"
// architecture; selection is described in frame_equalizer_model.h.

#include "frame_equalizer_model_mlp_16x16_weights.h"
#include "frame_equalizer_model_common.h"

namespace {

static_assert(frame_equalizer_ai::kHiddenLayers == 2,
              "weights header does not match the selected MLP architecture");

// Hidden activations live in data memory so the two dense layers do not put
// thirty-two simultaneously live vectors on the core stack.
alignas(aie::vector_decl_align) static int32_t
    d_ai_hidden1[frame_equalizer_ai::kHiddenSize * 16];
alignas(aie::vector_decl_align) static int32_t
    d_ai_hidden2[frame_equalizer_ai::kHiddenSize * 16];

struct AiMlp {
    /**
     * Runs the complete learned demapper over one 64-bin OFDM symbol.
     *
     * The symbol is processed in four 16-bin vectors. Each vector is normalized
     * and passed through the 3-to-hidden, hidden-to-hidden, and hidden-to-bit
     * layers. Per-bin decisions are first written to d_decision_lanes; only the
     * 48 data-subcarrier decisions are then copied to the caller's output.
     *
     * @param equalized Matched-filter complex values for all FFT bins.
     * @param channel_power Channel-power estimates for all FFT bins.
     * @param decisions Output storage for packed data-subcarrier decisions.
     * @param model Index of the modulation-specific network to execute.
     */
    static inline void forward(
            const cint32 equalized[FFT_SIZE],
            const int32_t channel_power[FFT_SIZE],
            int8_t decisions[64], unsigned model) {
        aie::scoped_mode rnd_mode{aie::rounding_mode::floor};
        aie::scoped_mode sat_mode{aie::saturation_mode::saturate};

        const unsigned shift1 = frame_equalizer_ai::kShifts[model][0];
        const unsigned shift2 = frame_equalizer_ai::kShifts[model][1];
        const unsigned bit_count = frame_equalizer_ai::kBitCounts[model];
        const unsigned input_bits = AiBlockNormalize::symbol_bits(
            equalized, channel_power);

        for (int bin = 0; bin < FFT_SIZE; bin += 16) {
            const auto parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(equalized + bin));
            const auto input = AiBlockNormalize::forward(
                aie::filter_even(parts, 1), //filter_even(buf, step)
                aie::filter_odd(parts, 1),
                aie::load_v<16>(channel_power + bin), input_bits);
            AiLinear3x16Relu::forward(
                input, d_ai_hidden1, frame_equalizer_ai::kW1[model], shift1);
            AiLinear16x16Relu::forward(
                d_ai_hidden1, d_ai_hidden2,
                frame_equalizer_ai::kW2[model], shift2);
            aie::store_v(
                d_decision_lanes + bin,
                AiLinear16xBits::forward(
                    d_ai_hidden2, frame_equalizer_ai::kW3[model], bit_count));
        }
        copy_data_decisions(d_decision_lanes, decisions);
    }
};

} // namespace

#endif
