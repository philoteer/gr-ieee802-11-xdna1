#ifndef FRAME_EQUALIZER_MODEL_COMMON_H
#define FRAME_EQUALIZER_MODEL_COMMON_H

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

// Architecture-independent building blocks shared by every learned demapper.
// The selected model header includes its generated weights header first, then
// this file, so the generated shape constants (kHiddenSize, kHiddenLayers)
// are already visible when these helpers are instantiated.

namespace {

using ai_lane_vector = aie::vector<int32_t, 16>;

struct AiInputTensor {
    ai_lane_vector real;
    ai_lane_vector imag;
    ai_lane_vector power;
};

/**
 * Scales a lane vector into the exported model's Q-format input range.
 *
 * The signed difference between kInputFractionalBits and the block's magnitude
 * bits selects the direction: wider blocks are shifted right, narrower blocks
 * are shifted left. Both shifts truncate, so the input stage matches the int64
 * reference's arithmetic shifts. The same operation is applied to every lane
 * so relative I/Q/channel-power magnitudes remain unchanged.
 *
 * @param value Input values sharing the block-wide magnitude estimate.
 * @param bits Significant bits required by the largest value in the block.
 * @return Values scaled to the Q(kInputFractionalBits) input format.
 */
static inline ai_lane_vector ai_scale_to_input(
        const ai_lane_vector &value, unsigned bits) {
    const int shift =
        int(frame_equalizer_ai::kInputFractionalBits) - int(bits);
    if (shift > 0) return aie::upshift(value, unsigned(shift));
    else if (shift < 0) return aie::downshift(value, unsigned(-shift));
    return value;
}

struct AiBlockNormalize {
    /**
     * Normalizes one 16-bin feature block to the model's Q-format input range.
     *
     * A single power-of-two scale is derived from the largest absolute real,
     * imaginary, or channel-power value. Sharing that scale across all three
     * features preserves their ratios. Scaling truncates, so a lane whose
     * magnitude is below the block's Q-format resolution maps to zero.
     *
     * @param real Matched-filter real components for 16 adjacent FFT bins.
     * @param imag Matched-filter imaginary components for the same bins.
     * @param power Channel-power estimates for the same bins.
     * @return The three normalized feature vectors consumed by the first layer.
     */
    static inline AiInputTensor forward(
            const ai_lane_vector &real,
            const ai_lane_vector &imag,
            const ai_lane_vector &power, unsigned bits) {
        AiInputTensor output;
        output.real = ai_scale_to_input(real, bits);
        output.imag = ai_scale_to_input(imag, bits);
        output.power = ai_scale_to_input(power, bits);
        return output;
    }

    static inline unsigned symbol_bits(
            const cint32 equalized[FFT_SIZE],
            const int32_t channel_power[FFT_SIZE]) {
        uint32_t maximum = 0;
        for (int bin = 0; bin < FFT_SIZE; bin += 16) {
            const auto parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(equalized + bin));
            const auto magnitude = aie::max(
                aie::max(aie::abs(aie::filter_even(parts, 1)),
                         aie::abs(aie::filter_odd(parts, 1))),
                aie::load_v<16>(channel_power + bin));
            const uint32_t lane_max = uint32_t(aie::reduce_max(magnitude));
            if (lane_max > maximum) maximum = lane_max;
        }
        return maximum == 0 ? 0 : 32 - __builtin_clz(maximum);
    }
};

struct AiLinear3x16Relu {
    /**
     * Evaluates the first bias-free dense layer for 16 FFT bins in parallel.
     *
     * Each hidden neuron multiplies the real, imaginary, and channel-power
     * features by its trained int8 weights. The acc64 result is shifted,
     * converted to int32, passed through ReLU, and stored contiguously.
     *
     * @param input Three normalized feature vectors, one lane per FFT bin.
     * @param destination Hidden activations, 16 lanes (bins) per neuron.
     * @param weights Model's first-layer matrix, one row per hidden neuron.
     * @param shift Exported fixed-point shift applied to each accumulator.
     */
    static inline void forward(
            const AiInputTensor &input, int32_t *destination,
            const int8_t weights[][3], unsigned shift) {
        const auto zero = aie::zeros<int32_t, 16>();
        // Three MACs per neuron are short enough that unrolling the hidden
        // loop only grows program memory; keep the compiler's rolled schedule.
        AIE_LOOP_NO_UNROLL
        for (unsigned output = 0;
             output < frame_equalizer_ai::kHiddenSize; ++output) {
            const int8_t *row = weights[output];
            auto acc = aie::mul<acc64>(input.real, int32_t(row[0]));
            acc = aie::mac(acc, input.imag, int32_t(row[1]));
            acc = aie::mac(acc, input.power, int32_t(row[2]));
            aie::store_v(
                destination + 16 * output,
                aie::max(acc.to_vector<int32_t>(shift), zero)); //relu
        }
    }
};

struct AiLinear16x16Relu {
    /**
     * Evaluates a bias-free hidden dense layer for 16 FFT bins in parallel.
     *
     * The function reads the previous layer's activations, applies the model's
     * int8 matrix, shifts and rectifies each neuron, then stores the resulting
     * int32 activations.
     *
     * @param source Previous-layer activations, 16 lanes per neuron.
     * @param destination Next-layer activations, 16 lanes per neuron.
     * @param weights Model's layer matrix, one row per output neuron.
     * @param shift Exported fixed-point shift applied to each accumulator.
     */
    static inline void forward(
            const int32_t *source, int32_t *destination,
            const int8_t weights[][16], unsigned shift) {
        const auto zero = aie::zeros<int32_t, 16>();
        AIE_LOOP_NO_UNROLL
        for (unsigned output = 0;
             output < frame_equalizer_ai::kHiddenSize; ++output) {
            const int8_t *row = weights[output];
            auto acc = aie::mul<acc64>(
                aie::load_v<16>(source), int32_t(row[0]));
            // Partial unrolling gives the scheduler independent load/MAC work
            // to overlap with the acc64 chain. Factor 3 measured fastest; full
            // unrolling costs about 1.5 KiB of program memory and is slower.
            AIE_LOOP_UNROLL(3)
            for (unsigned input = 1;
                 input < frame_equalizer_ai::kHiddenSize; ++input) {
                acc = aie::mac(
                    acc, aie::load_v<16>(source + 16 * input),
                    int32_t(row[input]));
            }
            aie::store_v(
                destination + 16 * output,
                aie::max(acc.to_vector<int32_t>(shift), zero)); //relu
        }
    }
};

struct AiLinear16xBits {
    /**
     * Converts final hidden activations into packed constellation decisions.
     *
     * One bias-free output neuron is evaluated for each coded bit. A nonnegative
     * logit sets that bit in every lane's LSB-first constellation index; a
     * negative logit leaves it clear.
     *
     * @param source Final hidden activations, 16 lanes per neuron.
     * @param weights Model's output matrix, one row per coded bit.
     * @param bit_count Number of output bits for the selected constellation.
     * @return Sixteen packed int8 constellation indices, one per FFT bin.
     */
    static inline aie::vector<int8_t, 16> forward(
            const int32_t *source, const int8_t weights[][16],
            unsigned bit_count) {
        auto decisions = aie::zeros<int8_t, 16>();
        AIE_LOOP_NO_UNROLL
        for (unsigned bit = 0; bit < bit_count; ++bit) {
            const int8_t *row = weights[bit];
            auto acc = aie::mul<acc64>(
                aie::load_v<16>(source), int32_t(row[0]));
            // Factor 5 divides the 15-term dot product exactly and measured
            // faster than 3 while staying far below the full-unroll cost.
            AIE_LOOP_UNROLL(5)
            for (unsigned input = 1;
                 input < frame_equalizer_ai::kHiddenSize; ++input) {
                acc = aie::mac(
                    acc, aie::load_v<16>(source + 16 * input),
                    int32_t(row[input]));
            }
            const auto logits = acc.to_vector<int32_t>(0);
            decisions = aie::add(
                decisions,
                aie::select(
                    int8_t(0), int8_t(1 << bit),
                    aie::ge(logits, int32_t(0))));
        }
        return decisions;
    }
};

} // namespace

#endif
