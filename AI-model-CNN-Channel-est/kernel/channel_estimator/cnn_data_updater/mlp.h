#ifndef FRAME_EQUALIZER_CNN_DATA_UPDATER_MLP_H
#define FRAME_EQUALIZER_CNN_DATA_UPDATER_MLP_H

#include <stdint.h>

#include "../common.h"
#include "../cnn_common.h"
#include "../model/select_updater.h"

namespace {

alignas(aie::vector_decl_align) static cint32 d_mlp_lts_channel[FFT_SIZE];
alignas(aie::vector_decl_align) static int16_t d_mlp_features[64];
alignas(aie::vector_decl_align) static int16_t d_mlp_hidden[16];
alignas(aie::vector_decl_align) static int16_t d_mlp_output[32];
alignas(aie::vector_decl_align) static int32_t d_mlp_rescaled[32];
alignas(aie::vector_decl_align) static int32_t d_mlp_base_parts[32];
const int kMlpPilotBins[4] = {11, 25, 39, 53};
const int kMlpOccupiedStart[2] = {6, 33};

static inline uint32_t mlp_magnitude(int32_t value) {
    return value < 0 ? uint32_t(0) - uint32_t(value) : uint32_t(value);
}

static inline int16_t mlp_normalize(int32_t value, int shift) {
    return shift >= 0 ? int16_t(value * (1 << shift)) : int16_t(value >> -shift);
}

static inline int16_t mlp_clamp_q15(int32_t value) {
    return value > 32767 ? 32767 : value < -32768 ? -32768 : value;
}

struct MlpInput {
    cint32 pilots[4];
    int shift;
    int16_t snr_feature;
};

struct MlpInputPrepare {
    static inline MlpInput forward(const cint32 *symbol, int number,
                                   uint64_t snr_ratio_x16) {
        MlpInput input;
        const int polarity = kPolarity[(number - 2) % 127];
        for (int i = 0; i < 4; ++i) {
            const int sign = (i == 3 ? -polarity : polarity);
            input.pilots[i] = {sign * symbol[kMlpPilotBins[i]].real,
                               sign * symbol[kMlpPilotBins[i]].imag};
        }
        auto largest = aie::zeros<int32_t, 16>();
        auto smallest = largest;
        for (int half = 0; half < 2; ++half) {
            const cint32 *channel = d_mlp_lts_channel + kMlpOccupiedStart[half];
            for (int bin = 0; bin < 26; bin += 8) {
                const int offset = bin == 24 ? 18 : bin;
                const auto parts = aie::vector_cast<int32_t>(aie::load_unaligned_v<8>(channel + offset));
                largest = aie::max(largest, parts);
                smallest = aie::min(smallest, parts);
            }
        }
        uint32_t maximum = uint32_t(aie::reduce_max(largest));
        const uint32_t negative_maximum = mlp_magnitude(aie::reduce_min(smallest));
        maximum = maximum > negative_maximum ? maximum : negative_maximum;
        for (int i = 0; i < 4; ++i) {
            maximum = maximum > mlp_magnitude(input.pilots[i].real) ? maximum : mlp_magnitude(input.pilots[i].real);
            maximum = maximum > mlp_magnitude(input.pilots[i].imag) ? maximum : mlp_magnitude(input.pilots[i].imag);
        }
        input.shift = 15 - (maximum ? 32 - __builtin_clz(maximum) : 0);
        input.snr_feature = int16_t(CnnInputNormalize::snr_feature_q15(snr_ratio_x16));
        return input;
    }
};

struct MlpFeatureEncode {
    static inline void forward(int group, const MlpInput &input) {
        const cint32 *half = d_mlp_lts_channel + kMlpOccupiedStart[group / 2];
        const unsigned down_shift = input.shift < 0 ? unsigned(-input.shift) : 0u;
        const int32_t up_scale = input.shift > 0 ? 1 << input.shift : 1;
        for (int bin = 0; bin < 24; bin += 8) {
            const auto parts = aie::vector_cast<int32_t>(aie::load_unaligned_v<8>(half + bin));
            const auto normalized = aie::mul<acc64>(parts, up_scale).to_vector<int16_t>(down_shift);
            aie::store_v(d_mlp_features + 2 * bin, normalized);
        }
        for (int bin = 24; bin < 26; ++bin) {
            d_mlp_features[2 * bin] = mlp_normalize(half[bin].real, input.shift);
            d_mlp_features[2 * bin + 1] = mlp_normalize(half[bin].imag, input.shift);
        }
        const cint32 *base = d_mlp_lts_channel + kMlpOccupiedStart[group / 2] + (group % 2) * 13;
        aie::store_v(d_mlp_base_parts,
            aie::vector_cast<int32_t>(aie::load_unaligned_v<8>(base)));
        for (int bin = 8; bin < 13; ++bin) {
            d_mlp_base_parts[2 * bin] = base[bin].real;
            d_mlp_base_parts[2 * bin + 1] = base[bin].imag;
        }
        for (int i = 0; i < 4; ++i) {
            const cint32 base = d_mlp_lts_channel[kMlpPilotBins[i]];
            const int32_t real = int32_t(mlp_normalize(input.pilots[i].real, input.shift)) -
                                 mlp_normalize(base.real, input.shift);
            const int32_t imag = int32_t(mlp_normalize(input.pilots[i].imag, input.shift)) -
                                 mlp_normalize(base.imag, input.shift);
            d_mlp_features[52 + 2 * i] = mlp_clamp_q15(real);
            d_mlp_features[53 + 2 * i] = mlp_clamp_q15(imag);
        }
        d_mlp_features[60] = input.snr_feature;
        d_mlp_features[61] = d_mlp_features[62] = d_mlp_features[63] = 0;
    }
};

template <unsigned Inputs, unsigned Outputs, bool Relu>
struct MlpLinear {
    static inline void forward(const int16_t *source, int16_t *destination,
                               const int8_t weights[Inputs][Outputs]) {
        const auto zero = aie::zeros<int16_t, 16>();
        for (unsigned output = 0; output < Outputs; output += 16) { //for each output 
            auto accum = aie::mul<acc32>(                              //Mul(input #0, weights[0])
                aie::broadcast<int16_t, 16>(source[0]), 
                aie::load_v<16>(weights[0] + output));
            AIE_LOOP_NO_UNROLL
            for (unsigned input = 1; input < Inputs; ++input) {     //for each input
                accum = aie::mac(accum,                             //mac (input #1, weights[input])
                    aie::broadcast<int16_t, 16>(source[input]),
                    aie::load_v<16>(weights[input] + output));
            }
            auto values = accum.template to_vector<int16_t>(7);
            if (Relu) values = aie::max(values, zero);
            aie::store_v(destination + output, values);
        }
    }
};

struct MlpResidualAdd {
    static inline void forward(int group, int shift, cint32 channel[FFT_SIZE]) {
        for (int block = 0; block < 32; block += 16) {
            auto corrections = aie::unpack(aie::load_v<16>(d_mlp_output + block));
            corrections = shift >= 0 ? aie::downshift(corrections, shift)
                                     : aie::upshift(corrections, -shift);
            aie::store_v(d_mlp_rescaled + block,
                aie::saturating_add(aie::load_v<16>(d_mlp_base_parts + block), corrections));
        }
        const int first_bin = kMlpOccupiedStart[group / 2] + (group % 2) * 13;
        for (int lane = 0; lane < 13; ++lane)
            channel[first_bin + lane] = {d_mlp_rescaled[2 * lane], d_mlp_rescaled[2 * lane + 1]};
    }
};

struct CnnDataUpdater {
    static inline void initialize(const cint32 *channel) {
        // NPU profile: 39 cycles once per LTS initialization.
        for (int bin = 0; bin < FFT_SIZE; bin += 8)
            aie::store_v(d_mlp_lts_channel + bin, aie::load_v<8>(channel + bin));
    }

    // NPU profile (20 runs, five calls per run, instrumented): ~5,058 cycles
    // per active post-gate call (previously ~5,568 with the same counters).
    static NOINLINE void forward(const cint32 *symbol, int number,
                                 uint64_t snr_ratio_x16, cint32 channel[FFT_SIZE]) {
        // The frame path leaves channel at the initialized estimate when gated.
        if (snr_ratio_x16 >= frame_equalizer_channel_updater_mlp::kGateSnrRatioX16)
            return;

        // Active calls restore the cached estimate (~40 cycles) before adding a
        // new residual, rather than accumulating updates across payload symbols.
        for (int bin = 0; bin < FFT_SIZE; bin += 8)
            aie::store_v(channel + bin, aie::load_v<8>(d_mlp_lts_channel + bin));

        // Pilot, occupied-bin magnitude, and scale setup: ~1,678 cycles/call.
        const MlpInput input = MlpInputPrepare::forward(symbol, number, snr_ratio_x16);
        aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
        aie::scoped_mode rnd_mode{aie::rounding_mode::floor};
        for (int group = 0; group < 4; ++group) {
            // Feature encoding across four groups: ~860 cycles/call (was ~17,192).
            MlpFeatureEncode::forward(group, input);
            // First dense layer (64x16): ~1,408 cycles across four groups.
            MlpLinear<64, 16, true>::forward(d_mlp_features, d_mlp_hidden,
                frame_equalizer_channel_updater_mlp::kW1[group]);
            // Second dense layer (16x32): ~748 cycles across four groups.
            MlpLinear<16, 32, false>::forward(d_mlp_hidden, d_mlp_output,
                frame_equalizer_channel_updater_mlp::kW2[group]);
            // Rescale, add residual, and write 13 occupied bins per group: ~583 cycles/call.
            MlpResidualAdd::forward(group, input.shift, channel);
        }
    }
};

} // namespace

#endif
