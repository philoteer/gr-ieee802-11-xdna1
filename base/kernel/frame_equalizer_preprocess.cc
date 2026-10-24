#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#define AIE_TRIG_ATAN2_FUNCTION_QUALIFIER __attribute__((noinline))
#include "aie_trig_utils.h"
#undef AIE_TRIG_ATAN2_FUNCTION_QUALIFIER
#include "frame_equalizer_metadata.h"

#define FFT_SIZE 64
#define SAMPLES_WITH_CP 80

namespace {
#define NOINLINE __attribute__((noinline))

alignas(aie::vector_decl_align) static constexpr uint32_t kPhaseLaneIndices[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};

// Standard 127-symbol pilot-polarity sequence. Index zero belongs to SIGNAL.
const int8_t kPolarity[127] = {
    1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1,
    -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
    -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
    -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1,
    -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1,
    -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
};

alignas(aie::vector_decl_align) static cint32 d_symbol[FFT_SIZE];
static int d_current_symbol = -1;
static int32_t d_sampling_angle_q29;
static int32_t d_residual_sampling_angle_q29;
static int32_t d_sample_rate_over_center_q30;
static cint32 d_pilot_sum;
alignas(aie::vector_decl_align) static cint32 d_pilots[4];
alignas(aie::vector_decl_align) static cint32 d_prev_pilots[4];

// Re-profile (LUT, full production graph, 50-run four-tile 130-byte 16-QAM
// workload): this region averages 2.4k cycles/tile (10.5% of preprocess), down
// from 13.6k with CORDIC; the scalar pilot extraction/polarity step is about
// 0.2k and the interpolated-LUT atan2 is about 0.3k.
static inline int32_t pilot_beta(const cint32 symbol[FFT_SIZE]) {
    if (d_current_symbol < 2) {
        d_pilots[0] = symbol[11];
        d_pilots[1] = -symbol[25];
        d_pilots[2] = symbol[39];
        d_pilots[3] = symbol[53];
    } else {
        const int p = kPolarity[(d_current_symbol - 2) % 127];
        d_pilots[0] = p * symbol[11];
        d_pilots[1] = p * symbol[25];
        d_pilots[2] = p * symbol[39];
        d_pilots[3] = -p * symbol[53];
    }
    d_pilot_sum = d_pilots[0] + d_pilots[1] + d_pilots[2] + d_pilots[3];
    return atan2_estimate_lut_q29(d_pilot_sum.imag, d_pilot_sum.real);
}

static inline int32_t phase_of_sum(int64_t real, int64_t imag) {
    const uint64_t abs_real = real < 0 ? 0ULL - (uint64_t)real : (uint64_t)real;
    const uint64_t abs_imag = imag < 0 ? 0ULL - (uint64_t)imag : (uint64_t)imag;
    uint64_t magnitude = abs_real | abs_imag;
    int shift = 0;
    if (magnitude > 0x7ffffffeULL) {
        shift = 33 - __builtin_clzll(magnitude);
        if ((magnitude >> shift) > 0x7ffffffeULL) ++shift;
    }
    return atan2_estimate_lut_q29(
        (int32_t)(imag >> shift), (int32_t)(real >> shift));
}

// Re-profile (LUT, full production graph): this region averages 3.6k
// cycles/tile (15.9% of preprocess), down from 12.4k with CORDIC. Correlation
// and extraction remain about 0.7k; the rest is the LUT phase estimate plus the
// Q29 sampling-rate scaling.
static inline int32_t update_sampling_error() {
    const auto previous = aie::load_v<4>(d_prev_pilots);
    const auto current = aie::load_v<4>(d_pilots);
    const auto correlation = aie::sliding_mul<2, 4, 1, 1, 1, cacc64>(
        aie::op_conj(previous), 0, current, 0);
    const auto components = correlation.template cast_to<acc64>();
    const auto words = aie::vector_cast<uint32_t>(components);
    const auto lower = aie::filter_even(words, 1);
    const auto upper = aie::filter_odd(words, 1);
    const uint64_t real_bits = (uint64_t)lower[0] | ((uint64_t)upper[0] << 32);
    const uint64_t imag_bits = (uint64_t)lower[1] | ((uint64_t)upper[1] << 32);
    aie::store_v(d_prev_pilots, current);
    if (d_current_symbol < 2) return 0;
    const int32_t phase_q29 = phase_of_sum((int64_t)real_bits, (int64_t)imag_bits);
    const int64_t scaled_phase =
        (int64_t)phase_q29 * d_sample_rate_over_center_q30;
    return (int32_t)(scaled_phase >> 30) / SAMPLES_WITH_CP;
}

// Remove the subcarrier-dependent sampling-error phase ramp. The phase is
// zero at FFT bin 32, so the oscillator phase for each lane is (bin - 32)*step.
// Build bins 32..47 with one trigonometric evaluation, then advance or rewind
// that oscillator by 16 bins with Q15 complex multiplies. Recurrence error is
// bounded to two block advances because the oscillator is rebuilt every symbol.
// Re-profile (full production graph, linear sine LUT): 6.8k cycles/tile (29.8%
// of preprocess), down from 9.3k with minimax sine/cosine and 30.5k before the
// 16-bin recurrence optimization.
static NOINLINE void rotate_sampling_ramp(cint32 *symbol, int32_t step_q29,
                                          SinCosLinearLutQ15 &sincos_lut) {
    const uint32_t step = radians_q29_to_phase_q(step_q29);
    aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
    aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
    aie::vector<uint32_t, 16> phase_offsets_v;
    {
        aie::scoped_mode sat_mode{aie::saturation_mode::none};
        phase_offsets_v = aie::mul<acc64>(
            aie::load_v<16>(kPhaseLaneIndices), step).to_vector<uint32_t>(0);
    }

    const auto oscillator = aie::interleave_zip(
        sincos_lut.cos(phase_offsets_v),
        sincos_lut.sin(phase_offsets_v), 1);
    const auto oscillator_32 = aie::vector_cast<cint16>(
        aie::concat(oscillator.first, oscillator.second));

    // Lane 8 is exp(j*8*step); squaring it estimates the 16-bin block factor.
    const cint16 rotation_8 = oscillator_32[8];
    const auto rotation_8_v = aie::broadcast<cint16, 16>(rotation_8);
    const auto rotation_16_v = aie::mul<cacc48>(rotation_8_v, rotation_8_v)
                                   .to_vector<cint16>(15);
    const cint16 rotation_16 = rotation_16_v[0];
    const cint16 rotation_minus_16 = {
        rotation_16.real,
        (int16_t)(rotation_16.imag == INT16_MIN
                      ? INT16_MAX
                      : -rotation_16.imag),
    };
    const auto rotation_minus_16_v =
        aie::broadcast<cint16, 16>(rotation_minus_16);

    const auto oscillator_48 = aie::mul<cacc48>(oscillator_32, rotation_16_v)
                                   .to_vector<cint16>(15);
    const auto oscillator_16 =
        aie::mul<cacc48>(oscillator_32, rotation_minus_16_v)
            .to_vector<cint16>(15);
    const auto oscillator_0 =
        aie::mul<cacc48>(oscillator_16, rotation_minus_16_v)
            .to_vector<cint16>(15);

    aie::store_v(
        symbol,
        aie::mul<cacc48>(aie::load_v<16>(symbol), oscillator_0)
            .to_vector<cint32>(15));
    aie::store_v(
        symbol + 16,
        aie::mul<cacc48>(aie::load_v<16>(symbol + 16), oscillator_16)
            .to_vector<cint32>(15));
    aie::store_v(
        symbol + 32,
        aie::mul<cacc48>(aie::load_v<16>(symbol + 32), oscillator_32)
            .to_vector<cint32>(15));
    aie::store_v(
        symbol + 48,
        aie::mul<cacc48>(aie::load_v<16>(symbol + 48), oscillator_48)
            .to_vector<cint32>(15));
}

// Remove the common pilot phase. All 64 bins use the same oscillator, so it
// is constructed once and reused for every vector in the symbol. Re-profile
// (full production graph, linear sine LUT): 5.8k cycles/tile (25.6% of
// preprocess), down from 8.4k with minimax sine/cosine.
static NOINLINE void rotate_common_phase(cint32 *symbol, int32_t beta_q29,
                                          int count,
                                          SinCosLinearLutQ15 &sincos_lut) {
    const uint32_t beta = radians_q29_to_phase_q(-beta_q29);
    aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
    aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
    const auto phase_v = aie::broadcast<uint32_t, 16>(beta);
    const auto oscillator = aie::interleave_zip(
        sincos_lut.cos(phase_v),
        sincos_lut.sin(phase_v), 1);
    const auto oscillator_v = aie::vector_cast<cint16>(
        aie::concat(oscillator.first, oscillator.second));
    for (int base = 0; base < count; base += 16) {
        const auto sample_v = aie::load_v<16>(symbol + base);
        const auto rotated = aie::mul<cacc48>(sample_v, oscillator_v);
        aie::store_v(symbol + base, rotated.to_vector<cint32>(15));
    }
}

static NOINLINE void reset_preprocessing(int32_t cfo_q29) {
    d_current_symbol = 0;
    d_sampling_angle_q29 =
        (int32_t)((int64_t)cfo_q29 * d_sample_rate_over_center_q30 >> 30);
    d_residual_sampling_angle_q29 = 0;
}
} // namespace

extern "C" void frame_equalizer_preprocess(
    const cint32 *__restrict in, const int32_t *__restrict in_meta,
    cint32 *__restrict prepared, int32_t *__restrict prepared_meta,
    const int32_t *__restrict center_frequency, int32_t N, int32_t meta_N) {
    auto *metadata =
        reinterpret_cast<frame_equalizer_prepared_tile_metadata *>(prepared_meta);
    SinCosLinearLutQ15 sincos_lut;
    d_sample_rate_over_center_q30 = center_frequency[1];
    metadata->center_frequency_mhz = center_frequency[0];

    // Preserve the FFT metadata for the second kernel. Its frame state is
    // independent because the kernels run on different AIE tiles.
    for (int i = 0; i < FRAME_EQUALIZER_INPUT_METADATA_WORDS; i += 16)
        aie::store_v(metadata->input + i, aie::load_v<16>(in_meta + i));

    const int input_count = in_meta[0];
    const int input_tags = in_meta[1];
    int tag_index = 0;
    for (int offset = 0; offset + FFT_SIZE <= input_count; offset += FFT_SIZE) {
        while (tag_index < input_tags && in_meta[2 + 2 * tag_index] <= offset) {
            reset_preprocessing(in_meta[3 + 2 * tag_index]);
            ++tag_index;
        }

        const int slot = offset / FFT_SIZE;
        if (d_current_symbol < 0) {
            metadata->beta[slot] = 0;
            for (int bin = 0; bin < FFT_SIZE; bin += 16)
                aie::store_v(prepared + offset + bin,
                             aie::load_v<16>(in + offset + bin));
            continue;
        }

        for (int bin = 0; bin < FFT_SIZE; bin += 16)
            aie::store_v(d_symbol + bin, aie::load_v<16>(in + offset + bin));

        //here be dragons
        const int32_t sampling_step = (int32_t)(
            (int64_t)(d_sampling_angle_q29 + d_residual_sampling_angle_q29) *
            d_current_symbol * 80 / 64);
        rotate_sampling_ramp(d_symbol, sampling_step, sincos_lut);
        const int32_t beta = pilot_beta(d_symbol);
        const int32_t residual = update_sampling_error();
        rotate_common_phase(d_symbol, beta, FFT_SIZE, sincos_lut);
        if (d_current_symbol >= 2)
            d_residual_sampling_angle_q29 +=
                (residual - d_residual_sampling_angle_q29) >> 3;

        for (int bin = 0; bin < FFT_SIZE; bin += 16)
            aie::store_v(prepared + offset + bin, aie::load_v<16>(d_symbol + bin));
        // Beta is indexed by the symbol's position within this tile. The
        // equalization kernel consumes it alongside the corrected symbol.
        metadata->beta[slot] = beta;
        ++d_current_symbol;
    }
}
