#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_trig_utils.h"

extern "C" void trig_utils_test(const int32_t *__restrict packed_in,
                                 int32_t *__restrict packed_out,
                                 int32_t sample_count) {
    const uint32_t *phase = reinterpret_cast<const uint32_t *>(packed_in);
    const int32_t *y = packed_in + sample_count;
    const int32_t *x = packed_in + 2 * sample_count;
    int32_t *sin_minimax = packed_out;
    int32_t *cos_minimax = packed_out + sample_count;
    int32_t *sin_linear = packed_out + 2 * sample_count;
    int32_t *cos_linear = packed_out + 3 * sample_count;
    int32_t *cordic = packed_out + 4 * sample_count;
    int32_t *atan_lut = packed_out + 5 * sample_count;
    int32_t *minimax_cycles = packed_out + 6 * sample_count;
    int32_t *linear_cycles = packed_out + 7 * sample_count;
    int32_t *cordic_cycles = packed_out + 8 * sample_count;
    int32_t *atan_lut_cycles = packed_out + 9 * sample_count;

    const uint64_t minimax_start = aie::tile::current().cycles();
    for (int32_t i = 0; i < sample_count; i += 16) {
        const auto phase_v = aie::load_v<16>(phase + i);
        const auto sin_v = sin_estimate_MINIMAX_v(phase_v);
        const auto cos_v = cos_estimate_MINIMAX_v(phase_v);
        alignas(32) int16_t sin_values[16];
        alignas(32) int16_t cos_values[16];
        aie::store_v(sin_values, sin_v);
        aie::store_v(cos_values, cos_v);
        for (int lane = 0; lane < 16; ++lane) {
            sin_minimax[i + lane] = sin_values[lane];
            cos_minimax[i + lane] = cos_values[lane];
        }
    }
    minimax_cycles[0] = (int32_t)(
        aie::tile::current().cycles() - minimax_start);

    SinCosLinearLutQ15 sincos_lut;
    const uint64_t linear_start = aie::tile::current().cycles();
    for (int32_t i = 0; i < sample_count; i += 16) {
        const auto phase_v = aie::load_v<16>(phase + i);
        const auto sin_v = sincos_lut.sin(phase_v);
        const auto cos_v = sincos_lut.cos(phase_v);
        alignas(32) int16_t sin_values[16];
        alignas(32) int16_t cos_values[16];
        aie::store_v(sin_values, sin_v);
        aie::store_v(cos_values, cos_v);
        for (int lane = 0; lane < 16; ++lane) {
            sin_linear[i + lane] = sin_values[lane];
            cos_linear[i + lane] = cos_values[lane];
        }
    }
    linear_cycles[0] = (int32_t)(
        aie::tile::current().cycles() - linear_start);

    const uint64_t cordic_start = aie::tile::current().cycles();
    for (int32_t i = 0; i < sample_count; ++i)
        cordic[i] = atan2_estimate_CORDIC_q29(y[i], x[i]);
    cordic_cycles[0] = (int32_t)(aie::tile::current().cycles() - cordic_start);

    const uint64_t lut_start = aie::tile::current().cycles();
    for (int32_t i = 0; i < sample_count; ++i)
        atan_lut[i] = atan2_estimate_lut_q29(y[i], x[i]);
    atan_lut_cycles[0] = (int32_t)(
        aie::tile::current().cycles() - lut_start);
}
