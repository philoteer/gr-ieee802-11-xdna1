//===- aie_trig_utils.h -----------------------------------------*- C++ -*-===//
// Fixed-point trigonometric helpers for AIE kernels.
//===----------------------------------------------------------------------===//

#ifndef AIE_TRIG_UTILS_H
#define AIE_TRIG_UTILS_H
// AIE vectors contain 16 cint16 samples; each component uses Q0.15.
#define _vec_factor 16
#define _q_shift 15

#include <stdint.h>
#include <algorithm>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_atan_lut_q29.h"
#include "aie_sin_linear_lut_q15.h"

#ifndef AIE_TRIG_ATAN2_FUNCTION_QUALIFIER
#define AIE_TRIG_ATAN2_FUNCTION_QUALIFIER inline
#define AIE_TRIG_ATAN2_FUNCTION_QUALIFIER_LOCAL
#endif


// Higher-precision degree-11 Taylor evaluation on 0 <= x <= pi/2.  Q29 is
// used internally because (pi/2)^2 fits in signed Q29 but not signed Q30.
// The Q0.15 output quantization, rather than polynomial error, is dominant.
static __attribute__((noinline)) aie::vector<int16_t, 16> sin_first_quadrant_MINIMAX_q15(aie::vector<int32_t, 16> phase_v) {
    constexpr int32_t half_pi_q29 = 843314857;
    constexpr int32_t sin_a_q29 = 536870912;
    constexpr int32_t sin_b_q29 = -89478485;
    constexpr int32_t sin_c_q29 = 4473924;
    constexpr int32_t sin_d_q29 = -106522;
    constexpr int32_t sin_e_q29 = 1479;
    constexpr int32_t sin_f_q29 = -13;
    aie::scoped_mode rnd_mode{aie::rounding_mode::positive_inf};
    aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
    aie::vector<int32_t, 16> x_v = aie::mul<acc64>(phase_v, half_pi_q29).template to_vector<int32_t>(30);
    aie::vector<int32_t, 16> x2_v = aie::mul<acc64>(x_v, x_v).template to_vector<int32_t>(29);
    aie::vector<int32_t, 16> poly_v = aie::add(aie::mul<acc64>(x2_v, sin_f_q29).template to_vector<int32_t>(29), sin_e_q29);
    poly_v = aie::add(aie::mul<acc64>(x2_v, poly_v).template to_vector<int32_t>(29), sin_d_q29);
    poly_v = aie::add(aie::mul<acc64>(x2_v, poly_v).template to_vector<int32_t>(29), sin_c_q29);
    poly_v = aie::add(aie::mul<acc64>(x2_v, poly_v).template to_vector<int32_t>(29), sin_b_q29);
    poly_v = aie::add(aie::mul<acc64>(x2_v, poly_v).template to_vector<int32_t>(29), sin_a_q29);
    return aie::mul<acc64>(x_v, poly_v).template to_vector<int16_t>(43);
}

static __attribute__((noinline)) aie::vector<int16_t, 16> sin_estimate_MINIMAX_v(aie::vector<uint32_t, 16> phase_v) {
    constexpr uint32_t quadrant_phase = 0x40000000u;
    aie::vector<uint32_t, 16> quadrant_v = aie::logical_downshift(phase_v, 30);
    aie::vector<uint32_t, 16> offset_v = aie::bit_and(quadrant_phase - 1, phase_v);
    aie::mask<16> odd_quadrant = aie::neq(aie::bit_and(1u, quadrant_v), 0u);
    aie::vector<uint32_t, 16> folded_v = aie::select(offset_v, aie::sub(quadrant_phase, offset_v), odd_quadrant);
    aie::vector<int16_t, 16> magnitude_v = sin_first_quadrant_MINIMAX_q15(aie::vector_cast<int32_t>(folded_v));
    aie::mask<16> negative_half = aie::neq(aie::bit_and(2u, quadrant_v), 0u);
    return aie::select(magnitude_v, aie::neg(magnitude_v), negative_half);
}

static inline aie::vector<int16_t, 16> cos_estimate_MINIMAX_v(aie::vector<uint32_t, 16> phase_v) {
    constexpr uint32_t quadrant_phase = 0x40000000u;
    return sin_estimate_MINIMAX_v(aie::add(phase_v, quadrant_phase));
}

// Preprocess-only alternative using AIE's four-access linear approximation
// hardware. One object owns the API scratch state and should be reused across
// all sine/cosine evaluations in a tile instead of reconstructed per symbol.
class SinCosLinearLutQ15 {
    using lut_type = aie::lut<4, int16_t, int16_t>;

    lut_type lut_;
    aie::linear_approx<int16_t, lut_type> approximation_;

public:
    SinCosLinearLutQ15()
        : lut_(kSinLinearLutEntries, kSinLinearLutAb, kSinLinearLutCd),
          approximation_(lut_, kSinLinearLutStepBits,
                         kSinLinearLutBias, kSinLinearLutStepBits) {}

    inline aie::vector<int16_t, 16> sin(aie::vector<uint32_t, 16> phase_v) {
        //divide to 4 quadrants, sin(1st quadrant) = - sin(3rd_quadrant); 2nd/4rd quadrant are the flipped in x-axis.
        //LUT is only for the first quadrant, the folding makes the 4 quadrants to fall within the LUT.
        constexpr uint32_t quadrant_phase = 0x40000000u;
        const auto quadrant_v = aie::logical_downshift(phase_v, 30);
        const auto offset_v = aie::bit_and(quadrant_phase - 1, phase_v);
        const auto odd_quadrant = aie::neq(aie::bit_and(1u, quadrant_v), 0u);
        const auto folded_v = aie::select(
            offset_v, aie::sub(quadrant_phase, offset_v), odd_quadrant);

        // Map the complete quarter-wave onto signed int16. Centering before
        // the Q30 -> Q16 conversion retains all 16 input bits; saturation maps
        // the exact pi/2 endpoint from +32768 to the representable +32767.
        const auto centered_v = aie::sub(
            aie::vector_cast<int32_t>(folded_v), int32_t(0x20000000));
        aie::accum<acc64, 16> centered_acc;
        centered_acc.from_vector(centered_v);
        aie::scoped_mode rnd_mode{aie::rounding_mode::positive_inf};
        aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
        const auto input_v = centered_acc.to_vector<int16_t>(14);
        const auto magnitude_v = approximation_.compute(input_v)
                                     .template to_vector<int16_t>(
                                         kSinLinearLutStepBits);
        const auto negative_half = aie::neq(aie::bit_and(2u, quadrant_v), 0u);
        return aie::select(magnitude_v, aie::neg(magnitude_v), negative_half);
    }

    inline aie::vector<int16_t, 16> cos(aie::vector<uint32_t, 16> phase_v) {
        return sin(aie::add(phase_v, uint32_t(0x40000000u)));
    }
};

static inline int32_t mul_q29_q30(int32_t a_q29, int32_t b_q30) {
    return (int32_t)(((int64_t)a_q29 * b_q30) >> 30);
}

// Debug-quality atan2 using 24 CORDIC vectoring iterations.  This avoids the
// fast estimator's degree-5 residual polynomial while retaining fixed-point
// inputs and Q3.29 output.  The input pair is roughly normalized below so
// arithmetic shifts retain useful angle information even for small vectors.
static AIE_TRIG_ATAN2_FUNCTION_QUALIFIER int32_t atan2_estimate_CORDIC_q29(int32_t y, int32_t x) {
    constexpr int32_t pi_q29 = 1686629713;
    constexpr int32_t cordic_atan_q29[] = {
        421657428, 248918915, 131521918, 66762579, 33510843, 16771758,
        8387925, 4194219, 2097141, 1048575, 524288, 262144,
        131072, 65536, 32768, 16384, 8192, 4096,
        2048, 1024, 512, 256, 128, 64,
    };
    if (x == 0 && y == 0)
        return 0;

    // Halve the one unrepresentable signed endpoint before keeping the
    // normal path entirely in int32_t.  The angle change is below one input
    // LSB for this exceptional case.
    if (x == INT32_MIN || y == INT32_MIN) {
        x /= 2;
        y /= 2;
    }

    int32_t x_v = x;
    int32_t y_v = y;
    int64_t angle_q29 = 0;
    if (x_v < 0) {
        angle_q29 = y_v >= 0 ? pi_q29 : -pi_q29;
        x_v = -x_v;
        y_v = -y_v;
    }
    // CORDIC is scale invariant mathematically, but its arithmetic shifts are
    // not. Raise small vectors into a useful fixed-point range without using
    // signed left shifts; the values remain far below int64_t's limit.
    constexpr int32_t normalization_target = 1 << 27;
    const int32_t abs_y = y_v < 0 ? -y_v : y_v;
    int32_t magnitude = x_v > abs_y ? x_v : abs_y;
    while (magnitude > 0 && magnitude < normalization_target) {
        x_v *= 16;
        y_v *= 16;
        magnitude *= 16;
    }
    int64_t x_wide = x_v;
    int64_t y_wide = y_v;
    for (int32_t i = 0; i < 24; ++i) {
        const int64_t x_shift = x_wide >> i;
        const int64_t y_shift = y_wide >> i;
        if (y_wide > 0) {
            x_wide += y_shift;
            y_wide -= x_shift;
            angle_q29 += cordic_atan_q29[i];
        } else if (y_wide < 0) {
            x_wide -= y_shift;
            y_wide += x_shift;
            angle_q29 -= cordic_atan_q29[i];
        }
    }
    return (int32_t)angle_q29;
}

// LUT alternative for users that can trade 2 KiB of local storage for lower
// latency. Reduce to one octant, linearly interpolate atan(min/max) over 512
// intervals, then restore the quadrant. The Q24 ratio leaves interpolation and
// ratio quantization well below one Q3.29 microradian.
static AIE_TRIG_ATAN2_FUNCTION_QUALIFIER int32_t atan2_estimate_lut_q29(
        int32_t y, int32_t x) {
    constexpr int32_t pi_4_q29 = 421657428;
    constexpr int32_t pi_2_q29 = 843314857;
    constexpr int32_t pi_q29 = 1686629713;
    const uint32_t abs_x = x < 0 ? (uint32_t)(-(int64_t)x) : (uint32_t)x;
    const uint32_t abs_y = y < 0 ? (uint32_t)(-(int64_t)y) : (uint32_t)y;
    const bool swap = abs_y > abs_x;
    const uint32_t major = swap ? abs_y : abs_x;
    const uint32_t minor = swap ? abs_x : abs_y;
    if (major == 0)
        return 0;

    // Normalize the denominator to unsigned Q1.31, refine an 8-bit reciprocal
    // seed twice in Q2.30, then form nine index and fifteen interpolation bits.
    const int normalization_shift = __builtin_clz(major);
    const uint32_t major_q31 = major << normalization_shift;
    const uint32_t minor_q31 = minor << normalization_shift;
    uint32_t reciprocal_q30 =
        kReciprocalSeedQ30[(major_q31 >> 23) - 256];
        
    for (int i = 0; i < 2; ++i) {
        //major_q31  * reciprocal_q30 = 1 (wanted outcome)
        const uint32_t product_q30 = (uint32_t)(
            ((uint64_t)major_q31 * reciprocal_q30 + (1ULL << 30)) >> 31);
        const uint32_t correction_q30 = (1U << 31) - product_q30;   //deviation
        reciprocal_q30 = (uint32_t)(        //iterate (newton's)
            ((uint64_t)reciprocal_q30 * correction_q30 + (1ULL << 29)) >> 30);
    }
    
    
    uint32_t position_q24 = (uint32_t)(
        ((uint64_t)minor_q31 * reciprocal_q30 + (1ULL << 36)) >> 37);
    if (position_q24 > (1U << 24))
        position_q24 = 1U << 24;
    const uint32_t index = position_q24 >> 15;
    int32_t reduced_q29;
    if (index == 512) {
        reduced_q29 = pi_4_q29;
    } else {
        const uint32_t fraction_q15 = position_q24 & 0x7fff;
        const int32_t low_q29 = kAtanLutQ29[index];
        const int32_t high_q29 =
            index == 511 ? pi_4_q29 : kAtanLutQ29[index + 1];
        reduced_q29 = low_q29 + (int32_t)(
            ((int64_t)(high_q29 - low_q29) * fraction_q15 + (1 << 14)) >> 15);
    }

    int32_t angle_q29 = swap ? pi_2_q29 - reduced_q29 : reduced_q29;
    if (x < 0)
        angle_q29 = pi_q29 - angle_q29;
    return y < 0 ? -angle_q29 : angle_q29;
}

// The IEEE 802.11 short training sequence repeats every 16 samples.  A
// 16-sample-lag autocorrelation therefore has phase 16*omega; dividing its
// angle by 16 estimates omega in Q3.29 radians/sample.
static inline int32_t estimate_freq_offset_q29(cint32 autocorr) {
    return atan2_estimate_CORDIC_q29(autocorr.imag, autocorr.real) / 16;
}

// Converts signed Q3.29 radians to the uint32 phase where one turn is 2^32.
static inline uint32_t radians_q29_to_phase_q(int32_t radians_q29) {
    // round((4/pi) * 2^30), paired with mul_q29_q30's Q30 shift.
    constexpr int32_t phase_per_radian_q30 = 1367130551;
    return (uint32_t)mul_q29_q30(radians_q29, phase_per_radian_q30);
}

static inline aie::vector<cint16, 16> make_unit_q15_v(aie::vector<int16_t, 16> real_v, aie::vector<int16_t, 16> imag_v) {
    // 32767 is the largest positive Q0.15 value, so this is the desired
    // squared magnitude in the raw Q30 product domain.
    constexpr int32_t unit_mag_sq = 32767 * 32767;
    aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
    aie::scoped_mode sat_mode{aie::saturation_mode::saturate};

    aie::accum<acc64, 16> mag_sq_acc = aie::mul_square<acc64>(real_v);
    mag_sq_acc = aie::mac_square(mag_sq_acc, imag_v);
    aie::vector<int32_t, 16> mag_err_v = aie::sub(mag_sq_acc.template to_vector<int32_t>(0), unit_mag_sq);

    // One Newton/Taylor normalization step uses
    //   1/sqrt(1+e) ~= 1 - e/2.
    // Here e ~= mag_err/2^30.  Expressing the scale in Q15 turns e/2 into
    // mag_err/2^16, explaining both the shift by 16 and unity value 32768.
    // This cheaply removes most radial error introduced by the sine/cosine
    // approximations before the phasor is used recursively.
    aie::vector<int32_t, 16> correction_v = aie::from_vector<acc64>(mag_err_v).template to_vector<int32_t>(16);
    aie::vector<int32_t, 16> scale_v = aie::sub(32768, correction_v);

    aie::vector<int16_t, 16> real_normalized_v = aie::mul<acc64>(real_v, scale_v).template to_vector<int16_t>(15);
    aie::vector<int16_t, 16> imag_normalized_v = aie::mul<acc64>(imag_v, scale_v).template to_vector<int16_t>(15);
    auto normalized_zip = aie::interleave_zip(real_normalized_v, imag_normalized_v, 1);
    return aie::vector_cast<cint16>(aie::concat(normalized_zip.first, normalized_zip.second));
}

// Rotate complex Q0.15 samples by a Q0.15 unit oscillator. The product is
// accumulated as Q0.30 and narrowed back to Q0.15 at the output.
static inline void rotate_16(cint16 *__restrict in, cint16 *__restrict out,
                             aie::vector<cint16, _vec_factor> oscillator) {
    const aie::vector<cint16, _vec_factor> in_vec =
        aie::load_unaligned_v<_vec_factor>(in);
    const aie::accum<cacc48, _vec_factor> rotated = aie::mul(in_vec, oscillator);
    const aie::vector<cint16, _vec_factor> rotated_q15 =
        rotated.to_vector<cint16>(_q_shift);
    aie::store_unaligned_v(out, rotated_q15);
}

static inline void copy_16(cint16 *__restrict in, cint16 *__restrict out) {
    aie::vector<cint16, _vec_factor> in_vec = aie::load_unaligned_v<_vec_factor>(in);
    aie::store_unaligned_v(out, in_vec);
}

#ifdef AIE_TRIG_ATAN2_FUNCTION_QUALIFIER_LOCAL
#undef AIE_TRIG_ATAN2_FUNCTION_QUALIFIER_LOCAL
#undef AIE_TRIG_ATAN2_FUNCTION_QUALIFIER
#endif

#endif
