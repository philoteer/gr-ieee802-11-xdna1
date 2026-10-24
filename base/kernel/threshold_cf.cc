#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#define vec_factor 16
constexpr unsigned autocorr_q = 24;
constexpr unsigned mag_q = 23;
constexpr unsigned threshold_q = 31;
// 0.56 cannot be represented exactly in binary. This is its nearest Q0.31
// integer representation, used so the kernel never needs floating point.
constexpr int32_t threshold_q31 = 1202590843; // round(0.56 * 2^31)

extern "C" {

void threshold_cf(cint32 * __restrict ac_in, int32_t * __restrict mag_in, int8_t * __restrict out, int32_t N) {
    cint32 *__restrict pAc = ac_in;
    int32_t *__restrict pMag = mag_in;
    int8_t *__restrict pOut = out;

    const int num_vectors = N / vec_factor;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < num_vectors; i++) {
        // Load autocorrelation and magnitude streams.
        const aie::vector<cint32, vec_factor> ac_q_vec = aie::load_v<vec_factor>(pAc);
        pAc += vec_factor;
        const aie::vector<int32, vec_factor> mag_q_vec = aie::load_v<vec_factor>(pMag);
        pMag += vec_factor;

        const aie::vector<int32_t, vec_factor> threshold_mag_q24 =
            aie::mul<acc64>(mag_q_vec, threshold_q31)
                .to_vector<int32_t>(mag_q + threshold_q - autocorr_q);

        const aie::vector<int32_t, vec_factor * 2> ac_parts =
            aie::vector_cast<int32_t>(ac_q_vec);
        const aie::vector<int32_t, vec_factor> ac_real =
            aie::filter_even(ac_parts, 1);
        const aie::vector<int32_t, vec_factor> ac_imag =
            aie::filter_odd(ac_parts, 1);

        auto difference_q48 = aie::mul<acc64>(ac_real, ac_real);
        difference_q48 = aie::mac(difference_q48, ac_imag, ac_imag);
        difference_q48 = aie::msc(
            difference_q48, threshold_mag_q24, threshold_mag_q24);

        // AIE vector comparisons operate on vectors rather than acc64 values.
        // Simply shifting the accumulator down could turn a small positive
        // difference into zero. Adding 2^31 - 1 before a floor shift computes
        // ceil(difference / 2^31): it is positive exactly when the original
        // difference is positive, including values as small as one Q48 LSB.
        difference_q48 = aie::add(difference_q48, INT32_MAX);
        const aie::vector<int32_t, vec_factor> difference_sign =
            difference_q48.to_vector<int32_t>(31);
        aie::vector<int8_t, vec_factor> threshold_met =
            aie::select(int8_t(0), int8_t(1),
                        aie::gt(difference_sign, int32_t(0)));

        aie::store_v(pOut, threshold_met);
        pOut += vec_factor;
    }
}
}
