//===- magsq_avg_cf.cc -----------------------------------------*- C++ -*-===//
// Magnitude Squared & Boxcar Filter Kernel
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

constexpr unsigned vec_factor = 16;
constexpr unsigned half_vec_factor = vec_factor / 2;
constexpr unsigned input_q = 15;
constexpr unsigned mag_q = 23;

// Four previous magnitude-squared blocks in int32 Q8.23. Q8.23 represents
// values up to nearly 256, leaving headroom beyond the worst-case 64-sample
// boxcar sum of at most 128.
static aie::vector<int32, vec_factor> h0 = aie::zeros<int32, vec_factor>();
static aie::vector<int32, vec_factor> h1 = aie::zeros<int32, vec_factor>();
static aie::vector<int32, vec_factor> h2 = aie::zeros<int32, vec_factor>();
static aie::vector<int32, vec_factor> h3 = aie::zeros<int32, vec_factor>();

// Previous complex input block, retained as cint16 Q0.15 for out_delay.
static aie::vector<cint16, vec_factor> delay_hist = aie::zeros<cint16, vec_factor>();

extern "C" {

/** @brief CORE 2: Magnitude Squared & Boxcar Filter Kernel.
 *
 *  Squares each complex input sample and emits the running sum of the last 64
 *  magnitude-squared values. The previous input block is written to out_delay
 *  for the short-preamble synchronizer, and static history makes calls
 *  consecutive-block stateful.
 *
 *  @param in Input samples, cint16 Q0.15.
 *  @param out Boxcar sums, int32 Q8.23.
 *  @param out_delay Previous input block, cint16 Q0.15.
 *  @param N Number of input samples; must be a multiple of 16.
 */
void magsq_avg_cf(cint16 * __restrict in, int32_t * __restrict out, cint16 * __restrict out_delay, int32_t N) {
    cint16 *__restrict pIn = in;
    int32_t *__restrict pOut = out;
    cint16 *__restrict pOutDelay = out_delay;

    const int num_vectors = N / vec_factor;
    // All magnitude and boxcar values after the initial multiply are int32 Q8.23.
    const auto zero = aie::zeros<int32, half_vec_factor>();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < num_vectors; i++) {
        const aie::vector<cint16, vec_factor> c_vec_i16 = aie::load_v<vec_factor>(pIn);
        pIn += vec_factor;

        // Input and delayed output remain complex cint16 Q0.15.
        aie::store_v(pOutDelay, delay_hist);
        pOutDelay += vec_factor;
        delay_hist = c_vec_i16;

        // The real and imaginary components are Q0.15. Squaring and adding
        // them produces Q0.30 in the wide accumulator. Shift by seven bits
        // when narrowing to int32 Q8.23 to retain value and boxcar headroom.
        const aie::vector<int16, vec_factor * 2> c_vec_i16_parts = aie::vector_cast<int16>(c_vec_i16);
        const aie::vector<int16, vec_factor> real_v = aie::filter_even(c_vec_i16_parts, 1);
        const aie::vector<int16, vec_factor> imag_v = aie::filter_odd(c_vec_i16_parts, 1);
        auto mag_sq_acc = aie::mul<acc48>(real_v, real_v);
        mag_sq_acc = aie::mac(mag_sq_acc, imag_v, imag_v);
        const aie::vector<int32, vec_factor> mag_sq = mag_sq_acc.to_vector<int32>(input_q * 2 - mag_q);

        // Sum the previous 64 Q8.23 magnitudes in a wide accumulator.
        auto base_acc = aie::add(aie::accum<acc48, vec_factor>(h0), h1);
        base_acc = aie::add(base_acc, h2);
        base_acc = aie::add(base_acc, h3);
        const auto base_vec = base_acc.to_vector<int32>();
        const int32 base = aie::reduce_add(base_vec);

        // Add each incoming Q8.23 magnitude while removing the oldest one.
        const auto delta_acc = aie::sub(aie::accum<acc48, vec_factor>(mag_sq), h0);
        const auto delta = delta_acc.to_vector<int32>();
        auto prefix_lo = delta.extract<half_vec_factor>(0);
        auto prefix_hi = delta.extract<half_vec_factor>(1);

        prefix_lo = aie::add(prefix_lo, aie::shuffle_up_fill(prefix_lo, zero, 1));
        prefix_lo = aie::add(prefix_lo, aie::shuffle_up_fill(prefix_lo, zero, 2));
        prefix_lo = aie::add(prefix_lo, aie::shuffle_up_fill(prefix_lo, zero, 4));
        prefix_hi = aie::add(prefix_hi, aie::shuffle_up_fill(prefix_hi, zero, 1));
        prefix_hi = aie::add(prefix_hi, aie::shuffle_up_fill(prefix_hi, zero, 2));
        prefix_hi = aie::add(prefix_hi, aie::shuffle_up_fill(prefix_hi, zero, 4));

        const auto out_lo = aie::add(prefix_lo, base);
        const auto out_hi = aie::add(prefix_hi, base + prefix_lo[half_vec_factor - 1]);

        // Keep the Q8.23 boxcar sum intact for threshold_cf.
        aie::vector<int32, vec_factor> out_v;
        out_v.insert(0, out_lo);
        out_v.insert(1, out_hi);
        aie::store_v(pOut, out_v);
        pOut += vec_factor;

        h0 = h1;
        h1 = h2;
        h2 = h3;
        h3 = mag_sq;
    }
}

} // extern "C"
