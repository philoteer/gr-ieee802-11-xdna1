#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

constexpr unsigned vec_factor = 16;
constexpr unsigned half_vec_factor = vec_factor / 2; // 8

// --- GLOBAL / STATIC STATE ---
// These retain their values across different function calls.
// delay_hist: Previous 16 input samples, retained as cint16 Q0.15.
static aie::vector<cint16, vec_factor> delay_hist = aie::zeros<cint16, vec_factor>();

// h0, h1, h2: Previous 3 blocks (48 samples) of correlation results in cint32 Q7.24.
// h0 is the oldest, h2 is the newest.
static aie::vector<cint32, vec_factor> h0 = aie::zeros<cint32, vec_factor>();
static aie::vector<cint32, vec_factor> h1 = aie::zeros<cint32, vec_factor>();
static aie::vector<cint32, vec_factor> h2 = aie::zeros<cint32, vec_factor>();

extern "C" {

// N must be a multiple of 16.
void autocorr_cf(cint16 *__restrict in, cint32 *__restrict out, int32_t N) {
    // All values after the correlation are cint32 Q7.24.  The seven integer
    // bits provide a range of approximately +/-128 for the 48-term sum.
    const auto zero = aie::zeros<cint32, half_vec_factor>();

    // Compiler pragmas to ensure the loop is hardware-pipelined for maximum efficiency.
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    
    // Process the data in chunks of 16 samples.
    for (int i = 0; i < N / vec_factor; ++i) {
        
        // ---------------------------------------------------------
        // 1. DATA LOADING & AUTOCORRELATION
        // ---------------------------------------------------------
        // Input and delay_hist are complex Q0.15.  Their product is Q0.30
        // in the wide accumulator.  Shift it by six bits when narrowing to
        // cint32 Q7.24; this preserves the value while leaving headroom for
        // the 48-sample sum below.
        const aie::vector<cint16, vec_factor> input = aie::load_v<vec_factor>(in);
        in += vec_factor;

        // Perform a lag-16 autocorrelation:
        // corr[k] = input[k] * conjugate(input[k-16]).
        const aie::accum<cacc48, vec_factor> corr_product = aie::mul(input, aie::conj(delay_hist));
        const aie::vector<cint32, vec_factor> corr = corr_product.to_vector<cint32>(6);
        delay_hist = input; // Save current input for the next loop iteration.

        // ---------------------------------------------------------
        // 2. BOXCAR BASE CALCULATION (The sum of the last 48 points)
        // ---------------------------------------------------------
        // Sum all 48 elements in the history buffers (h0 + h1 + h2).
        // The accumulator prevents overflow during the vector additions.
        auto base_acc = aie::add(aie::accum<cacc48, vec_factor>(h0), h1);
        base_acc = aie::add(base_acc, h2);
        
        // reduce_add squashes the 16-element vector down into a single scalar value.
        // 'base' is now the total sum of the previous 48 Q7.24 samples.
        const auto base_vec = base_acc.to_vector<cint32>();
        const cint32 base = aie::reduce_add(base_vec);

        // ---------------------------------------------------------
        // 3. THE DELTA CALCULATION
        // ---------------------------------------------------------
        // For every new sample entering the 48-point window from 'corr', 
        // an old sample from 'h0' exits the window. 
        // delta_acc contains the difference: (New Sample - Oldest Sample)
        const auto delta_acc = aie::sub(aie::accum<cacc48, vec_factor>(corr), h0);
        
        // Split the 16-element delta vector into two 8-element halves to do the math.
        // The prefix sums remain in Q7.24; their worst-case magnitude is
        // below 64, so cint32 remains safe throughout this stage.
        auto prefix_lo = delta_acc.extract<half_vec_factor>(0).to_vector<cint32>();
        auto prefix_hi = delta_acc.extract<half_vec_factor>(1).to_vector<cint32>();

        // ---------------------------------------------------------
        // 4. PARALLEL PREFIX SUM (Kogge-Stone Adder)
        // ---------------------------------------------------------
        // This calculates the running sum of the deltas in just 3 steps.
        // shuffle_up_fill(vec, zero, X) shifts the vector right by X, filling the left with zeros.
        
        // Step 1: Add immediate neighbor (Shift by 1)
        prefix_lo = aie::add(prefix_lo, aie::shuffle_up_fill(prefix_lo, zero, 1));
        // Step 2: Add neighbor 2 spaces away (Shift by 2)
        prefix_lo = aie::add(prefix_lo, aie::shuffle_up_fill(prefix_lo, zero, 2));
        // Step 3: Add neighbor 4 spaces away (Shift by 4)
        prefix_lo = aie::add(prefix_lo, aie::shuffle_up_fill(prefix_lo, zero, 4));
        
        // Repeat the exact same parallel prefix sum for the upper half.
        prefix_hi = aie::add(prefix_hi, aie::shuffle_up_fill(prefix_hi, zero, 1));
        prefix_hi = aie::add(prefix_hi, aie::shuffle_up_fill(prefix_hi, zero, 2));
        prefix_hi = aie::add(prefix_hi, aie::shuffle_up_fill(prefix_hi, zero, 4));

        // ---------------------------------------------------------
        // 5. RECOMBINATION & OUTPUT
        // ---------------------------------------------------------
        // Add the running sum of the deltas back to the 'base' sum.
        const auto out_lo = aie::add(prefix_lo, base);
        
        // For the upper half, the base must also include the total accumulated deltas 
        // from the lower half, which is the last element: prefix_lo[7].
        const auto out_hi = aie::add(prefix_hi, base + prefix_lo[7]);
        
        // Stitch the two halves together and retain Q7.24 for downstream kernels.
        aie::vector<cint32, vec_factor> result;
        result.insert(0, out_lo);
        result.insert(1, out_hi);

        // Store the results back to the output pointer and advance the pointer.
        aie::store_v(out, result);
        out += vec_factor;

        // ---------------------------------------------------------
        // 6. SHIFT HISTORY FOR THE NEXT ITERATION
        // ---------------------------------------------------------
        h0 = h1;   // The middle block becomes the oldest
        h1 = h2;   // The newest history block becomes the middle
        h2 = corr; // The block we just processed becomes the newest history
    }
}

} // extern "C"
