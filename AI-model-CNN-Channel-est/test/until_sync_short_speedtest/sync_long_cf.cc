//===- sync_long_cf.cc ------------------------------------------*- C++ -*-===//
// Long preamble synchronizer placeholder; currently passes sync_short_cf outputs through unchanged.
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#define vec_factor 16
#define delay_samples 320

extern "C" {

/** @brief Speed-test placeholder for the long-preamble synchronizer.
 *
 *  Skips the 320-sample delay history carried by the sync_short_cf output,
 *  copies the remaining N samples unchanged, and forwards meta_N metadata
 *  words. It performs no correlation or CFO refinement.
 *
 *  @param _in sync_short_cf output data area (delay history plus samples).
 *  @param in_meta sync_short_cf metadata words to forward.
 *  @param out Pass-through complex samples, N cint16.
 *  @param out_meta Pass-through metadata, meta_N int32 words.
 *  @param N Number of complex samples to copy.
 *  @param meta_N Number of metadata words to copy.
 */
void sync_long_cf(cint16 *__restrict _in, int32_t *__restrict in_meta, cint16 *__restrict out, int32_t *__restrict out_meta, int32_t N, int32_t meta_N) {
    cint16 *__restrict in_current = _in + delay_samples;

    cint16 *__restrict pIn = in_current;
    cint16 *__restrict pOut = out;

    const int num_vectors = N / vec_factor;

    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < num_vectors; i++) {
        aie::vector<cint16, vec_factor> in_vec = aie::load_v<vec_factor>(pIn);
        pIn += vec_factor;
        aie::store_v(pOut, in_vec);
        pOut += vec_factor;
    }

    for (int i = num_vectors * vec_factor; i < N; i++) {
        out[i] = in_current[i];
    }

    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < meta_N; i++) {
        out_meta[i] = in_meta[i];
    }
}

}
