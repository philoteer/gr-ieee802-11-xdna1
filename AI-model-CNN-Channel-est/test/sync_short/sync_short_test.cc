// Test-only packing adapter for the standalone sync_short_cf graph.
#include <stdint.h>

#include "../../kernel/sync_short_cf.cc"

#define delay_samples 320

extern "C" {

/** @brief Unpacks one test-graph tile and runs the production short-preamble
 *  synchronizer.
 *
 *  @param packed_in Packed cint16[N], cint32[N], and int8[N] input regions.
 *  @param packed_out Output data area followed by the kernel metadata.
 *  @param N Number of input samples in this tile.
 */
void sync_short_test(int32_t *__restrict packed_in,
                     int32_t *__restrict packed_out,
                     int32_t N)
{
    // Input layout, in int32 words: cint16[N], cint32[N], int8[N].
    cint16 *in = reinterpret_cast<cint16 *>(packed_in);
    cint32 *abs_in = reinterpret_cast<cint32 *>(packed_in + N);
    int8_t *threshold_met = reinterpret_cast<int8_t *>(packed_in + 3 * N);

    // A cint16 occupies one int32 word. Metadata follows the enlarged output.
    cint16 *out = reinterpret_cast<cint16 *>(packed_out);
    int32_t *out_meta = packed_out + N + delay_samples;
    sync_short_cf(in, abs_in, threshold_met, out, out_meta, N);
}

}
