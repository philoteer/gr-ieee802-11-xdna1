#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>
#include "fft64_twiddles.h"

#define FFT_SIZE 64
#define VECTOR_SIZE 8
#define COPY_VECTOR_SIZE 16
#define TWIDDLE_SHIFT 15
#define STAGE_SHIFT 15
#define MAX_PENDING_TAGS 31

namespace {
// sync_long may end a tile partway through an OFDM symbol. Keep that tail until
// enough samples arrive to run a complete FFT.
alignas(aie::vector_decl_align) static cint16 pending_samples[FFT_SIZE];
static int pending_sample_count = 0;

// Tags in the incomplete tail cannot be emitted until their samples are output.
static int pending_tag_offsets[MAX_PENDING_TAGS];
static int32_t pending_tag_values[MAX_PENDING_TAGS];
static int pending_tag_count = 0;

alignas(aie::vector_decl_align) static cint32 scratch[FFT_SIZE];
alignas(aie::vector_decl_align) static cint32 bins[FFT_SIZE];

/** @brief Copies count cint16 samples with 16-wide unaligned vector moves and
 *  a scalar tail. */
static inline void copy_samples(cint16 *__restrict dst,
                                const cint16 *__restrict src, int count) {
    int i = 0;
    for (; i + COPY_VECTOR_SIZE <= count; i += COPY_VECTOR_SIZE) {
        aie::store_unaligned_v(dst + i,
            aie::load_unaligned_v<COPY_VECTOR_SIZE>(src + i));
    }
    for (; i < count; i++) {
        dst[i] = src[i];
    }
}

/** @brief 64-point radix-4 DIT FFT in GNU Radio's shifted bin order.
 *
 *  Runs three radix-4 stages with 16, 4, and 1 butterfly units, Q15 twiddles,
 *  and Q15 per-stage shifts, then swaps the two 32-bin halves so negative
 *  frequencies precede non-negative ones.
 *
 *  @param input 64 time-domain samples, cint16 Q0.15.
 *  @param output 64 frequency bins in shifted order, cint32.
 */
static inline void fft64(const cint16 *__restrict input,
                         cint32 *__restrict output) {
    aie::fft_dit_r4_stage<16>(input, kStage0Tw2, kStage0Tw1, kStage0Tw3,
                              FFT_SIZE, TWIDDLE_SHIFT, STAGE_SHIFT, false, bins);
    aie::fft_dit_r4_stage<4>(bins, kStage1Tw2, kStage1Tw1, kStage1Tw3,
                             FFT_SIZE, TWIDDLE_SHIFT, STAGE_SHIFT, false, scratch);
    aie::fft_dit_r4_stage<1>(scratch, kStage2Tw2, kStage2Tw1, kStage2Tw3,
                             FFT_SIZE, TWIDDLE_SHIFT, STAGE_SHIFT, false, bins);

    // Match GNU Radio's shifted FFT: negative frequencies precede non-negative.
    for (int i = 0; i < FFT_SIZE / 2; i += VECTOR_SIZE) {
        aie::store_unaligned_v(output + i,
            aie::load_v<VECTOR_SIZE>(bins + i + FFT_SIZE / 2));
        aie::store_unaligned_v(output + i + FFT_SIZE / 2,
            aie::load_v<VECTOR_SIZE>(bins + i));
    }
}

} // namespace

extern "C" {

/** @brief 64-point FFT kernel with stream tags preserved across tile
 *  boundaries.
 *
 *  Buffers a partial OFDM symbol in static state and carries its tags until
 *  the samples are emitted. Input metadata is [valid samples, tag count, then
 *  (offset, value) pairs]; output metadata is [produced samples, tag count,
 *  then (offset, value) pairs]. Tag offsets are rounded up to the next
 *  64-sample FFT boundary to match GNU Radio's vector port behavior.
 *
 *  @param in Time-domain input samples, cint16 Q0.15.
 *  @param in_meta Input metadata (sample count, tag count, then tag pairs).
 *  @param out FFT bins in shifted order, cint32.
 *  @param out_meta Output metadata (produced count, tag count, then tag pairs).
 *  @param N Tile input length; unused, the count comes from the metadata.
 *  @param meta_N Metadata buffer size; unused.
 */
void fft_cf(cint16 *__restrict in, int32_t *__restrict in_meta,
            cint32 *__restrict out, int32_t *__restrict out_meta,
            int32_t N, int32_t meta_N) {
    aie::set_rounding(aie::rounding_mode::positive_inf);
    aie::set_saturation(aie::saturation_mode::saturate);

    const int input_count = in_meta[0];
    const int produced = (pending_sample_count + input_count) & ~(FFT_SIZE - 1); //produced = floor(samples/fftlen)*fftlen
    int output_tag_count = 0;
    int retained_tag_count = 0;

    // First resolve tags retained with the previous incomplete FFT frame.
    for (int i = 0; i < pending_tag_count; i++) {
        if (pending_tag_offsets[i] < produced) {
                out_meta[2 + 2 * output_tag_count] = pending_tag_offsets[i];
                out_meta[3 + 2 * output_tag_count] = pending_tag_values[i];
                output_tag_count++;
        } else {
            pending_tag_offsets[retained_tag_count] =
                pending_tag_offsets[i] - produced;
            pending_tag_values[retained_tag_count] = pending_tag_values[i];
            retained_tag_count++;
        }
    }

    // Current input tags are translated past the samples buffered from the
    // preceding call. Tags beyond `produced` remain attached to the saved tail.
    const int input_tag_count = in_meta[1];
    for (int i = 0; i < input_tag_count; i++) {
        const int scalar_offset = pending_sample_count + in_meta[2 + 2 * i];
        // GNU Radio's stream-to-vector/vector-to-stream path moves a scalar tag
        // to the next 64-sample FFT boundary.
        const int output_offset =
            (scalar_offset + FFT_SIZE - 1) & ~(FFT_SIZE - 1);
        const int32_t value = in_meta[3 + 2 * i];
        if (output_offset < produced) {
                out_meta[2 + 2 * output_tag_count] = output_offset;
                out_meta[3 + 2 * output_tag_count] = value;
                output_tag_count++;
        } else if (retained_tag_count < MAX_PENDING_TAGS) {
            pending_tag_offsets[retained_tag_count] = output_offset - produced;
            pending_tag_values[retained_tag_count] = value;
            retained_tag_count++;
        }
    }
    pending_tag_count = retained_tag_count;

    int input_offset = 0;
    int output_offset = 0;
    if (pending_sample_count != 0) {
        const int needed = FFT_SIZE - pending_sample_count;
        if (input_count >= needed) {
            copy_samples(pending_samples + pending_sample_count, in, needed);
            fft64(pending_samples, out);
            input_offset = needed;
            output_offset = FFT_SIZE;
            pending_sample_count = 0;
        } else {
            copy_samples(pending_samples + pending_sample_count, in, input_count);
            pending_sample_count += input_count;
            out_meta[0] = 0;
            out_meta[1] = output_tag_count;
            return;
        }
    }

    while (input_offset + FFT_SIZE <= input_count) {
        if ((input_offset & (COPY_VECTOR_SIZE - 1)) == 0) {
            for (int i = 0; i < FFT_SIZE; i += COPY_VECTOR_SIZE) {
                aie::store_v(pending_samples + i,
                    aie::load_v<COPY_VECTOR_SIZE>(in + input_offset + i));
            }
        } else {
            for (int i = 0; i < FFT_SIZE; i += COPY_VECTOR_SIZE) {
                aie::store_v(pending_samples + i,
                    aie::load_unaligned_v<COPY_VECTOR_SIZE>(
                        in + input_offset + i));
            }
        }
        fft64(pending_samples, out + output_offset);
        input_offset += FFT_SIZE;
        output_offset += FFT_SIZE;
    }

    const int remaining = input_count - input_offset;
    copy_samples(pending_samples, in + input_offset, remaining);
    pending_sample_count = remaining;

    out_meta[0] = produced;
    out_meta[1] = output_tag_count;
}

}
