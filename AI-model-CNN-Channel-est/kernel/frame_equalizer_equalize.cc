#include <stdint.h>
#include <stddef.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#include "frame_equalizer_metadata.h"
#include "demodulator/frame_equalizer_engine.h"

namespace {
static bool d_pending_tag;
alignas(aie::vector_decl_align) static cint32 d_pending_channel[FFT_SIZE];
static uint64_t d_pending_snr_linear;

// The selected engine (demodulator/frame_equalizer_engine.h) provides equalize_symbol(),
// and the selected estimator (channel_estimator/select.h) provides
// update_channel_estimate() for the per-symbol update and reset_channel_estimate()
// for the frame reset below.

/** @brief Resets frame state at a tag boundary.
 *
 *  Restores the frame-relative symbol counter, rate fields, and pending-tag
 *  flag, and gives the selected estimator its frame reset.
 *
 *  @param cfo_q29 Frame CFO in Q3.29 radians/sample.
 */
static inline void reset_equalization(int32_t cfo_q29) {
    d_equalizer_current_symbol = 0; d_frame_symbols = 0; d_frame_bytes = 0;
    d_encoding = 0; d_modulation = 2; d_equalizer_cfo_q29 = cfo_q29;
    d_pending_tag = false;
}

} // namespace

/** @brief Equalizes one prepared tile and emits payload decisions and frame
 *  metadata tags.
 *
 *  The copied FFT metadata drives this kernel's frame resets; the SIGNAL
 *  symbol at frame index 2 selects the payload modulation. In production mode
 *  one tag carrying a 64-bin CSI snapshot is emitted per frame.
 *
 *  @param prepared Phase-corrected FFT bins from frame_equalizer_preprocess.
 *  @param prepared_meta Prepared metadata (FFT metadata, beta, center frequency).
 *  @param out One decision byte per data carrier per payload symbol.
 *  @param out_meta frame_equalizer_tile_metadata with length, tags, and CSI.
 *  @param N Unused; the valid sample count comes from the copied FFT metadata.
 *  @param meta_N Unused.
 */
extern "C" void frame_equalizer_equalize(
    const cint32 *__restrict prepared, const int32_t *__restrict meta_in,
    int8_t *__restrict out, int32_t *__restrict out_meta,
    int32_t N, int32_t meta_N) {
    const auto *estimated_metadata =
        reinterpret_cast<const frame_equalizer_estimated_tile_metadata *>(meta_in);
    const auto *prepared_metadata = &estimated_metadata->prepared;
    // The copied FFT metadata drives this kernel's own frame counter and tag
    // resets; no persistent state is shared with preprocess.
    const int32_t *in_meta = prepared_metadata->input;
    d_center_frequency_mhz = prepared_metadata->center_frequency_mhz;
    auto *metadata = reinterpret_cast<frame_equalizer_tile_metadata *>(out_meta);
    metadata->output_length = 0; 
    metadata->tag_count = 0;
    int produced = 0;
    //const int input_count = in_meta[0] < N ? in_meta[0] : N; //we are short on memory, 'precautions' has to go.
    const int input_count = in_meta[0];
    const int input_tags = in_meta[1];
    int tag_index = 0;
    // Frame-relative symbols are 0/1=LTS, 2=SIGNAL, and 3 onward=payload.
    for (int offset = 0; offset + 64 <= input_count; offset += 64) {
        while (tag_index < input_tags && in_meta[2 + 2 * tag_index] <= offset) {
            reset_equalization(in_meta[3 + 2 * tag_index]);
            ++tag_index;
        }
        if (d_equalizer_current_symbol < 0 ||
            d_equalizer_current_symbol > d_frame_symbols + 2) continue;

        const cint32 *prepared_symbol = prepared + offset;
        const cint32 *channel = reinterpret_cast<const cint32 *>(
            estimated_metadata->channel[offset / FFT_SIZE]);
        const uint64_t snr_linear =
            estimated_metadata->snr_linear[offset / FFT_SIZE];
        // Beta belongs to the physical symbol slot in this 1024-sample tile,
        // while d_equalizer_current_symbol remains frame-relative state.
        const int32_t beta = prepared_metadata->beta[offset / 64];

        // SIGNAL is always BPSK. Later symbols use the modulation selected
        // by SIGNAL; pilots, guards, and DC never enter the decisions.
        const int current_symbol = d_equalizer_current_symbol;
        const int modulation = current_symbol == 2 ? 2 : d_modulation;
#if !FRAME_EQUALIZER_OUTPUT_SYMBOLS
        // GNU Radio publishes frame CSI immediately after SIGNAL equalization.
        // Emit the delayed stream tag before the estimator incorporates the
        // first payload symbol, avoiding a separate 64-bin channel snapshot.
        if (current_symbol > 2 && d_pending_tag &&
            metadata->tag_count < FRAME_EQUALIZER_MAX_TAGS) {
            fill_tag(metadata->tags[metadata->tag_count++], produced,
                     d_pending_channel, d_pending_snr_linear);
            d_pending_tag = false;
        }
#endif 
        //for original code:
        //equalize_symbol(...)	13,187 cycles/tile share:	67.6%
        equalize_symbol(
            prepared_symbol, channel, current_symbol, modulation, d_decisions);
        if (current_symbol == 2) {
            d_signal_beta_q29 = beta;
            decode_signal(d_decisions, d_decoded);
            d_pending_tag = parse_signal(d_decoded);
#if !FRAME_EQUALIZER_OUTPUT_SYMBOLS
            if (d_pending_tag) {
                for (int bin = 0; bin < FFT_SIZE; bin += 8) {
                    aie::store_v(d_pending_channel + bin,
                                 aie::load_v<8>(channel + bin));
                }
                d_pending_snr_linear = snr_linear;
            }
#endif
        } else if (current_symbol > 2) {
            // Delay frame metadata until the first payload decision so its
            // offset refers to an actual output item rather than SIGNAL.
#if FRAME_EQUALIZER_OUTPUT_SYMBOLS
            if (metadata->tag_count < FRAME_EQUALIZER_MAX_TAGS) {
                fill_tag(metadata->tags[metadata->tag_count++], produced,
                         channel, snr_linear);
                d_pending_tag = false;
            }
#endif
            // produced advances by 48 bytes and the runtime buffer chunks
            // are 1 KiB-aligned, so every store address is 16-byte aligned;
            // the aligned store avoids the unaligned read-modify-write.
            aie::store_unaligned_v<16>(out + produced, aie::load_v<16>(d_decisions));
            aie::store_unaligned_v<16>(out + produced + 16, aie::load_v<16>(d_decisions + 16));
            aie::store_unaligned_v<16>(out + produced + 32, aie::load_v<16>(d_decisions + 32));
            produced += 48;
        }
        ++d_equalizer_current_symbol;
    }
    metadata->output_length = produced;
}
