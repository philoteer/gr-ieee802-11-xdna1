#include <stdint.h>
#include <stddef.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#include "frame_equalizer_metadata.h"
#include "channel_estimator/select.h"
#include "demodulator/frame_equalizer_engine.h"

namespace {
// The selected engine (demodulator/frame_equalizer_engine.h) provides equalize_symbol(),
// and the selected estimator (channel_estimator/select.h) provides
// reset_channel_estimate() for the frame reset below.

static inline void reset_equalization(int32_t cfo_q29) {
    d_equalizer_current_symbol = 0; d_frame_symbols = 0; d_frame_bytes = 0;
    d_encoding = 0; d_modulation = 2; d_equalizer_cfo_q29 = cfo_q29;
    d_pending_tag = false;
    reset_channel_estimate(d_snr_linear);
}
} // namespace

extern "C" void frame_equalizer_equalize(
    const cint32 *__restrict prepared, const int32_t *__restrict prepared_meta,
    int8_t *__restrict out, int32_t *__restrict out_meta,
    int32_t N, int32_t meta_N) {
    const auto *prepared_metadata =
        reinterpret_cast<const frame_equalizer_prepared_tile_metadata *>(prepared_meta);
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
            fill_tag(metadata->tags[metadata->tag_count++], produced);
            d_pending_tag = false;
        }
#endif 
        //for original code:
        //equalize_symbol(...)	13,187 cycles/tile share:	67.6%
        equalize_symbol(prepared_symbol, current_symbol, modulation, d_decisions);
        if (current_symbol == 2) {
            d_signal_beta_q29 = beta;
            decode_signal(d_decisions, d_decoded);
            d_pending_tag = parse_signal(d_decoded);
        } else if (current_symbol > 2) {
            // Delay frame metadata until the first payload decision so its
            // offset refers to an actual output item rather than SIGNAL.
#if FRAME_EQUALIZER_OUTPUT_SYMBOLS
            if (metadata->tag_count < FRAME_EQUALIZER_MAX_TAGS) {
                fill_tag(metadata->tags[metadata->tag_count++], produced);
                d_pending_tag = false;
            }
#endif
            // produced advances by 48 bytes and the runtime buffer chunks
            // are 1 KiB-aligned, so every store address is 16-byte aligned;
            // the aligned store avoids the unaligned read-modify-write.
            aie::store_v(out + produced, aie::load_v<16>(d_decisions));
            aie::store_v(out + produced + 16, aie::load_v<16>(d_decisions + 16));
            aie::store_v(out + produced + 32, aie::load_v<16>(d_decisions + 32));
            produced += 48;
        }
        ++d_equalizer_current_symbol;
    }
    metadata->output_length = produced;
}
