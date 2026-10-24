#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#include "frame_equalizer_metadata.h"
#include "channel_estimator/select.h"

#if FRAME_EQUALIZER_ALGORITHM != FRAME_EQUALIZER_ALGORITHM_LS && \
    FRAME_EQUALIZER_ALGORITHM != FRAME_EQUALIZER_ALGORITHM_COMB && \
    FRAME_EQUALIZER_ALGORITHM != FRAME_EQUALIZER_ALGORITHM_CNN
#error "The split channel estimator supports LS, COMB, and CNN"
#endif

namespace {
alignas(aie::vector_decl_align) static cint32 d_channel[FFT_SIZE];
static uint64_t d_snr_linear;
static int d_current_symbol = -1;

static inline void reset_estimation() {
    d_current_symbol = 0;
    reset_channel_estimate(d_snr_linear);
}
} // namespace

extern "C" void frame_equalizer_channel_estimate(
    const cint32 *__restrict prepared, const int32_t *__restrict prepared_meta,
    int32_t *__restrict meta_out) {
    const auto *prepared_metadata =
        reinterpret_cast<const frame_equalizer_prepared_tile_metadata *>(prepared_meta);
    auto *metadata =
        reinterpret_cast<frame_equalizer_estimated_tile_metadata *>(meta_out);
    const int32_t *in_meta = prepared_metadata->input;
    const int input_count = in_meta[0];
    const int input_tags = in_meta[1];
    int tag_index = 0;

    for (int offset = 0; offset + FFT_SIZE <= input_count; offset += FFT_SIZE) {
        while (tag_index < input_tags && in_meta[2 + 2 * tag_index] <= offset) {
            reset_estimation();
            ++tag_index;
        }
        if (d_current_symbol < 0) continue;

        update_channel_estimate(
            prepared + offset, d_current_symbol, d_channel, d_snr_linear);
        auto *destination = reinterpret_cast<cint32 *>(
            metadata->channel[offset / FFT_SIZE]);
        for (int bin = 0; bin < FFT_SIZE; bin += 8) {
            aie::store_v(destination + bin, aie::load_v<8>(d_channel + bin));
        }
        metadata->snr_linear[offset / FFT_SIZE] = d_snr_linear;
        ++d_current_symbol;
    }

    auto *prepared_out = reinterpret_cast<int32_t *>(&metadata->prepared);
    for (int word = 0; word < FRAME_EQUALIZER_PREPARED_METADATA_WORDS; ++word)
        prepared_out[word] = prepared_meta[word];

}
