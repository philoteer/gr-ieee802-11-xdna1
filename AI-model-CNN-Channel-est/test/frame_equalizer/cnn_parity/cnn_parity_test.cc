#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

#define FRAME_EQUALIZER_ALGORITHM FRAME_EQUALIZER_ALGORITHM_CNN
#include "channel_estimator/select.h"

/** @brief Runs the CNN-refined LTS channel estimator on two packed symbols.
 *
 *  Calls update_channel_estimate() for the first and second long-training
 *  symbols, leaving the resulting 64-bin channel estimate in packed_channel.
 *  The SNR accumulated during initialization stays local because the parity
 *  check compares channel bins only.
 *
 *  @param packed_lts Two FFT-shifted 64-bin LTS symbols, cint32.
 *  @param packed_channel Output 64-bin channel estimate, cint32.
 */
extern "C" void cnn_parity_test(const int32_t *__restrict packed_lts,
                                int32_t *__restrict packed_channel) {
    const auto *lts = reinterpret_cast<const cint32 *>(packed_lts);
    auto *channel = reinterpret_cast<cint32 *>(packed_channel);
    uint64_t snr = 0;

    update_channel_estimate(lts, 0, channel, snr);
    update_channel_estimate(lts + FFT_SIZE, 1, channel, snr);
}
