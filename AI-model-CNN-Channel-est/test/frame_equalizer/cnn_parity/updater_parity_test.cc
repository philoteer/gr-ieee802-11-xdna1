#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

#define FRAME_EQUALIZER_ALGORITHM FRAME_EQUALIZER_ALGORITHM_CNN
#define FRAME_EQUALIZER_CNN_DATA_UPDATER FRAME_EQUALIZER_CNN_DATA_UPDATER_MLP
#include "channel_estimator/select.h"

extern "C" void updater_parity_test(const int32_t *__restrict packed_symbols,
                                     int32_t *__restrict packed_channel) {
    const auto *symbols = reinterpret_cast<const cint32 *>(packed_symbols);
    alignas(aie::vector_decl_align) static cint32 channel[FFT_SIZE];
    uint64_t snr = 0;
    update_channel_estimate(symbols, 0, channel, snr);
    update_channel_estimate(symbols + FFT_SIZE, 1, channel, snr);
    for (int number = 2; number <= 6; ++number) {
        update_channel_estimate(symbols + number * FFT_SIZE, number, channel, snr);
        if (number == 2 || number == 6) {
            auto *destination = reinterpret_cast<cint32 *>(packed_channel) +
                (number == 2 ? 0 : FFT_SIZE);
            for (int bin = 0; bin < FFT_SIZE; bin += 8)
                aie::store_v(destination + bin, aie::load_v<8>(channel + bin));
        }
    }
}
