#ifndef FRAME_EQUALIZER_EQUALIZER_COMMON_H
#define FRAME_EQUALIZER_EQUALIZER_COMMON_H

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

// Shared configuration and PHY tables for the statically selected
// frame-equalizer algorithms. This mirrors the data that the original GNU Radio
// base equalizer (stage34/original/equalizer/base.h) made available to every
// algorithm.
#define FFT_SIZE 64
#define NOINLINE __attribute__((noinline))

// All bin-indexed tables use FFT-shifted order: DC is bin 32, negative
// subcarriers are on its left, and positive subcarriers are on its right.
//
// kLong is the frequency-domain BPSK pattern of the long training symbol
// (base::LONG), used by the LTS-based estimators (LS, LMS, STA, and the CNN) to
// remove the transmitted sign from the two repeated training symbols.
alignas(aie::vector_decl_align) const int16_t kLong[FFT_SIZE] = {
    0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
    1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
    0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
};
// The 52 occupied bins comprise 48 data carriers and four pilots. Guards and
// DC are zero, so the shared LTS SNR loop, STA's spatial smoothing, and the
// CNN input/residual masking all skip them.
alignas(aie::vector_decl_align) const int8_t kOccupied[FFT_SIZE] = {
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
};
// Standard 127-symbol pilot-polarity sequence for data OFDM symbols
// (base::POLARITY). Index zero belongs to SIGNAL. Needed wherever the known
// pilot signs are removed: comb, STA, and the preprocess kernel's beta
// estimate.
const int8_t kPolarity[127] = {
    1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1,
    -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
    -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
    -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1,
    -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1,
    -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
};

// Frame-persistent first long training symbol. It is held aside until the
// second LTS arrives so signal and noise power can be measured from their sum
// and difference before the combined channel estimate replaces it. LS, LMS,
// STA, and the CNN all perform the identical LTS capture (number == 0).
alignas(aie::vector_decl_align) static cint32 d_first_long[FFT_SIZE];

/** @brief Shared LTS capture, two-symbol channel estimate, and scaled SNR ratio.
 *
 *  Symbols 0 and 1 are the repeated LTS; every LTS-based estimator (ls.h,
 *  lms.h, sta.h, and cnn.h) calls this exact capture, sum/difference power
 *  measurement, and SNR-ratio calculation. The stored ratio is multiplied by
 *  16 to retain four fractional bits in a uint64_t; the value is not limited
 *  to an 8-bit fixed-point range. Later symbols are a no-op because the
 *  estimate has already been formed.
 *
 *  @param symbol Frequency-domain symbol in FFT-shifted order.
 *  @param number Frame-relative symbol index; 0 saves the first LTS, 1 forms
 *         the estimate from the pair.
 *  @param channel In/out 64-bin channel estimate, FFT-shifted order.
 *  @param snr In/out sum/difference power ratio scaled by 16.
 */
static NOINLINE void init_channel_estimate_from_lts(const cint32 *symbol,
                                                    int number,
                                                    cint32 channel[FFT_SIZE],
                                                    uint64_t &snr) {
    if (number == 0) {
        for (int bin = 0; bin < FFT_SIZE; bin += 16)
            aie::store_v(d_first_long + bin, aie::load_v<16>(symbol + bin));
        return;
    }
    if (number != 1) return;

    uint64_t signal_power = 0, noise_power = 0;
    aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_floor};
    for (int bin = 0; bin < FFT_SIZE; bin += 8) {
        const auto first_v = aie::load_v<8>(d_first_long + bin);
        const auto symbol_v = aie::load_v<8>(symbol + bin);
        const auto signal_v = aie::add(first_v, symbol_v);
        const auto noise_v = aie::sub(first_v, symbol_v);
        const auto long_v = aie::load_v<8>(kLong + bin);

        aie::store_v(
            channel + bin,
            aie::mul<cacc64>(signal_v, long_v).to_vector<cint32>(1));

        for (int lane = 0; lane < 8; ++lane) {
            if (!kOccupied[bin + lane]) continue;
            const cint32 signal_sample = signal_v[lane];
            const cint32 noise_sample = noise_v[lane];
            signal_power += (int64_t)signal_sample.real * signal_sample.real +
                            (int64_t)signal_sample.imag * signal_sample.imag;
            noise_power += (int64_t)noise_sample.real * noise_sample.real +
                           (int64_t)noise_sample.imag * noise_sample.imag;
        }
    }
    snr = (signal_power << 4) / (noise_power + 1);
}

#endif
