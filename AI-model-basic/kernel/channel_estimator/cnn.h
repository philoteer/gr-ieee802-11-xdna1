#ifndef FRAME_EQUALIZER_EQUALIZER_CNN_H
#define FRAME_EQUALIZER_EQUALIZER_CNN_H

#include <stdint.h>

#include "cnn_common.h"
#include "model/frame_equalizer_channel_ai_weights.h"

/**
 * @file cnn.h
 * @brief CNN-refined least-squares channel estimator.
 *
 * Selected through FRAME_EQUALIZER_ALGORITHM_CNN in select.h. The estimator
 * starts from the same two-symbol LTS least-squares estimate as ls.h, then adds
 * a learned per-bin residual predicted by a compact convolutional network:
 *
 *   input       5 x 64  first/second LTS I/Q (Q15) plus occupied-bin mask
 *   conv 5-tap  8 x 64  int8 weights (Q7), ReLU
 *   conv 3-tap  8 x 64  int8 weights (Q7), ReLU
 *   conv 3-tap  2 x 64  linear real/imaginary residual
 *
 * The residual is returned to the raw input scale and saturated before it is
 * added to the LS estimate, so the network only refines occupied subcarriers by
 * a bounded amount. Weights come from the generated
 * model/frame_equalizer_channel_ai_weights.h.
 */

namespace {

// Each row keeps an eight-column zero halo on both sides. The halo is wider
// than the 5-tap and 3-tap padding needs so the first data column starts on a
// 32-byte boundary; with FFT_SIZE + 16 element rows, every store in the
// estimator can use an aligned vector store. The halo itself is never
// written, so it stays zero for the convolution edge taps.
alignas(aie::vector_decl_align) static int32_t d_cnn_input[5][FFT_SIZE + 16];
alignas(aie::vector_decl_align) static int32_t d_cnn_hidden0[12][FFT_SIZE + 16];
alignas(aie::vector_decl_align) static int32_t d_cnn_hidden1[12][FFT_SIZE + 16];
alignas(aie::vector_decl_align) static int32_t d_cnn_residual[2][FFT_SIZE];

static_assert(__alignof__(d_cnn_input) >= aie::vector_ldst_align_v<int32_t, 16>,
              "d_cnn_input rows must support aligned 16-lane stores");
static_assert(__alignof__(d_cnn_hidden0) >= aie::vector_ldst_align_v<int32_t, 16>,
              "d_cnn_hidden0 rows must support aligned 16-lane stores");
static_assert(__alignof__(d_cnn_hidden1) >= aie::vector_ldst_align_v<int32_t, 16>,
              "d_cnn_hidden1 rows must support aligned 16-lane stores");
static_assert(__alignof__(d_cnn_residual) >= aie::vector_ldst_align_v<int32_t, 16>,
              "d_cnn_residual rows must support aligned 16-lane stores");
static_assert((8 * sizeof(int32_t)) % aie::vector_ldst_align_v<int32_t, 16> == 0,
              "CNN data offset must preserve aligned 16-lane stores");
static_assert(((FFT_SIZE + 16) * sizeof(int32_t)) %
                      aie::vector_ldst_align_v<int32_t, 16> == 0,
              "CNN row stride must preserve aligned 16-lane stores");

struct CnnChannelEstimator {
    static NOINLINE void forward(
            const cint32 second[FFT_SIZE], cint32 channel[FFT_SIZE]) {
        aie::scoped_mode rnd_mode{aie::rounding_mode::floor};
        aie::scoped_mode sat_mode{aie::saturation_mode::saturate};

        const int normalization_shift = CnnInputNormalize::forward(
            d_first_long, second, &d_cnn_input[0][8], FFT_SIZE + 16);
        CnnConv1dRelu<5, 12, 7>::forward(
            &d_cnn_input[0][8], FFT_SIZE + 16,
            &d_cnn_hidden0[0][8], FFT_SIZE + 16,
            frame_equalizer_channel_ai::kW1);
        CnnConv1dRelu<12, 12, 5>::forward(
            &d_cnn_hidden0[0][8], FFT_SIZE + 16,
            &d_cnn_hidden1[0][8], FFT_SIZE + 16,
            frame_equalizer_channel_ai::kW2);
        CnnConv1d<12, 2, 3>::forward(
            &d_cnn_hidden1[0][8], FFT_SIZE + 16,
            &d_cnn_residual[0][0], FFT_SIZE,
            frame_equalizer_channel_ai::kW3);
        CnnResidualAdd::forward(
            &d_cnn_residual[0][0], FFT_SIZE, normalization_shift, channel);
    }
};

/**
 * Updates the channel estimate for one received symbol.
 *
 * Always delegates to the shared LTS initialization. Once the second training
 * symbol has arrived (number == 1) the newly formed least-squares estimate is
 * refined by the CNN; later calls leave the fixed estimate untouched.
 *
 * @param symbol Frequency-domain symbol, FFT-shifted order.
 * @param number Symbol index within the frame, 0 for the first LTS.
 * @param channel In/out channel estimate, FFT-shifted order.
 * @param snr In/out linear signal-to-noise ratio in Q4.4.
 */
static inline void update_channel_estimate(
        const cint32 *symbol, int number, cint32 channel[FFT_SIZE],
        uint64_t &snr) {
    init_channel_estimate_from_lts(symbol, number, channel, snr);
    if (number == 1) CnnChannelEstimator::forward(symbol, channel);
}

/**
 * No-op: the CNN derives the estimate from the LTS pair only, so payload
 * decisions never feed back into the channel.
 */
static inline void track_channel_estimate(
        const cint32 *, const int8_t *, int, int, cint32 [FFT_SIZE]) {}

/**
 * No-op: the CNN keeps no per-frame state beyond the shared LTS capture.
 */
static inline void reset_channel_estimate(uint64_t &snr) { (void)snr; }

} // namespace

#endif
