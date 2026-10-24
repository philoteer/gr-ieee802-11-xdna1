#ifndef FRAME_EQUALIZER_CHANNEL_ESTIMATOR_CNN_COMMON_H
#define FRAME_EQUALIZER_CHANNEL_ESTIMATOR_CNN_COMMON_H

#include <stdint.h>

#include "common.h"

/**
 * @file cnn_common.h
 * @brief Architecture-independent layers for the CNN channel estimator.
 *
 * cnn.h composes the estimator from these primitives. Every layer operates on
 * the 64-bin FFT axis, one output per bin, and shares the fixed-point contract
 * of the exported network: Q7 int8 weights, Q15 int32 activations, 64-bit
 * convolution accumulators, arithmetic shifts, and int32 saturation.
 */

namespace {

/**
 * Builds the CNN input tensor from the two long training symbols.
 *
 * Rows 0..3 hold the Q15-normalized, kLong-corrected real and imaginary parts
 * of the first and second LTS; row 4 holds the occupied-subcarrier mask. Each
 * row keeps an eight-column zero halo on both sides so the following 5-tap
 * convolution can load full 16-lane windows at every bin, and the caller
 * passes the first data column with a padded row stride so every store is
 * vector-aligned.
 */
struct CnnInputNormalize {
    /**
     * Normalizes both LTS observations into the network's five input channels.
     *
     * The largest absolute sign-corrected component over the occupied bins of
     * both symbols determines how far the inputs must be shifted to fill Q15
     * without clipping. All four I/Q channels share that scale so their
     * relative magnitudes are preserved.
     *
     * @param first First long training symbol, FFT-shifted order.
     * @param second Second long training symbol, FFT-shifted order.
     * @param output Five-row input buffer starting at the first data column;
     *        row c receives channel c and must be vector-aligned.
     * @param output_stride Elements between consecutive channel rows.
     * @return Signed normalization shift applied to the raw components.
     */
    static inline int forward(
            const cint32 first[FFT_SIZE], const cint32 second[FFT_SIZE],
            int32_t *output, unsigned output_stride) {
        const unsigned magnitude_bits = magnitude_bits_of(
            first, second);
        const int shift = 15 - int(magnitude_bits);

        const auto zero = aie::zeros<int32_t, 16>();
        const auto occupied_value = aie::broadcast<int32_t, 16>(int32_t(32767));
        const unsigned down_shift = shift < 0 ? unsigned(-shift) : 0u;
        const unsigned up_shift = shift > 0 ? unsigned(shift) : 0u;

        for (int bin = 0; bin < FFT_SIZE; bin += 16) {
            // kLong is +1 or -1 on occupied bins and 0 on guards and DC, so
            // folding the normalization shift into it keeps the four channel
            // products exact in acc64 and leaves a saturating narrowing to the
            // int16 activation range.
            auto sign = aie::unpack(aie::load_v<16>(kLong + bin));
            if (up_shift) sign = aie::upshift(sign, up_shift);
            const auto first_parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(first + bin));
            const auto second_parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(second + bin));

            aie::store_v(
                output + 0 * output_stride + bin,
                aie::unpack(aie::mul<acc64>(
                    aie::filter_even(first_parts, 1), sign)
                    .to_vector<int16_t>(down_shift)));
            aie::store_v(
                output + 1 * output_stride + bin,
                aie::unpack(aie::mul<acc64>(
                    aie::filter_odd(first_parts, 1), sign)
                    .to_vector<int16_t>(down_shift)));
            aie::store_v(
                output + 2 * output_stride + bin,
                aie::unpack(aie::mul<acc64>(
                    aie::filter_even(second_parts, 1), sign)
                    .to_vector<int16_t>(down_shift)));
            aie::store_v(
                output + 3 * output_stride + bin,
                aie::unpack(aie::mul<acc64>(
                    aie::filter_odd(second_parts, 1), sign)
                    .to_vector<int16_t>(down_shift)));

            const auto occupied =
                aie::eq(aie::load_v<16>(kOccupied + bin), int8_t(1));
            aie::store_v(
                output + 4 * output_stride + bin,
                aie::select(zero, occupied_value, occupied));
        }
        return shift;
    }

private:
    /**
     * Finds the significant-bit width of the two LTS observations.
     *
     * Sixteen FFT bins are reduced at a time. On occupied bins kLong is either
     * +1 or -1, so its sign correction does not change an absolute magnitude;
     * the occupancy mask removes guard and DC bins.
     *
     * @param first First long training symbol, FFT-shifted order.
     * @param second Second long training symbol, FFT-shifted order.
     * @return Significant bits of the largest occupied-bin component.
     */
    static inline unsigned magnitude_bits_of(
            const cint32 first[FFT_SIZE], const cint32 second[FFT_SIZE]) {
        uint32_t maximum = 0;
        const auto zero = aie::zeros<int32_t, 16>();

        for (int bin = 0; bin < FFT_SIZE; bin += 16) {
            const auto occupied =
                aie::eq(aie::load_v<16>(kOccupied + bin), int8_t(1));
            const auto first_parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(first + bin));
            const auto second_parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(second + bin));
            const auto first_real = aie::select(
                zero, aie::filter_even(first_parts, 1), occupied);
            const auto first_imag = aie::select(
                zero, aie::filter_odd(first_parts, 1), occupied);
            const auto second_real = aie::select(
                zero, aie::filter_even(second_parts, 1), occupied);
            const auto second_imag = aie::select(
                zero, aie::filter_odd(second_parts, 1), occupied);

            const auto magnitude = aie::max(
                aie::max(aie::abs(first_real), aie::abs(first_imag)),
                aie::max(aie::abs(second_real), aie::abs(second_imag)));
            const uint32_t lane_max =
                uint32_t(aie::reduce_max(magnitude));
            if (lane_max > maximum) maximum = lane_max;
        }
        return maximum == 0 ? 0 : 32 - __builtin_clz(maximum);
    }
};

/**
 * Same-padded 1-D convolution over the FFT bins, one output per bin.
 *
 * Every output channel accumulates InputChannels * KernelSize Q7 products in
 * a 64-bit accumulator, shifts the sum back to Q15, optionally applies ReLU,
 * and stores one 16-lane vector per step. The caller passes the first data
 * column and a padded stride, so taps that reach past a data edge read the
 * zero halo instead of branching, and the aligned stores stay valid.
 *
 * @tparam InputChannels Number of input feature channels.
 * @tparam OutputChannels Number of filters.
 * @tparam KernelSize Number of taps per filter.
 * @tparam ApplyRelu Whether to rectify the output.
 */
template <unsigned InputChannels, unsigned OutputChannels,
          unsigned KernelSize, bool ApplyRelu>
struct CnnConv1dImpl {
    /**
     * Evaluates one convolution layer for all FFT bins.
     *
     * @param input Base of the input channels at the first data column; row c
     *        starts at input + c * input_stride.
     * @param input_stride Elements between consecutive input channel rows.
     * @param output Base of the output channels at the first bin; row c starts
     *        at output + c * output_stride and must be vector-aligned.
     * @param output_stride Elements between consecutive output channel rows.
     * @param weights Q7 weights in [output][input][tap] layout.
     */
    static inline void forward(
            const int32_t *input, unsigned input_stride,
            int32_t *output, unsigned output_stride,
            const int8_t (&weights)[OutputChannels][InputChannels][KernelSize]) {
        const auto zero = aie::zeros<int32_t, 16>();
        for (unsigned output_channel = 0;
             output_channel < OutputChannels; ++output_channel) {
            for (int bin = 0; bin < FFT_SIZE; bin += 16) {
                aie::accum<acc64, 16> accumulator;
                accumulator.from_vector(zero);
                AIE_LOOP_NO_UNROLL
                for (unsigned input_channel = 0;
                     input_channel < InputChannels; ++input_channel) {
                    AIE_LOOP_NO_UNROLL
                    for (unsigned tap = 0; tap < KernelSize; ++tap) {
                        // Same padding: the tap window starts KernelSize / 2
                        // elements before the bin, reading the zero halo at the
                        // block edges. The load stays unaligned because of the
                        // tap offset, but the destination store does not.
                        const int input_offset =
                            int(input_channel * input_stride) + bin +
                            int(tap) - int(KernelSize / 2);
                        accumulator = aie::mac(
                            accumulator,
                            aie::load_unaligned_v<16>(input + input_offset),
                            int32_t(weights[output_channel][input_channel][tap]));
                    }
                }
                auto values = accumulator.to_vector<int32_t>(7);
                if (ApplyRelu) values = aie::max(values, zero);
                aie::store_v(
                    output + output_channel * output_stride + bin, values);
            }
        }
    }
};

/** Same-padded convolution without an activation. */
template <unsigned InputChannels, unsigned OutputChannels, unsigned KernelSize>
struct CnnConv1d
    : CnnConv1dImpl<InputChannels, OutputChannels, KernelSize, false> {};

/** Same-padded convolution followed by ReLU. */
template <unsigned InputChannels, unsigned OutputChannels, unsigned KernelSize>
struct CnnConv1dRelu
    : CnnConv1dImpl<InputChannels, OutputChannels, KernelSize, true> {};

/**
 * Applies the learned residual correction to the LTS channel estimate.
 *
 * The residual is returned to the raw input scale and saturated before it is
 * added to the least-squares estimate, so the network only refines occupied
 * subcarriers by a bounded amount. Unoccupied bins are cleared. The estimate
 * must be vector-aligned: the shared LTS initialization already stores to it
 * with aligned cint32 stores.
 */
struct CnnResidualAdd {
    /**
     * Adds the de-normalized CNN residual to the complex channel estimate.
     *
     * Sixteen bins are processed per iteration: the real and imaginary
     * residual rows are scaled back with the input normalization shift
     * (arithmetic downshift, or upshift with int32 saturation for a negative
     * shift), added to the deinterleaved least-squares estimate with
     * saturating adds, and written back as interleaved complex samples. The
     * occupied-subcarrier mask zeroes guard and DC bins.
     *
     * @param residual Residual rows; row 0 real, row 1 imaginary.
     * @param residual_stride Elements between the real and imaginary rows.
     * @param normalization_shift Shift returned by CnnInputNormalize::forward().
     * @param channel In/out channel estimate, FFT-shifted order; must meet the
     *        8-lane cint32 alignment required by aligned vector stores.
     */
    static inline void forward(
            const int32_t *residual, unsigned residual_stride,
            int normalization_shift, cint32 channel[FFT_SIZE]) {
        const auto zero = aie::zeros<int32_t, 16>();
        const unsigned up_shift =
            normalization_shift < 0 ? unsigned(-normalization_shift) : 0;
        const unsigned down_shift =
            normalization_shift > 0 ? unsigned(normalization_shift) : 0;
        for (int bin = 0; bin < FFT_SIZE; bin += 16) {
            auto real = aie::load_v<16>(residual + bin);
            auto imag = aie::load_v<16>(residual + residual_stride + bin);
            if (down_shift) {
                real = aie::downshift(real, down_shift);
                imag = aie::downshift(imag, down_shift);
            } else if (up_shift) {
                real = aie::upshift(real, up_shift);
                imag = aie::upshift(imag, up_shift);
            }

            const auto occupied =
                aie::eq(aie::load_v<16>(kOccupied + bin), int8_t(1));
            const auto parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(channel + bin));
            const auto out_real = aie::select(
                zero, aie::saturating_add(aie::filter_even(parts, 1), real),
                occupied);
            const auto out_imag = aie::select(
                zero, aie::saturating_add(aie::filter_odd(parts, 1), imag),
                occupied);
            const auto zipped = aie::interleave_zip(out_real, out_imag, 1);
            aie::store_v(
                channel + bin, aie::vector_cast<cint32>(zipped.first));
            aie::store_v(
                channel + bin + 8, aie::vector_cast<cint32>(zipped.second));
        }
    }
};

} // namespace

#endif
