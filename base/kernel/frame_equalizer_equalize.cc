#include <stdint.h>
#include <stddef.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "aie_kernel_utils.h"
#include "frame_equalizer_metadata.h"
#include "equalizer/select.h"

// Test builds can replace the CSI metadata with one LS-equalized vector for
// every payload symbol. Production builds retain one CSI vector per frame.
#ifndef FRAME_EQUALIZER_OUTPUT_SYMBOLS
#define FRAME_EQUALIZER_OUTPUT_SYMBOLS 0
#endif

namespace {
// The selected equalizer implementation is provided by equalizer/select.h,
// which defines update_channel_estimate() and reset_channel_estimate().

// Vectorized Viterbi transition tables. Entry s is the parity of
// ((s << 1) & polynomial), with the hypothesis bit omitted. The two tables
// correspond to generator polynomials 0155 (A) and 0117 (B); the hypothesis
// bit is XORed into the loaded vector at runtime because both masks include
// the least-significant bit of the transition register.
alignas(aie::vector_decl_align) static constexpr uint8_t kViterbiParityA[64] = {
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
};
alignas(aie::vector_decl_align) static constexpr uint8_t kViterbiParityB[64] = {
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
    1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
};
alignas(aie::vector_decl_align) static constexpr uint8_t kViterbiStates[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
};
// Persistent kernel state. Calls may contain only part of a frame, so channel
// estimates, tracking loops, and frame counters must survive between calls.
alignas(aie::vector_decl_align) static cint32 d_channel[64];
// comb.h reads and writes the channel estimate with aligned 8-lane vectors.
static_assert(__alignof__(d_channel) >= aie::vector_ldst_align_v<cint32, 8>,
              "d_channel must support aligned 8-lane vector loads");
alignas(aie::vector_decl_align) static cint32 d_equalized[64];
alignas(aie::vector_decl_align) static int32_t d_channel_power[64];
// The first 48 bytes are one decision per data carrier. The extra tail keeps
// the final overlapped vector store within bounds.
alignas(aie::vector_decl_align) static int8_t d_decisions[64];
// The final overlap copy reads a 16-byte tail after the 64 decision lanes.
// Static storage keeps this tail zero without adding a runtime clear.
alignas(aie::vector_decl_align) static int8_t d_decision_lanes[80];
static int8_t d_decoded[24];
// Working storage for the specialized 24-step SIGNAL-field decoder:
// - coded holds the deinterleaved pair of received bits for each trellis step;
// - survivor[t][s] is the predecessor of state s selected at step t;
// - metric[s] is the minimum accumulated Hamming distance ending in state s.
static int8_t d_viterbi_coded[48];
alignas(aie::vector_decl_align) static uint8_t d_viterbi_survivor[24][64];
alignas(aie::vector_decl_align) static uint8_t d_viterbi_metric[64];
alignas(aie::vector_decl_align) static uint8_t d_viterbi_next_metric[64];
static int d_equalizer_current_symbol = -1;
static int d_frame_symbols;
static int d_frame_bytes;
static int d_encoding;
static int d_modulation = 2;
static int32_t d_equalizer_cfo_q29;
static uint64_t d_snr_linear;
static int32_t d_center_frequency_mhz;
static int32_t d_signal_beta_q29;
static bool d_pending_tag;

// Hard-Viterbi L-SIG decoder (BPSK 1/2 rate).
static inline void decode_signal(const int8_t hard[48], int8_t decoded[24]) {
    // The convolutional encoder starts in state zero. Allowing every state to
    // start at zero cost weakens the short SIGNAL trellis under bit errors.
    const auto unreachable = aie::broadcast<uint8_t, 32>(uint8_t(100));
    aie::store_v(d_viterbi_metric, unreachable);
    aie::store_v(d_viterbi_metric + 32, unreachable);
    d_viterbi_metric[0] = 0;

    // deinterleave
    for (int i = 0; i < 48; ++i)
        d_viterbi_coded[i] = hard[3 * (i & 15) + (i >> 4)];

    // The full CSI profile puts SIGNAL decode and parsing at about 3.5k
    // cycles/tile (15.72% of equalize) on the 130-byte workload.
    // Per 2 interleaved A, B symbols (A[n] = XOR(x[n], x[n-2], x[n-3],
    // x[n-5], x[n-6]), B[n] = XOR(x[n], x[n-1], x[n-2], x[n-3], x[n-6])):
    for (int index = 0; index < 24; ++index) {
        // For a fixed hypothesis bit, next = ((state << 1) | bit) & 63.
        // States s and s + 32 therefore target the same next state. Keep the
        // two predecessor ranges separate so their metrics can be reduced in
        // parallel across 32 lanes.
        const auto state_lo = aie::load_v<32>(kViterbiStates);  //0 to 31
        const auto state_hi = aie::add(state_lo, uint8_t(32));  //31 to 63
        const auto metric_lo = aie::load_v<32>(d_viterbi_metric);   //hamming dist
        const auto metric_hi = aie::load_v<32>(d_viterbi_metric + 32);
        aie::vector<uint8_t, 32> best[2];
        aie::vector<uint8_t, 32> survivor[2];

        // Each bit value maps state pairs (s, s + 32) to one destination.
        // This replaces the scalar state loop. The old loop visited s before
        // s + 32; using a strict less-than comparison below preserves that
        // lower-state-wins tie rule.
        for (int bit = 0; bit < 2; ++bit) { //hypothesis bit
            const uint8_t coded_a = (uint8_t)d_viterbi_coded[2 * index]; //demodulated bit
            const uint8_t coded_b = (uint8_t)d_viterbi_coded[2 * index + 1];
            const auto bit_v = aie::broadcast<uint8_t, 32>(uint8_t(bit));   //broadcast(hypothesis bit)
            const auto parity_a = aie::bit_xor(
                aie::load_v<32>(kViterbiParityA), bit_v);   //Expected bit at coded_a, per each state. A[n]  = XOR(x[n], x[n-2], x[n-3], x[n-5], x[n-6]).  (for first 32 states - using a LUT)
            const auto parity_b = aie::bit_xor(
                aie::load_v<32>(kViterbiParityB), bit_v);   //Expected bit at coded_b, per each state. B[n] = XOR(x[n], x[n-1], x[n-2], x[n-3], x[n-6] (for first 32 sates)
            const auto parity_a_hi = aie::bit_xor(
                aie::load_v<32>(kViterbiParityA + 32), bit_v);  //ditto, for the rest of states
            const auto parity_b_hi = aie::bit_xor(
                aie::load_v<32>(kViterbiParityB + 32), bit_v);

            // Build the Hamming distance vector by comparing both generator
            // parity vectors with the two received coded bits.
            const auto distance_lo = aie::add(
                aie::select(uint8_t(0), uint8_t(1), aie::neq(parity_a, coded_a)),
                aie::select(uint8_t(0), uint8_t(1), aie::neq(parity_b, coded_b)));
            const auto distance_hi = aie::add(
                aie::select(uint8_t(0), uint8_t(1), aie::neq(parity_a_hi, coded_a)),
                aie::select(uint8_t(0), uint8_t(1), aie::neq(parity_b_hi, coded_b)));
            const auto candidate_lo = aie::add(metric_lo, distance_lo);
            const auto candidate_hi = aie::add(metric_hi, distance_hi); //128 states
            const auto choose_hi = aie::lt(candidate_hi, candidate_lo); //reduce to 64 (ya know, the viterbi decoder thing.)
            best[bit] = aie::select(candidate_lo, candidate_hi, choose_hi);
            survivor[bit] = aie::select(state_lo, state_hi, choose_hi);
        }
        // Bit zero produces even next-state indices and bit one produces odd
        // indices. Interleave the two 32-lane results back into the original
        // 64-state layout before storing the next metrics and survivors.
        const auto best_interleaved = aie::interleave_zip(best[0], best[1], 1);
        const auto survivor_interleaved = aie::interleave_zip(survivor[0], survivor[1], 1);
        // Both candidates for every next state were reduced above, so the
        // previous scalar sentinel initialization is no longer necessary.
        aie::store_v(d_viterbi_next_metric, best_interleaved.first);
        aie::store_v(d_viterbi_next_metric + 32, best_interleaved.second);
        aie::store_v(d_viterbi_survivor[index], survivor_interleaved.first);
        aie::store_v(d_viterbi_survivor[index] + 32, survivor_interleaved.second);
        aie::store_v(d_viterbi_metric, aie::load_v<32>(d_viterbi_next_metric));
        aie::store_v(
            d_viterbi_metric + 32,
            aie::load_v<32>(d_viterbi_next_metric + 32));
    }
    //Trellis diagram forward pass end.

    //trace back and recover the payload.
    int state = 0;  //L-SIG's last 6 bits are 0, so we can just start from the state of 0b(000 000). (Note: this is not how the original gr-ieee802-11 behaves - that uses more generic implementation. This implementation is actually a little better, so I am not replicating that behavior.)
    AIE_LOOP_NO_UNROLL
    for (int i = 23; i >= 0; --i) {
        decoded[i] = state & 1; //decoded bit (0 or 1).
        state = d_viterbi_survivor[i][state];
    }
}

static inline bool parse_signal(const int8_t bits[24]) {
    // SIGNAL fields are LSB-first: four RATE bits, one reserved bit, twelve
    // LENGTH bits, one even-parity bit, then six zero tail bits.
    int rate = 0, length = 0, parity = 0, n_dbps;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 17; ++i) {
        parity ^= bits[i];
        if (i < 4) rate |= bits[i] << i;
        else if (i > 4) length |= bits[i] << (i - 5);
    }
    if (parity != bits[17]) return false;
    // Map the legacy RATE code to the downstream encoding ID, constellation
    // size, and N_DBPS (uncoded payload bits carried by one OFDM symbol).
    switch (rate) {
    case 11: d_encoding = 0; d_modulation = 2; n_dbps = 24; break;
    case 15: d_encoding = 1; d_modulation = 2; n_dbps = 36; break;
    case 10: d_encoding = 2; d_modulation = 4; n_dbps = 48; break;
    case 14: d_encoding = 3; d_modulation = 4; n_dbps = 72; break;
    case 9: d_encoding = 4; d_modulation = 16; n_dbps = 96; break;
    case 13: d_encoding = 5; d_modulation = 16; n_dbps = 144; break;
    case 8: d_encoding = 6; d_modulation = 64; n_dbps = 192; break;
    case 12: d_encoding = 7; d_modulation = 64; n_dbps = 216; break;
    default: return false;
    }
    d_frame_bytes = length;
    // Include 16 SERVICE and six convolutional-encoder tail bits, then round
    // upward to a whole number of OFDM payload symbols.
    d_frame_symbols = (16 + 8 * length + 6 + n_dbps - 1) / n_dbps;
    return true;
}

// Return magnitude * 32768 - power * threshold. Narrowing the acc64 result
// with saturation preserves its sign, which is all the demapper needs.
static inline aie::vector<int32_t, 16> dimension_residual(
        const aie::vector<int32_t, 16> &magnitude,
        const aie::vector<int32_t, 16> &power, int32_t threshold) {
    return aie::msc(
        aie::mul<acc64>(magnitude, int32_t(32768)), power, threshold)
        .to_vector<int32_t>(0);
}

static NOINLINE aie::vector<int8_t, 16> dimension_codes_16(
        aie::vector<int32_t, 16> value, aie::vector<int32_t, 16> power) {
    const auto positive = aie::ge(value, int32_t(0));
    // Saturating abs maps INT_MIN to INT_MAX, which remains beyond every
    // possible normalized boundary and therefore preserves the outer level.
    const auto magnitude = aie::abs(value);
    const auto inner = aie::le(
        dimension_residual(magnitude, power, 20724), int32_t(0));
    return aie::add(
        aie::select(int8_t(0), int8_t(2), inner),
        aie::select(int8_t(0), int8_t(1), positive));
}

static NOINLINE aie::vector<int8_t, 16> dimension_codes_64(
        aie::vector<int32_t, 16> value, aie::vector<int32_t, 16> power) {
    const auto positive = aie::ge(value, int32_t(0));
    const auto magnitude = aie::abs(value);
    const auto beyond_0 = aie::ge(
        dimension_residual(magnitude, power, 10112), int32_t(0));
    const auto beyond_1 = aie::ge(
        dimension_residual(magnitude, power, 20225), int32_t(0));
    const auto beyond_2 = aie::ge(
        dimension_residual(magnitude, power, 30337), int32_t(0));
    auto negative = aie::select(int8_t(2), int8_t(6), beyond_0);
    negative = aie::select(negative, int8_t(4), beyond_1);
    negative = aie::select(negative, int8_t(0), beyond_2);
    return aie::add(
        negative, aie::select(int8_t(0), int8_t(1), positive));
}
struct OverlapCopy {
    uint8_t source;
    uint8_t shift;
    uint8_t destination;
};

static constexpr OverlapCopy kDataDecisionCopies[] = {
    { 0, 6, 0 }, { 0, 12, 5 }, { 16, 10, 18 },
    { 32, 1, 24 }, { 32, 8, 30 }, { 48, 6, 43 },
};
static constexpr OverlapCopy kCsiCopies[] = {
    { 0, 6, 0 }, { 16, 0, 10 }, { 32, 1, 26 }, { 32, 11, 36 },
};
template <typename T, unsigned VectorSize, size_t CopyCount>
static inline void copy_overlapped(const T *source, T *destination,
                                   const OverlapCopy (&copies)[CopyCount]) {
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_RANGE(CopyCount, CopyCount)
    for (const OverlapCopy copy : copies) {
        const auto lo = aie::load_v<VectorSize>(source + copy.source);
        const auto hi = aie::load_v<VectorSize>(source + copy.source + VectorSize);
        const auto packed = aie::shuffle_down_fill(lo, hi, copy.shift);
        aie::store_unaligned_v<VectorSize>(destination + copy.destination, packed);
    }
} 

static NOINLINE void copy_data_decisions(const int8_t source[80], int8_t destination[64]) {
    copy_overlapped<int8_t, 16>(source, destination, kDataDecisionCopies);
}

static inline void hard_decision(const cint32 equalized[FFT_SIZE],
                                 const int32_t channel_power[FFT_SIZE],
                                 int8_t decisions[64], int modulation) {
    // A hardware microbenchmark of the QAM path measured 96 scalar dimension
    // decisions versus four 16-carrier SIMD batches. QAM16 fell from 3903 to
    // 279 cycles and QAM64 from 6635 to 351 cycles per symbol workload.
    // In the full CSI profile, hard decisions measured about 5.49k cycles/tile
    // (24.48% of equalize) on the 130-byte workload.
    if (modulation == 2 || modulation == 4) {
        const auto qpsk_mask = aie::mask<16>(modulation == 4);
        for (int bin = 0; bin < FFT_SIZE; bin += 16) {
            const auto parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(equalized + bin));
            const auto real = aie::filter_even(parts, 1);
            const auto imag = aie::filter_odd(parts, 1);
            const auto real_decisions = aie::select(
                int8_t(0), int8_t(1), aie::ge(real, int32_t(0)));
            const auto imag_decisions = aie::select(
                int8_t(0), int8_t(2), aie::ge(imag, int32_t(0)));
            const auto qpsk_decisions = aie::add(real_decisions, imag_decisions);
            aie::store_v(d_decision_lanes + bin,
                         aie::select(real_decisions, qpsk_decisions, qpsk_mask));
        }
        copy_data_decisions(d_decision_lanes, decisions);
    } else {
        aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
        for (int bin = 0; bin < FFT_SIZE; bin += 16) {
            const auto parts = aie::vector_cast<int32_t>(
                aie::load_v<16>(equalized + bin));
            const auto real = aie::filter_even(parts, 1);
            const auto imag = aie::filter_odd(parts, 1);
            const auto power = aie::load_v<16>(channel_power + bin);
            aie::vector<int8_t, 16> x, y;
            if (modulation == 16) {
                x = dimension_codes_16(real, power);
                y = dimension_codes_16(imag, power);
                const auto y2 = aie::add(y, y);
                aie::store_v(
                    d_decision_lanes + bin,
                    aie::add(x, aie::add(y2, y2)));
            } else {
                x = dimension_codes_64(real, power);
                y = dimension_codes_64(imag, power);
                const auto y2 = aie::add(y, y);
                const auto y4 = aie::add(y2, y2);
                aie::store_v(
                    d_decision_lanes + bin,
                    aie::add(x, aie::add(y4, y4)));
            }
        }
        copy_data_decisions(d_decision_lanes, decisions);
    }
}

static inline void copy_csi(frame_equalizer_csi csi[FRAME_EQUALIZER_CSI_SIZE]) {
    auto *destination = reinterpret_cast<cint32 *>(csi);

    // Zero padding first; the final overlapped data store restores CSI 48..51.
    aie::store_unaligned_v<16>(destination + 48, aie::zeros<cint32, 16>());
    copy_overlapped<cint32, 16>(d_channel, destination, kCsiCopies);
}

//////
//these two functions are only used for debugging purposes (for direct gr-ieee-80211 vs NPU kernel module comparison)
/////
#if FRAME_EQUALIZER_OUTPUT_SYMBOLS
static inline int32_t normalized_component_q15(int32_t numerator,
                                               int32_t channel_power) {
    if (channel_power <= 0) return 0;
    const int64_t value = ((int64_t)numerator << 15) / channel_power;
    if (value > INT32_MAX) return INT32_MAX;
    if (value < INT32_MIN) return INT32_MIN;
    return (int32_t)value;
}

static inline void copy_equalized(frame_equalizer_csi symbols[FRAME_EQUALIZER_CSI_SIZE]) {
    // d_equalized and d_channel_power contain the matched-filter numerator and
    // |H|^2 with the same shift. Divide them here to reproduce GNU Radio's
    // sample/H output in signed Q16.15. This division is test-output-only; the
    // production hard demapper remains in the faster numerator/power domain.
    int output = 0;
    for (int bin = 6; bin <= 58; ++bin) {
        if (bin == 11 || bin == 25 || bin == 32 || bin == 39 || bin == 53)
            continue;
        symbols[output].real = normalized_component_q15(
            d_equalized[bin].real, d_channel_power[bin]);
        symbols[output].imag = normalized_component_q15(
            d_equalized[bin].imag, d_channel_power[bin]);
        ++output;
    }
    for (; output < FRAME_EQUALIZER_CSI_SIZE; ++output)
        symbols[output] = { 0, 0 };
}
#endif

static inline void fill_tag(frame_equalizer_tag_metadata &tag, int offset) {
    // Attach frame information to the first payload output. Test builds instead
    // attach one normalized equalized-symbol vector to every payload output.
    tag.offset = offset;
    tag.frame_start = d_equalizer_current_symbol == 3 ? 1u : 0u;
    tag.frame_bytes = d_frame_bytes;
    tag.encoding = d_encoding; tag.snr_linear = d_snr_linear;
    tag.center_frequency_mhz = d_center_frequency_mhz;
    // Keep both phase values in Q3.29; GNU Radio converts them to float tags.
    tag.frequency_offset = d_equalizer_cfo_q29;
    tag.beta = d_signal_beta_q29;
#if FRAME_EQUALIZER_OUTPUT_SYMBOLS
        copy_equalized(tag.csi);
#else
        copy_csi(tag.csi);
#endif
}
static NOINLINE void reset_equalization(int32_t cfo_q29) {
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

#if !FRAME_EQUALIZER_OUTPUT_SYMBOLS
        // GNU Radio publishes frame CSI immediately after SIGNAL equalization.
        // Emit the delayed stream tag before the estimator incorporates the
        // first payload symbol, avoiding a separate 64-bin channel snapshot.
        if (d_equalizer_current_symbol > 2 && d_pending_tag &&
            metadata->tag_count < FRAME_EQUALIZER_MAX_TAGS) {
            fill_tag(metadata->tags[metadata->tag_count++], produced);
            d_pending_tag = false;
        }
#endif
        update_channel_estimate(prepared_symbol, d_equalizer_current_symbol,
                                d_channel, d_snr_linear);
        if (d_equalizer_current_symbol >= 2) {
            // SIGNAL is always BPSK. Later symbols use the modulation selected
            // by SIGNAL; pilots, guards, and DC never enter d_decisions.
            {
                // Equalization plus QAM channel-power generation measured about
                // 1.51k cycles/tile (6.72% of equalize) in the full CSI profile.
                // Equalize every bin in contiguous 8-lane batches.
                aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
                aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
                for (int bin = 0; bin < 64; bin += 8) {
                    const auto sample_v = aie::load_v<8>(prepared_symbol + bin);
                    const auto channel_v = aie::load_v<8>(d_channel + bin);
                    aie::store_v(
                        d_equalized + bin,
                        aie::mul<cacc64>(sample_v, aie::op_conj(channel_v))
                            .to_vector<cint32>(12));
                }
                // BPSK and QPSK need only signs; avoid power calculation for
                // those modes and keep the QAM path branch-free per batch.
                if (d_modulation > 4 || FRAME_EQUALIZER_OUTPUT_SYMBOLS) {
                    for (int bin = 0; bin < 64; bin += 8) {
                        const auto channel_v = aie::load_v<8>(d_channel + bin);
                        aie::store_v(
                            d_channel_power + bin,
                            aie::abs_square(channel_v, 12));
                    }
                }
            }
            int _modulation = d_equalizer_current_symbol == 2 ? 2 : d_modulation;
            hard_decision(d_equalized, d_channel_power, d_decisions, _modulation);
            // Decision-directed estimators update the channel after demapping,
            // so this symbol was equalized with the previous estimate. LS and
            // comb use a no-op here.
            track_channel_estimate(prepared_symbol, d_decisions, _modulation,
                                   d_equalizer_current_symbol, d_channel);
            if (d_equalizer_current_symbol == 2) {
                d_signal_beta_q29 = beta;
                decode_signal(d_decisions, d_decoded);
                d_pending_tag = parse_signal(d_decoded);
            } else {
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
        }
        ++d_equalizer_current_symbol;
    }
    metadata->output_length = produced;
}
