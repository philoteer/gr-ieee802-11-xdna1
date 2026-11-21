/** @file @brief Shared tables, helpers, and metadata for the decode_mac kernel. */
#ifndef DECODE_MAC_H
#define DECODE_MAC_H

#include <stdint.h>
#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "frame_equalizer_metadata.h"
#include "viterbi_tables.h"

// ---------------------------------------------------------------------------
// Host-visible tile metadata and buffer limits.
// ---------------------------------------------------------------------------

constexpr int DECODE_MAC_MAX_PSDU = 4095;
constexpr int DECODE_MAC_OUTPUT_BYTES = 8192;
constexpr int DECODE_MAC_MAX_FRAMES = 16;

struct alignas(8) decode_mac_frame_metadata {
    int32_t offset; // byte offset into this call's output buffer
    int32_t length; // MAC bytes, without SERVICE or FCS
    frame_equalizer_tag_metadata phy; // metadata captured at frame start
};

struct alignas(8) decode_mac_tile_metadata {
    int32_t output_length;
    int32_t frame_count;
    decode_mac_frame_metadata frames[DECODE_MAC_MAX_FRAMES];
};

static_assert(sizeof(decode_mac_frame_metadata) == 568, "MAC frame ABI changed");
static_assert(sizeof(decode_mac_tile_metadata) == 9096, "MAC tile ABI changed");

namespace decode_mac_detail {

// ---------------------------------------------------------------------------
// Precomputed hard-decision branch scores for the DATA trellis.
// ---------------------------------------------------------------------------

struct alignas(aie::vector_decl_align) viterbi_branch_scores {
    // One 64-byte row per (a,b), including erasure 2. Padding b to four choices
    // makes the index 4*a+b rather than 3*a+b (one cycle/step faster on AIE2P).
    // Columns 0--31 score low predecessors with hypothesis 0; columns 32--63
    // score those SAME predecessors with hypothesis 1. Every row is aligned.
    uint8_t row[12][64] = {};

    constexpr viterbi_branch_scores() {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int state = 0; state < 64; ++state)
                    row[4 * a + b][state] =
                        (a != 2 && kViterbiParityA[state] == a) +
                        (b != 2 && kViterbiParityB[state] == b);
    }
};

// Compile-time initialization only: no per-frame table construction.
static constexpr viterbi_branch_scores kViterbiBranchScores{};
static_assert(sizeof(kViterbiBranchScores) == 12 * 64);

// ---------------------------------------------------------------------------
// Byte-at-a-time reflected IEEE CRC-32 for the MAC FCS.
// ---------------------------------------------------------------------------

struct crc32_table {
    uint32_t entry[256] = {};

    constexpr crc32_table() {
        for (unsigned byte = 0; byte < 256; ++byte) {
            uint32_t crc = byte;
            for (unsigned bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320u : 0u);
            entry[byte] = crc;
        }
    }
};

// Compile-time initialization only. The running CRC is not complemented here.
static constexpr crc32_table kCrc32{};
static_assert(sizeof(kCrc32) == 1024);

static inline uint32_t crc32_byte(uint32_t crc, uint8_t byte) {
    return (crc >> 8) ^ kCrc32.entry[(crc ^ byte) & 255];
}

// ---------------------------------------------------------------------------
// Byte-at-a-time DATA descrambler tables.
// ---------------------------------------------------------------------------

struct descramble_tables {
    // Low byte: eight feedback bits in PSDU (LSB-first) order.
    // High byte: LFSR state after those eight steps.
    uint16_t step[128] = {};
    uint8_t reverse[256] = {};

    constexpr descramble_tables() {
        for (unsigned state = 0; state < 128; ++state) {
            unsigned next = state, mask = 0;
            for (unsigned bit = 0; bit < 8; ++bit) {
                const unsigned feedback = ((next >> 6) ^ (next >> 3)) & 1;
                mask |= feedback << bit;
                next = ((next << 1) & 0x7e) | feedback;
            }
            step[state] = (next << 8) | mask;
        }
        // Traceback emits each byte MSB first; PSDU bytes are LSB first.
        for (unsigned byte = 0; byte < 256; ++byte)
            for (unsigned bit = 0; bit < 8; ++bit)
                reverse[byte] |= ((byte >> bit) & 1) << (7 - bit);
    }
};

// Compile-time initialization, shared by every frame. Include state zero too:
// malformed/noisy SERVICE bits must behave exactly like the bitwise decoder.
static constexpr descramble_tables kDescramble{};
static_assert(sizeof(kDescramble) == 512);

// ---------------------------------------------------------------------------
// SIMD unpacking and DATA-bit deinterleaving for one OFDM symbol.
// ---------------------------------------------------------------------------

// Each input byte holds Bpsc packed bits for one of the 48 data carriers.
// The output holds one byte (0 or 1) per coded bit, in Viterbi input order.
// For carrier stream t in [0, 3), bit plane b in [0, Bpsc), and lane r in [0, 16):
// out[16 * (t * Bpsc + b) + r] = (in[3*r + t] >> shift) & 1,
// shift = s * (b / s) + ((b + s - r % s) % s), s = max(Bpsc/2, 1).
// Reorder the packed bytes first, undo the within-group bit rotations once,
// then extract each bit plane with the same shift across all sixteen lanes.

/// @brief Reorder 48 packed decisions into three sixteen-carrier streams.
// Inter-core FIFO buffers are not necessarily vector aligned. In particular,
// the AIE2 allocator can place these after metadata at an 8-byte boundary.
static inline aie::vector<uint8_t, 64> deinterleave_carriers(const uint8_t *in) {
    // Use a supported 64-lane vector; only its first 48 lanes carry symbol data.
    auto carriers = aie::zeros<uint8_t, 64>();
    // Fill lanes 0--15, 16--31, and 32--47 without reading past the symbol.
    carriers.insert<16>(0, aie::load_unaligned_v<16>(in));
    carriers.insert<16>(1, aie::load_unaligned_v<16>(in + 16));
    carriers.insert<16>(2, aie::load_unaligned_v<16>(in + 32));
    // Four half-symbol zips transpose the 16 x 3 carrier layout to 3 x 16:
    // lanes 0--15 = in[0,3,...,45], 16--31 = in[1,4,...,46],
    // and 32--47 = in[2,5,...,47]. Lanes 48--63 are scratch, not output.
    AIE_LOOP_UNROLL_FULL
    for (int stage = 0; stage < 4; ++stage) {
        // Move the second 24-byte half to lane zero; upper fill lanes are unused.
        const auto second_half = aie::shuffle_down(carriers, 24);
        // Zip one byte from each half: a0,b0,a1,b1,...,a23,b23.
        // The API takes 32-lane inputs; its trailing sixteen results are ignored.
        const auto zipped = aie::interleave_zip(
            carriers.extract<32>(0), second_half.extract<32>(0), 1);
        // Reassemble both 32-byte results for the next zip stage.
        carriers = aie::concat(zipped.first, zipped.second);
    }
    return carriers;
}

/// @brief Undo the second interleaver permutation within each decision byte.
template<int Bpsc>
static inline aie::vector<uint8_t, 16> normalize_bits(aie::vector<uint8_t, 16> v) {
    if constexpr (Bpsc == 4) {
        // 16-QAM: swap bits 0/1 and 2/3, but only for odd carrier-stream lanes.
        const auto swapped = aie::bit_or(
            // Move bits 0 and 2 up to positions 1 and 3.
            aie::upshift(aie::bit_and(uint8_t(0x05), v), 1),
            // Move bits 1 and 3 down to positions 0 and 2.
            aie::logical_downshift(aie::bit_and(uint8_t(0x0a), v), 1));
        // Mask bit r selects the swapped byte for lane r; 0xaaaa marks odd lanes.
        return aie::select(v, swapped, aie::mask<16>::from_uint32(0xaaaa));
    } else if constexpr (Bpsc == 6) {
        // 64-QAM: rotate bits 0--2 and 3--5 left separately by lane % 3.
        // Mask before upshifting so no intermediate overflows a byte, even
        // when the caller has enabled saturation.
        const auto rotate_one = aie::bit_or(
            // Advance bits 0,1,3,4; wrap bits 2,5 to positions 0,3.
            aie::upshift(aie::bit_and(uint8_t(0x1b), v), 1),
            aie::logical_downshift(aie::bit_and(uint8_t(0x24), v), 2));
        const auto rotate_two = aie::bit_or(
            // Advance bits 0,3 twice; wrap bits 1,2,4,5 down one position.
            aie::upshift(aie::bit_and(uint8_t(0x09), v), 2),
            aie::logical_downshift(aie::bit_and(uint8_t(0x36), v), 1));
        // Select rotation one for lanes 1,4,7,10,13.
        v = aie::select(v, rotate_one, aie::mask<16>::from_uint32(0x2492));
        // Select rotation two for lanes 2,5,8,11,14; lanes 0,3,...,15 stay unchanged.
        return aie::select(v, rotate_two, aie::mask<16>::from_uint32(0x4924));
    } else {
        // BPSK and QPSK have s == 1, so no within-byte permutation is needed.
        return v;
    }
}

/// @brief Expand sixteen packed decisions into Bpsc contiguous bit-plane rows.
template<int Bpsc>
static inline void unpack_stream(aie::vector<uint8_t, 16> carriers, uint8_t *out) {
    // Normalize once and reuse the same sixteen bytes for every bit plane.
    const auto normalized = normalize_bits<Bpsc>(carriers);
    // Unrolling makes each shift a compile-time constant, not a per-lane value.
    AIE_LOOP_UNROLL_FULL
    for (int bit = 0; bit < Bpsc; ++bit)
        // Row bit holds bit 'bit' from all sixteen carriers as byte values 0/1.
        aie::store_v(out + 16 * bit, aie::bit_and(
            uint8_t(1), aie::logical_downshift(normalized, bit)));
}

/**
 * @brief Unpack and deinterleave one symbol without per-bit division or scatter.
 * @param in Forty-eight packed carrier-decision bytes.
 * @param out Space for 48 * Bpsc coded-bit bytes, in Viterbi input order.
 * @pre out is 16-byte aligned; the buffers do not overlap.
 */
template<int Bpsc>
static inline void unpack_deinterleave(const uint8_t *in, uint8_t *out) {
    // Instantiate only the four legacy OFDM modulation sizes.
    static_assert(Bpsc == 1 || Bpsc == 2 || Bpsc == 4 || Bpsc == 6,
                  "unsupported OFDM modulation");
    // Reorder while bits are still packed, rather than permuting 48 * Bpsc bytes.
    const auto carriers = deinterleave_carriers(in);
    // Each sixteen-carrier stream produces 16 * Bpsc consecutive output bytes.
    unpack_stream<Bpsc>(carriers.extract<16>(0), out);
    unpack_stream<Bpsc>(carriers.extract<16>(1), out + 16 * Bpsc);
    unpack_stream<Bpsc>(carriers.extract<16>(2), out + 32 * Bpsc);
}

// ---------------------------------------------------------------------------
// SIMD insertion of erased observations before DATA Viterbi decoding.
// ---------------------------------------------------------------------------

/**
 * @brief Expand one complete OFDM symbol into consecutive Viterbi (a,b) bytes.
 * @param coded Deinterleaved observations, followed by at least four padding bytes.
 * @param pairs Scratch for 2 * dbps bytes, followed by at least eight padding bytes.
 * @param dbps Uncoded DATA bits per symbol for a legacy OFDM rate.
 * @param period Puncturing period: 2 (rate 1/2), 4 (2/3), or 6 (3/4).
 * @pre coded is 16-byte aligned; pairs is 32-byte aligned and does not overlap it.
 * @return coded for rate 1/2 (no copy), otherwise pairs.
 *
 * Every legacy OFDM symbol contains an integral number of puncturing periods,
 * so each symbol starts at phase zero, including the symbol containing TAIL/PAD.
 */
static inline const uint8_t *depuncture_symbol(
        const uint8_t *coded, uint8_t *pairs, int dbps, int period) {
    if (period == 2) return coded;

    if (period == 4) {
        // Twelve transmitted bytes become sixteen: abc def ghi jkl ->
        // abc2 def2 ghi2 jkl2. Only the final load uses the input padding.
        AIE_LOOP_NO_UNROLL
        for (int out = 0, in = 0; out < 2 * dbps; out += 16, in += 12) {
            const auto v = aie::load_unaligned_v<16>(coded + in);
            auto expanded = aie::select(
                v, aie::shuffle_up(v, 1), aie::mask<16>::from_uint32(0x00f0));
            expanded = aie::select(expanded, aie::shuffle_up(v, 2),
                                  aie::mask<16>::from_uint32(0x0f00));
            expanded = aie::select(expanded, aie::shuffle_up(v, 3),
                                  aie::mask<16>::from_uint32(0xf000));
            expanded = aie::select(expanded, uint8_t(2),
                                  aie::mask<16>::from_uint32(0x8888));
            aie::store_v(pairs + out, expanded);
        }
    } else {
        // Sixteen transmitted bytes become twenty-four: abcd -> abc22d.
        // Shift each following four-byte group by two more output positions.
        AIE_LOOP_NO_UNROLL
        for (int out = 0, in = 0; out < 2 * dbps; out += 24, in += 16) {
            const auto v = aie::load_v<16>(coded + in).grow<32>();
            auto expanded = aie::select(
                v, aie::shuffle_up(v, 2), aie::mask<32>::from_uint32(0x000001e0));
            expanded = aie::select(expanded, aie::shuffle_up(v, 4),
                                  aie::mask<32>::from_uint32(0x00007800));
            expanded = aie::select(expanded, aie::shuffle_up(v, 6),
                                  aie::mask<32>::from_uint32(0x001e0000));
            expanded = aie::select(expanded, aie::shuffle_up(v, 8),
                                  aie::mask<32>::from_uint32(0x00800000));
            // Mark all non-observation lanes as erasures, including the store tail.
            expanded = aie::select(expanded, uint8_t(2),
                                  aie::mask<32>::from_uint32(0xff618618));
            // Eight trailing bytes overlap the next store, or land in padding.
            aie::store_unaligned_v(pairs + out, expanded);
        }
    }
    return pairs;
}

} // namespace decode_mac_detail

#endif
