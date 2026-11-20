#include <stdint.h>
#include <aie_api/aie.hpp>

#include "common/decode_mac.h"

#define NOINLINE __attribute__((noinline))

namespace {
alignas(aie::vector_decl_align) static uint8_t d_metric[64];
alignas(aie::vector_decl_align) static uint8_t d_path[64];
alignas(aie::vector_decl_align) static uint8_t d_history[10][64];
alignas(aie::vector_decl_align) static uint8_t d_coded[288 + 16];
alignas(aie::vector_decl_align) static uint8_t d_pairs[2 * 216 + 32];
alignas(aie::vector_decl_align) static uint8_t d_psdu[DECODE_MAC_MAX_PSDU + 32];
static frame_equalizer_tag_metadata d_phy;
static int d_bpsc, d_dbps, d_symbols, d_copied;
static int d_steps, d_windows, d_store_pos, d_traceback;
static int d_bits, d_scrambler, d_puncture_period;
static bool d_active;

#define BPSK 1
#define QPSK 2
#define _16QAM 4
#define _64QAM 6

static constexpr int kBpsc[8] = {BPSK, BPSK, QPSK, QPSK, _16QAM, _16QAM, _64QAM, _64QAM};	
static constexpr int kDbps[8] = {24, 36, 48, 72, 96, 144, 192, 216};

static NOINLINE void begin_frame(const frame_equalizer_tag_metadata &tag) {
    if (tag.encoding > 7 || tag.frame_bytes < 4 ||
        tag.frame_bytes > DECODE_MAC_MAX_PSDU) {
			d_active = false;
			return;
	}

    d_phy = tag;
    const int encoding = (int)tag.encoding;
    d_bpsc = kBpsc[encoding];
    d_dbps = kDbps[encoding];
    d_symbols = (16 + 8 * (int)tag.frame_bytes + 6 + d_dbps - 1) / d_dbps;
    d_copied = 0;
    d_steps = 0;
    d_windows = 0;
    d_store_pos = 0;
    d_bits = 0;
    d_scrambler = 0;
    d_traceback = encoding == 6 ? 9 : (encoding == 0 || encoding == 2 || encoding == 4 ? 5 : 10);
    d_puncture_period = d_traceback == 5 ? 2 : (d_traceback == 9 ? 4 : 6);
    const auto zero = aie::zeros<uint8_t, 32>();
    aie::store_v(d_metric, zero); 
    aie::store_v(d_metric + 32, zero);
    
    aie::store_v(d_path, zero); 
    aie::store_v(d_path + 32, zero);
    
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 10; ++i) {
        aie::store_v(d_history[i], zero); 
        aie::store_v(d_history[i] + 32, zero);
    }
    d_active = true;
}

static inline void descramble_byte(uint8_t bits) {
    if (d_bits == 0) {
        const int seed = bits >> 1;
        const int feedback = ((seed >> 6) ^ (seed >> 3)) & 1;
        d_scrambler = ((seed << 1) & 0x7e) | feedback;
    } else if (d_bits < 16 + 8 * (int)d_phy.frame_bytes) {
        const uint16_t step = decode_mac_detail::kDescramble.step[d_scrambler];
        if (d_bits >= 16)
            d_psdu[(d_bits - 16) / 8] =
                decode_mac_detail::kDescramble.reverse[bits] ^ uint8_t(step);
        d_scrambler = step >> 8;
    }
    d_bits += 8;
}

static NOINLINE void traceback_byte() {
    if (++d_store_pos == d_traceback)
        d_store_pos = 0;
    aie::store_v(d_history[d_store_pos], aie::load_v<32>(d_path));
    aie::store_v(d_history[d_store_pos] + 32, aie::load_v<32>(d_path + 32));
    const auto metrics = aie::load_v<64>(d_metric);
    const uint8_t maximum = aie::reduce_max(metrics);
    int state = aie::eq(aie::reverse(metrics), maximum).clz();
    int position = d_store_pos;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < d_traceback - 1; ++i) {
        state = d_history[position][state] >> 2;
        position = position == 0 ? d_traceback - 1 : position - 1;
    }
    const bool tail_end = d_steps == 16 + 8 * (int)d_phy.frame_bytes + 6;
    if (d_windows >= d_traceback && !tail_end)
        descramble_byte(d_history[position][state]);
    ++d_windows;
    if ((d_windows & 7) == 0) {
        const uint8_t minimum = aie::reduce_min(metrics);
        aie::store_v(d_metric, aie::sub(metrics.extract<32>(0), minimum));
        aie::store_v(d_metric + 32, aie::sub(metrics.extract<32>(1), minimum));
    }
    aie::store_v(d_path, aie::zeros<uint8_t, 32>());
    aie::store_v(d_path + 32, aie::zeros<uint8_t, 32>());
}

static NOINLINE void viterbi_run(const uint8_t *pairs, int count) {
    auto lo = aie::load_v<32>(d_metric);
    auto hi = aie::load_v<32>(d_metric + 32);
    auto path_lo = aie::load_v<32>(d_path);
    auto path_hi = aie::load_v<32>(d_path + 32);
    AIE_LOOP_NO_UNROLL
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(1, 8)
    for (int step = 0; step < count; ++step) {
        const auto shifted_lo = aie::add(path_lo, path_lo);
        const auto shifted_hi = aie::add(aie::add(path_hi, path_hi), uint8_t(1));
        const auto score_lo = aie::load_v<64>(
            decode_mac_detail::kViterbiBranchScores.row[4 * pairs[2 * step] + pairs[2 * step + 1]]);
        const auto score_hi = aie::concat(
            score_lo.extract<32>(1), score_lo.extract<32>(0));
        const auto candidate_lo = aie::add(aie::concat(lo, lo), score_lo);
        const auto candidate_hi = aie::add(aie::concat(hi, hi), score_hi);
        const auto [maximum, choose_lo] = aie::max_cmp(candidate_hi, candidate_lo);
        const auto selected_path = aie::select(
            aie::concat(shifted_hi, shifted_hi),
            aie::concat(shifted_lo, shifted_lo), choose_lo);
        const auto metrics = aie::interleave_zip(
            maximum.extract<32>(0), maximum.extract<32>(1), 1);
        const auto paths = aie::interleave_zip(
            selected_path.extract<32>(0), selected_path.extract<32>(1), 1);
        lo = metrics.first;
        hi = metrics.second;
        path_lo = paths.first;
        path_hi = paths.second;
    }
    aie::store_v(d_metric, lo);
    aie::store_v(d_metric + 32, hi);
    aie::store_v(d_path, path_lo);
    aie::store_v(d_path + 32, path_hi);
}

static inline void decode_symbol(const uint8_t *decisions) {
    switch (d_bpsc) {
		case BPSK: decode_mac_detail::unpack_deinterleave<BPSK>(decisions, d_coded); break;
		case QPSK: decode_mac_detail::unpack_deinterleave<QPSK>(decisions, d_coded); break;
		case _16QAM: decode_mac_detail::unpack_deinterleave<_16QAM>(decisions, d_coded); break;
		case _64QAM: decode_mac_detail::unpack_deinterleave<_64QAM>(decisions, d_coded); break;
    }
    
    const uint8_t *pairs = decode_mac_detail::depuncture_symbol(
        d_coded, d_pairs, d_dbps, d_puncture_period);
    const int remaining = 16 + 8 * (int)d_phy.frame_bytes + 6 - d_steps;
    const int count = remaining < d_dbps ? remaining : d_dbps;
    int step = 0;
    AIE_LOOP_NO_UNROLL
    while (step < count) {
        const int until_trace = d_steps < 6 ? 6 - d_steps : 8 - ((d_steps - 6) & 7);
        const int available = count - step;
        const int run = available < until_trace ? available : until_trace;
        viterbi_run(pairs + 2 * step, run);
        d_steps += run;
        step += run;
        if (run == until_trace)
            traceback_byte();
    }
}

static NOINLINE void finish_frame(uint8_t *out, decode_mac_tile_metadata &metadata) {
    const int saved_bits = d_bits;
    const int saved_scrambler = d_scrambler;
    const int count = (16 + 8 * (int)d_phy.frame_bytes - d_bits) / 8;
    const auto metrics = aie::load_v<64>(d_metric);
    const uint8_t maximum = aie::reduce_max(metrics);
    const int best = 63 - aie::eq(metrics, maximum).clz();
    uint32_t crc = 0;
    AIE_LOOP_NO_UNROLL
    for (int candidate = 0; candidate < 2; ++candidate) {
        if (candidate && best == 0) break;
        uint8_t pending[10];
        int state = candidate == 0 ? 0 : best;
        int position = d_store_pos;
        AIE_LOOP_NO_UNROLL
        for (int i = count - 1; i >= 0; --i) {
            pending[i] = d_history[position][state];
            state = pending[i] >> 2;
            position = position == 0 ? d_traceback - 1 : position - 1;
        }
        d_bits = saved_bits;
        d_scrambler = saved_scrambler;
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < count; ++i) descramble_byte(pending[i]);
        crc = 0xffffffffu;
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < (int)d_phy.frame_bytes; ++i)
            crc = decode_mac_detail::crc32_byte(crc, d_psdu[i]);
        if (~crc == 0x2144df1cu) break;
    }
    const int length = (int)d_phy.frame_bytes - 4;
    if (~crc == 0x2144df1cu && metadata.frame_count < DECODE_MAC_MAX_FRAMES &&
        metadata.output_length + length <= DECODE_MAC_OUTPUT_BYTES) {
        auto &frame = metadata.frames[metadata.frame_count++];
        frame.offset = metadata.output_length;
        frame.length = length;
        frame.phy = d_phy;
        const int vector_bytes = length - length % 32;
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < vector_bytes; i += 32)
            aie::store_unaligned_v(out + metadata.output_length + i,
                                   aie::load_v<32>(d_psdu + i));
        AIE_LOOP_NO_UNROLL
        for (int i = vector_bytes; i < length; ++i)
            out[metadata.output_length + i] = d_psdu[i];
        metadata.output_length += length;
    }
    d_active = false;
}
}

extern "C" void decode_mac(
        const int8_t *decisions, const int32_t *in_meta,
        int8_t *out, int32_t *out_meta,
        int32_t N, int32_t meta_N) {
    const auto *input =
        reinterpret_cast<const frame_equalizer_tile_metadata *>(in_meta);
    auto *output = reinterpret_cast<decode_mac_tile_metadata *>(out_meta);
    output->output_length = 0; output->frame_count = 0;

    int tag = 0;
    AIE_LOOP_NO_UNROLL
    for (int offset = 0; offset < input->output_length; offset += 48) {
        while (tag < input->tag_count && input->tags[tag].offset <= offset) {
            const auto &record = input->tags[tag++];
            if (record.offset == offset && (record.frame_start & 1)) begin_frame(record);
        }
        if (!d_active) continue;
        decode_symbol(reinterpret_cast<const uint8_t *>(decisions + offset));
        if (++d_copied == d_symbols)
            finish_frame(reinterpret_cast<uint8_t *>(out), *output);
    }
}
