//===- sync_long_cf.cc ------------------------------------------*- C++ -*-===//
// IEEE 802.11 long preamble synchronizer kernel.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <algorithm>

#include "aie_kernel_utils.h"
#include "aie_trig_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#define vec_factor 16
#define q_shift 15
#define delay_samples 320
#define sync_length 320
#define long_length 64
#define correlation_shift 8
#define correlation_power_shift 24

static constexpr int state_wait = 0;
static constexpr int state_sync = 1;
static constexpr int state_copy = 2;

// GNU Radio's fir_filter reverses LONG in set_taps(). Store that effective
// order directly, quantized as signed Q2.14.
alignas(32) static constexpr cint16 long_taps[long_length] = {
    { 22721,      0 }, {  -745,  17496 }, {  5780,  16163 }, { 14080, -12039 },
    {  3070,  -4055 }, {  8698,  12753 }, {-16741,   8023 }, { -5572,  15439 },
    { 14184,   3765 }, {  7756,   -593 }, {   144,  16723 }, {-19893,   6889 },
    {  3559,   8511 }, {  8531,   2173 }, { -3269, -23362 }, { 17339,    595 },
    {  9088,   9088 }, {  5369, -14300 }, { -8318,  -5715 }, {-19087,  -9485 },
    { 11955, -13430 }, { 10114,  -2053 }, { -8770, -11819 }, { -8210,   3170 },
    { -5095,  21941 }, {-17724,   2408 }, {-18514,   2982 }, { 10917,  10766 },
    {  -408,  -7820 }, {-13361, -16741 }, { 13337, -15394 }, {  1786, -14192 },
    {-22721,      0 }, {  1786,  14192 }, { 13337,  15394 }, {-13361,  16741 },
    {  -408,   7820 }, { 10917, -10766 }, {-18514,  -2982 }, {-17724,  -2408 },
    { -5095, -21941 }, { -8210,  -3170 }, { -8770,  11819 }, { 10114,   2053 },
    { 11955,  13430 }, {-19087,   9485 }, { -8318,   5715 }, {  5369,  14300 },
    {  9088,  -9088 }, { 17339,   -595 }, { -3269,  23362 }, {  8531,  -2173 },
    {  3559,  -8511 }, {-19893,  -6889 }, {   144, -16723 }, {  7756,    593 },
    { 14184,  -3765 }, { -5572, -15439 }, {-16741,  -8023 }, {  8698, -12753 },
    {  3070,   4055 }, { 14080,  12039 }, {  5780, -16163 }, {  -745, -17496 }
};

alignas(32) static constexpr uint32_t phase_lane_indices[vec_factor] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

// Persistent state: one AIE tile processes a continuous stream across calls.
static int d_state = state_wait;
// Number of current-stream samples accumulated while searching for the preamble.
static int d_available = 0;
// Next candidate offset at which to correlate the 64-sample LONG sequence.
static int d_next_correlation = 0;
// Position in the delayed input used by COPY; unlike `produced`, this includes skipped samples.
static int d_copy_offset = 0;
// First long-preamble correlation peak, hence the start of the useful frame data.
static int d_frame_start = 0;
// Number of samples copied for this frame, used to pad a frame boundary to 64 samples.
static int d_count = 0;
// CFO supplied by sync_short and the refined CFO measured from the long preamble.
static int32_t d_freq_offset_short_q29 = 0;
static int32_t d_freq_offset_q29 = 0;
// A refined-CFO tag is emitted with the first output sample of each frame.
static bool d_tag_pending = false;
// Correlations are buffered during SYNC and ranked together in finish_sync().
alignas(32) static cint32 d_correlations[sync_length] = {};
alignas(32) static int32_t d_power_hi[sync_length] = {};
// The four strongest correlation results are retained to locate two peaks one LONG apart.
static int32_t d_top_real[4] = {};
static int32_t d_top_imag[4] = {};
static int d_top_offset[4] = {};

/** @brief Resets the synchronizer for a new frame and records its coarse CFO.
 *
 *  @param freq_offset_short_q29 Short-preamble CFO in Q3.29 radians/sample.
 */
static inline void start_sync(int32_t freq_offset_short_q29) {
    d_state = state_sync;
    d_available = 0;
    d_next_correlation = 0;
    d_freq_offset_short_q29 = freq_offset_short_q29;
    d_tag_pending = false;
    for (int i = 0; i < 4; i++) {
        d_top_real[i] = 0;
        d_top_imag[i] = 0;
        d_top_offset[i] = 0;
    }
}

/** @brief Computes 16 adjacent 64-sample LONG correlations into d_correlations.
 *
 *  Each of the 16 lanes accumulates one offset against the 64-tap sequence.
 *
 *  @param in Input window containing the correlation samples.
 *  @param offset Destination offset in d_correlations, a multiple of 16.
 */
static inline void correlate_16(const cint16 *__restrict in, int offset) {
    // Each lane computes one adjacent 64-sample correlation. A 16-tap block
    // needs 16 output samples plus 15 samples of overlap from the input.
    auto correlations = aie::sliding_mul<vec_factor, vec_factor, 1, 1, 1, cacc64>(
        aie::load_v<vec_factor>(long_taps), 0,
        aie::load_unaligned_v<2 * vec_factor>(in), 0);

    AIE_LOOP_UNROLL_FULL
    for (int k = vec_factor; k < long_length; k += vec_factor) {
        correlations = aie::sliding_mac<vec_factor, vec_factor>(
            correlations,
            aie::load_v<vec_factor>(long_taps + k), 0,
            aie::load_unaligned_v<2 * vec_factor>(in + k), 0);
    }

    aie::store_v(d_correlations + offset,
                 correlations.template to_vector<cint32>(correlation_shift));
}

/** @brief Ranks the buffered correlations and records the four strongest offsets.
 *
 *  Shifts each |correlation|^2 by correlation_power_shift for the vector scan,
 *  then resolves equal buckets with the exact 64-bit power.
 */
static inline void select_top_correlations() {
    // REAL-DATA PROFILE (2026-08-03): 63.4% of SYNC and 48.8% of total core
    // cycles. Within this function, power generation was 5.0%.
    // Q0.15 input, Q2.14 taps, 64 terms, and correlation_shift give a power
    // below 2^55. Shift 24 is the smallest value that cannot saturate int32.
    // Exact power resolves the remaining equal buckets.
    for (int i = 0; i < sync_length; i+= vec_factor) {
        const auto correlations = aie::load_unaligned_v<vec_factor> (d_correlations + i);
        aie::store_v(d_power_hi + i,
                     aie::abs_square<int32>(correlations, correlation_power_shift));
    }

    for (int rank = 0; rank < 4; rank++) {
        // HOTTEST PART OF PEAK SELECTION: the four full reduce_max scans and
        // candidate bookkeeping consumed 59.2% of select_top_correlations().
        int32_t max_power_hi = -1;
        int candidate_vectors[sync_length / vec_factor];
        int candidate_count = 0;
        for (int i = 0; i < sync_length; i += vec_factor) {
            const int32_t vector_max =
                aie::reduce_max(aie::load_v<vec_factor>(d_power_hi + i));
            if (vector_max > max_power_hi) {
                max_power_hi = vector_max;
                candidate_count = 0;
                candidate_vectors[candidate_count++] = i;
            } else if (vector_max == max_power_hi) {
                candidate_vectors[candidate_count++] = i;
            }
        }

        // Exact-power tie resolution consumed another 32.2% of peak selection.
        // Its cost varies with the number of equal shifted-power buckets.
        int max_offset = 0;
        int64_t max_power = -1;
        for (int candidate = 0; candidate < candidate_count; candidate++) {
            const int vector_offset = candidate_vectors[candidate];
            for (int lane = 0; lane < vec_factor; lane++) {
                const int offset = vector_offset + lane;
                if (d_power_hi[offset] != max_power_hi) {
                    continue;
                }
                const int64_t power = (int64_t)d_correlations[offset].real * d_correlations[offset].real + (int64_t)d_correlations[offset].imag *d_correlations[offset].imag; 
                if (power > max_power) {
                    max_power = power;
                    max_offset = offset;
                }
            }
        }

        d_top_real[rank] = d_correlations[max_offset].real;
        d_top_imag[rank] = d_correlations[max_offset].imag;
        d_top_offset[rank] = max_offset;
        d_power_hi[max_offset] = -1;
    }
}

/** @brief Locks the frame start from the top correlations and enters COPY.
 *
 *  Searches the ranked offsets for a pair 63..65 samples apart, estimates the
 *  residual CFO from the phase change across that separation, then sets the
 *  copy offset/count and raises the pending tag for the next state.
 */
static inline void finish_sync() {
    select_top_correlations();
    // A valid long preamble gives two large correlation peaks 64 samples apart.
    // Nearby (63/65) pairs tolerate a one-sample timing discrepancy.
    d_frame_start = sync_length;
    bool adjacent_match = false;

    for (int i = 0; i < 3 && !adjacent_match; i++) {
        for (int k = i + 1; k < 4; k++) {
            const int diff = std::abs(d_top_offset[i] - d_top_offset[k]);
            if (diff != 63 && diff != 64 && diff != 65) {
                continue;
            }

            const int first_idx = d_top_offset[i] < d_top_offset[k] ? i : k;
            const int second_idx = first_idx == i ? k : i;
            // Downscale both peaks together before forming their product, preventing
            // the 32-bit atan2 input from overflowing without changing their phase.
            int32_t max_component = std::max(
                std::max(std::abs(d_top_real[first_idx]), std::abs(d_top_imag[first_idx])),
                std::max(std::abs(d_top_real[second_idx]), std::abs(d_top_imag[second_idx])));
            int phase_shift = 0;
            if (max_component > 16383)
                phase_shift = 18 - __builtin_clz(static_cast<uint32_t>(max_component));
            const int32_t first_real = d_top_real[first_idx] >> phase_shift;
            const int32_t first_imag = d_top_imag[first_idx] >> phase_shift;
            const int32_t second_real = d_top_real[second_idx] >> phase_shift;
            const int32_t second_imag = d_top_imag[second_idx] >> phase_shift;
            const int64_t product_real =
                (int64_t)first_real * second_real + (int64_t)first_imag * second_imag;
            const int64_t product_imag =
                (int64_t)first_imag * second_real - (int64_t)first_real * second_imag;

            d_frame_start = std::min(d_top_offset[i], d_top_offset[k]);
            // The phase change over the peak separation is the residual CFO.
            d_freq_offset_q29 = atan2_estimate_CORDIC_q29(
                (int32_t)product_imag, (int32_t)product_real) / diff;
            if (diff == 64) {
                adjacent_match = true;
                break;
            }
        }
    }

    // GNU Radio enters COPY after consuming SYNC_LENGTH current-stream
    // samples. Its delayed input therefore starts at offset zero here.
    d_copy_offset = 0;
    d_count = 0;
    d_tag_pending = true;
    d_state = state_copy;
}

static inline aie::vector<cint16, vec_factor> multiply_q15_v(
    aie::vector<cint16, vec_factor> a,
    aie::vector<cint16, vec_factor> b) {
    aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
    aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
    return aie::mul(a, b).template to_vector<cint16>(q_shift);
}

/** @brief Advances a per-lane oscillator by a number of samples.
 *
 *  Whole 16-sample blocks use block_rotate_factor_v; leftover samples use
 *  rotate_factor_v.
 *
 *  @param oscillators Oscillator lanes, advanced in place.
 *  @param rotate_factor_v Per-sample rotation factor.
 *  @param block_rotate_factor_v 16-sample rotation factor.
 *  @param samples Number of samples to advance.
 */
static inline void advance_oscillators(
    aie::vector<cint16, vec_factor>& oscillators,
    aie::vector<cint16, vec_factor> rotate_factor_v,
    aie::vector<cint16, vec_factor> block_rotate_factor_v,
    int samples) {
    while (samples >= vec_factor) {
        oscillators = multiply_q15_v(oscillators, block_rotate_factor_v);
        samples -= vec_factor;
    }
    while (samples-- > 0) {
        oscillators = multiply_q15_v(oscillators, rotate_factor_v);
    }
}

extern "C" {

/** @brief Long-preamble synchronizer: correlates the 64-sample LONG sequence,
 *  locks the frame start from the correlation peaks, refines the CFO, and
 *  copies the frame samples rotated by the refined estimate while dropping
 *  payload GI intervals.
 *
 *  The first 320 input samples are the delay history, and stream-relative
 *  indexes in in_meta start after that prefix. Output metadata words 0/1 are
 *  the produced sample count and tag count, followed by (offset, CFO) pairs.
 *
 *  @param _in Complex input samples with the 320-sample delay history first.
 *  @param in_meta [valid samples, tag count, (offset, short CFO), ...].
 *  @param out Rotated output samples; payload GI intervals are dropped.
 *  @param out_meta Produced count, tag count, then (offset, short - refined CFO).
 *  @param N Unused; the valid sample count comes from in_meta[0].
 *  @param meta_N Unused.
 */
void sync_long_cf(cint16 *__restrict _in, int32_t *__restrict in_meta,
                  cint16 *__restrict out, int32_t *__restrict out_meta,
                  int32_t N, int32_t meta_N) {
    // The first 320 input samples are the delayed history; indexes in stream
    // metadata are relative to the current stream immediately after that prefix.
    cint16 *__restrict in_current = _in + delay_samples;
    // Metadata format: [valid samples, tag count, tag offset, short CFO, ...].
    const int input_count = in_meta[0];
    const int input_tag_count = in_meta[1];
    int produced = 0;
    int output_tag_count = 0;
    int cursor = 0;

    // REAL-DATA PROFILE (2026-08-03): hardware cycle counters over 188,000 tile
    // calls from sync_short_out_short.iqdata, including its recorded wifi_start
    // tags. Average total was 17.6k core cycles/tile: WAIT <0.001%, SYNC 77.0%,
    // COPY 20.2%, and tag/state-loop overhead 2.9%. Compared with the immediately
    // preceding profile, total cycles fell 35.0% and COPY fell 73.0%; oscillator
    // setup fell 80.7% and output/tail work fell 45.6% at the matched checkpoint.
    //
    // Reproduction method:
    // 1. Generate test/sync_long/sync-long-cui-npu_only_from_file.grc, then
    //    temporarily route generated sync_long_test.py through
    //    sync_long_test_epy_block_0.py instead of mlir_aie_cpp_sync_long_test.
    //    Set its file-meta source to /home/piloteer/xdna_phy/test_data/
    //    sync_short_out_short.iqdata and the matching .hdr file. This runs only
    //    sync_long on the NPU and matches the previous profile's input.
    // 2. Temporarily bracket the function, each state, and the annotated inner
    //    sections with aie::tile::current().cycles(); accumulate uint64_t deltas.
    // 3. At each kernel return, write total/state/section deltas as int32 values
    //    into the last 16 packed-int32 words of each 1024-sample output tile.
    //    Layout: [magic, total, wait, sync, copy, correlation, select, power,
    //    rank_scan, tie, finish_other, copy_setup, copy_skip, copy_output,
    //    input_tag_count, produced]. The magic value used was 0x53594e43.
    // 4. In sync_long_test_epy_block_0.py, read those 16 words from each of the
    //    four tiles after DefaultNPURuntime.run(), validate a magic word, and sum
    //    counters in int64 arrays. The percentages here use the 188,000-tile sum.
    // 5. Build serially with `make JOBS=1 all`; generate with `grcc
    //    sync-long-cui-npu_only_from_file.grc`; apply the temporary Python routing
    //    above; then collect for 30 seconds with
    //    `timeout --signal=INT --kill-after=5s 30s python3 -u sync_long_test.py`.
    // Remove the counters and output-tail writes before the production rebuild.

    // Tags divide this tile into segments. A tag marks the start of a new frame,
    // so process preceding samples before resetting the synchronizer for it.
    for (int tag_idx = 0; tag_idx <= input_tag_count; tag_idx++) {
        int segment_end = input_count;
        int32_t next_short_freq_q29 = 0;
        bool has_tag = false;
        if (tag_idx < input_tag_count) {
            segment_end = in_meta[2 + 2 * tag_idx];
            next_short_freq_q29 = in_meta[3 + 2 * tag_idx];
            has_tag = true;
        }

        while (cursor < segment_end) {
            if (d_state == state_wait) {
                // WAIT was effectively 0% on this recording. Its only work is
                // discarding the segment; this result depends on tag density.
                // Ignore samples until sync_short supplies a candidate frame tag.
                cursor = segment_end;
                break;
            }

            if (d_state == state_sync) {
                // HOTTEST STATE: SYNC consumed 77.0% of total core time,
                // averaging 13.5k
                // cycles/tile on the recorded stream.
                const int old_available = d_available;
                const int new_available = old_available + segment_end - cursor;
                // A 16-window batch is usable only after all 79 samples have arrived.
                // Batched correlation consumed 31.0% of SYNC (23.8% total,
                // 4.2k cycles/tile). HOTTEST PART OF SYNC is peak selection at
                // 63.4% of SYNC; the remainder of finish_sync() was 4.4%.
                const int available_correlations = std::min(
                    sync_length, new_available - long_length + 1);
                while (d_next_correlation + vec_factor <= available_correlations) {
                    const int local_start = cursor - old_available + d_next_correlation;
                    correlate_16(in_current + local_start, d_next_correlation);
                    d_next_correlation += vec_factor;
                }
                d_available = new_available;
                cursor = segment_end;
                if (d_next_correlation == sync_length) {
                    finish_sync();
                    // COPY may consume samples retained in the 320-sample prefix.
                    cursor = segment_end;
                }
            }

            if (d_state == state_copy) {
                // COPY consumed 20.2% of total core time, averaging 3.5k
                // cycles/tile on the recorded stream.
                // The delayed prefix makes the 320 synchronization samples precede
                // copy offset zero, so no source sample is available until then.
                const int new_received = d_available + segment_end - cursor;
                const int source_available = std::max(new_received - sync_length, 0);
                // HOTTEST PART OF COPY: phase and oscillator-lane setup
                // through the rounding-mode setup below consumed 62.5% of COPY
                // (12.6% total, 2.2k cycles/tile).
                const uint32_t phase_step = radians_q29_to_phase_q(d_freq_offset_q29);

                // Keep the oscillators for 16 consecutive samples in SIMD lanes.
                // A full output block rotates all samples and advances all phases
                // by rotate_factor^16 with one vector multiply each.
                aie::vector<uint32_t, vec_factor> phase_v;
                {
                    aie::scoped_mode sat_mode{aie::saturation_mode::none};
                    phase_v = aie::mul<acc64>(
                        aie::load_v<vec_factor>(phase_lane_indices), phase_step)
                                          .template to_vector<uint32_t>(0);
                }
                const uint32_t phase = (uint32_t)d_copy_offset * phase_step;
                phase_v = aie::add(
                    aie::broadcast<uint32_t, vec_factor>(phase), phase_v);
                aie::vector<cint16, vec_factor> oscillators = make_unit_q15_v(
                    cos_estimate_MINIMAX_v(phase_v), sin_estimate_MINIMAX_v(phase_v));

                const aie::vector<cint16, vec_factor> rotate_factor_v =
                    make_unit_q15_v(
                        cos_estimate_MINIMAX_v(
                            aie::broadcast<uint32_t, vec_factor>(phase_step)),
                        sin_estimate_MINIMAX_v(
                            aie::broadcast<uint32_t, vec_factor>(phase_step)));
                const uint32_t block_phase_step = (uint32_t)vec_factor * phase_step;
                const aie::vector<cint16, vec_factor> block_rotate_factor_v =
                    make_unit_q15_v(
                        cos_estimate_MINIMAX_v(
                            aie::broadcast<uint32_t, vec_factor>(block_phase_step)),
                        sin_estimate_MINIMAX_v(
                            aie::broadcast<uint32_t, vec_factor>(block_phase_step)));
                aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
                aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
                while (d_copy_offset < source_available) {
                    if (d_copy_offset < d_frame_start) {
                        // Discard samples before the detected long-preamble peak,
                        // but advance the oscillator to preserve absolute phase.
                        const int skip = std::min(d_frame_start - d_copy_offset,
                                                  source_available - d_copy_offset);
                        advance_oscillators(oscillators, rotate_factor_v,
                                            block_rotate_factor_v, skip);
                        d_copy_offset += skip;
                        continue;
                    }

                    const int relative = d_copy_offset - d_frame_start;
                    if (relative >= 128) {
                        // After the two 64-sample LONG symbols, each OFDM symbol is
                        // 16-sample GI plus 64 useful samples. Drop only the GI.
                        const int symbol_position = (relative - 128) % 80;
                        if (symbol_position < 16) {
                            const int skip = std::min(16 - symbol_position,
                                                      source_available - d_copy_offset);
                            advance_oscillators(oscillators, rotate_factor_v,
                                                block_rotate_factor_v, skip);
                            d_copy_offset += skip;
                            continue;
                        }
                    }

                    int run = source_available - d_copy_offset;
                    if (relative < 128) {
                        run = std::min(run, 128 - relative);
                    } else {
                        run = std::min(run, 80 - ((relative - 128) % 80));
                    }
                    if (d_tag_pending && run > 0) {
                        // Downstream uses the difference between short and refined
                        // CFO estimates to correct the frequency estimate.
                        out_meta[2 + 2 * output_tag_count] = produced;
                        out_meta[3 + 2 * output_tag_count] =
                            d_freq_offset_short_q29 - d_freq_offset_q29;
                        output_tag_count++;
                        d_tag_pending = false;
                    }

                    // Convert a stream-relative delayed-buffer offset to this tile's
                    // input pointer. `cursor - d_available` is the segment origin.
                    const int local_start = cursor - d_available + d_copy_offset;
                    int processed = 0;
                    // Output rotation/copy consumed 11.1% of COPY; oscillator
                    // advancement while skipping samples consumed another 9.8%.
                    while (processed + vec_factor <= run) {
                        rotate_16(in_current + local_start + processed,
                                  out + produced + processed, oscillators);
                        oscillators =
                            multiply_q15_v(oscillators, block_rotate_factor_v);
                        processed += vec_factor;
                    }
                    if (processed < run) {
                        alignas(32) cint16 input_lanes[vec_factor] = {};
                        alignas(32) cint16 output_lanes[vec_factor];
                        for (int i = processed; i < run; i++) {
                            input_lanes[i - processed] = in_current[local_start + i];
                        }
                        const auto rotated = multiply_q15_v(
                            aie::load_v<vec_factor>(input_lanes), oscillators);
                        aie::store_v(output_lanes, rotated);
                        for (int i = processed; i < run; i++) {
                            out[produced + i] = output_lanes[i - processed];
                        }
                        advance_oscillators(oscillators, rotate_factor_v,
                                            block_rotate_factor_v, run - processed);
                    }
                    d_copy_offset += run;
                    produced += run;
                    d_count += run;
                }
                d_available = new_received;
                cursor = segment_end;
            }
        }

        if (has_tag) {
            if (d_state == state_copy) {
                // Preserve GNU Radio's 64-sample output alignment between frames.
                const int padding = (64 - (d_count & 63)) & 63;
                const aie::vector<cint16, vec_factor> zeros =
                    aie::zeros<cint16, vec_factor>();
                for (int i = 0; i < padding; i += vec_factor) {
                    aie::store_unaligned_v(out + produced + i, zeros);
                }
                // Stores round padding up to 16 samples; retain the exact logical
                // count so subsequent output overwrites any excess zero samples.
                produced += padding;
            }
            // This tag begins the next frame; retain its coarse CFO for output metadata.
            start_sync(next_short_freq_q29);
        }
    }

    out_meta[0] = produced;
    out_meta[1] = output_tag_count;
}

}
