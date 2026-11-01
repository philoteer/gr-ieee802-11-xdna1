/** @file @brief Short-preamble detection and coarse frequency correction. */

#include <stdint.h>
#include <algorithm>

#include "aie_kernel_utils.h"
#include "aie_trig_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#define vec_factor 16
#define q_shift 15
#define autocorr_q_shift 24
#define max_tags_per_tile 7
#define max_segments_per_tile 16

#define min_gap 480
#define max_samples (540 * 80)
#define min_plateau 2
#define delay_samples 320

static constexpr int state_search = 0;
static constexpr int state_copy = 1;

/// Detection and copy state retained across input tiles.
static int d_state = state_search;
static int d_copied = 0;
static int d_plateau = 0;
static int32_t d_freq_offset_q29 = 0;
static uint16_t segment_start[max_segments_per_tile];
static uint16_t segment_len[max_segments_per_tile];
static uint16_t segment_copied[max_segments_per_tile];
static int32_t segment_freq_offset_q29[max_segments_per_tile];
static cint16 d_delay[delay_samples] = {};
alignas(32) static constexpr uint32_t phase_lane_indices[vec_factor] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

extern "C" {

/** @brief Short-preamble synchronizer: detects frame starts, estimates the
 *  frequency offset, and copies rotated samples into the output stream.
 *
 *  Threshold flags mark candidate starts; copied samples are rotated by the
 *  per-frame CFO estimate while the 320-sample delay history is carried across
 *  calls. Output metadata words 0/1 hold the produced sample count and tag
 *  count, followed by up to seven (offset, CFO) tag pairs.
 *
 *  @param in1 Complex Q0.15 input delayed by 16 samples.
 *  @param abs_in Complex Q7.24 lag-16 correlations for CFO estimation.
 *  @param threshold_met_in Threshold flags marking candidate frame starts.
 *  @param out Rotated samples preceded by the 320-sample delay history.
 *  @param out_meta Produced count, tag count, then (offset, Q3.29 CFO) pairs.
 *  @param N Number of input samples in this tile.
 *  @pre Output has room for the history prefix and a padded 16-sample tail.
 */
void sync_short_cf(cint16 *__restrict in1, cint32 *__restrict abs_in, int8_t *__restrict threshold_met_in, cint16 *__restrict out, int32_t * __restrict out_meta, int32_t N) {
    int32_t produced_out = 0;
    int32_t tag_count = 0;
    int32_t segment_count = 0;
    int32_t active_segment = -1;
    alignas(32) int32_t metadata[2 + 2 * max_tags_per_tile] = {};

    // Resume an unfinished segment from the preceding tile.
    if (d_state != state_search) {
        active_segment = 0;
        segment_start[0] = 0;
        segment_len[0] = 0;
        segment_copied[0] = d_copied;
        segment_freq_offset_q29[0] = d_freq_offset_q29;
        segment_count = 1;
    }
    
    int i = 0;
    // Identify copy segments before rotating their samples.
    while (i < N) {
        if (threshold_met_in[i]) {
            int skip_start = i;
            // Skip uniform flag blocks while a new frame cannot yet begin.
            while (d_state != state_search && i + vec_factor <= N && d_copied + vec_factor <= max_samples && d_copied + vec_factor - 1 <= min_gap) {
                aie::vector<int8_t, vec_factor> threshold_met =
                    aie::load_unaligned_v<vec_factor>(threshold_met_in + i);
                if (aie::reduce_add(threshold_met) != vec_factor) {
                    break;
                }
                d_plateau = min_plateau;
                segment_len[active_segment] += vec_factor;
                produced_out += vec_factor;
                d_copied += vec_factor;
                d_state = d_copied - max_samples;
                i += vec_factor;
            }
            if (i != skip_start) {
                continue;
            }

            if (d_plateau < min_plateau) {
                d_plateau++;
            } else if (d_state == state_search || d_copied > min_gap) {
                d_copied = 0;
                d_plateau = (d_state == state_search);
                d_freq_offset_q29 = estimate_freq_offset_q29(abs_in[i]);
                if (tag_count < max_tags_per_tile) {
                    metadata[2 + 2 * tag_count] = produced_out;
                    metadata[3 + 2 * tag_count] = d_freq_offset_q29;
                    tag_count++;
                }
                
                active_segment = segment_count;
                segment_start[segment_count] = i;
                segment_len[segment_count] = 0;
                segment_copied[segment_count] = 0;
                segment_freq_offset_q29[segment_count] = d_freq_offset_q29;
                segment_count++;

                d_state = state_copy;
            }
        } else {
            d_plateau = 0;

            int skip_start = i;
            // Skip uniform non-detection blocks without per-sample scanning.
            while ((d_state == state_search || d_copied + vec_factor <= max_samples) && i + vec_factor <= N) {
                aie::vector<int8_t, vec_factor> threshold_met =
                    aie::load_unaligned_v<vec_factor>(threshold_met_in + i);
                if (aie::reduce_add(threshold_met) != 0) {
                    break;
                }
                i += vec_factor;
                if (d_state != state_search) {
                    segment_len[active_segment] += vec_factor;
                    produced_out += vec_factor;
                    d_copied += vec_factor;
                    d_state = d_copied - max_samples;
                }
            }
            if (i != skip_start) {
                continue;
            }
        }


        if (d_state != state_search) {            
            segment_len[active_segment]++;
            produced_out++;
            d_copied++;
            d_state = d_copied - max_samples;
        }

        i++;
    }

    // Prefix output with history required by long-preamble synchronization.
    for (int j = 0; j < delay_samples; j += vec_factor) {
        copy_16(d_delay + j, out + j);
    }

    produced_out = 0;
    
    for (int s = 0; s < segment_count; s++) {
        int sample_idx = segment_start[s];
        int remaining = segment_len[s];
        uint32_t phase_step_q = radians_q29_to_phase_q(-segment_freq_offset_q29[s]);

        uint32_t phase = (uint32_t)segment_copied[s] * phase_step_q;
        aie::vector<uint32_t, vec_factor> phase_offsets_v;
        {
            aie::scoped_mode sat_mode{aie::saturation_mode::none};
            phase_offsets_v = aie::mul<acc64>(aie::load_v<vec_factor>(phase_lane_indices), phase_step_q).template to_vector<uint32_t>(0);
        }
        aie::vector<uint32_t, vec_factor> phase_v = aie::add(aie::broadcast<uint32_t, vec_factor>(phase), phase_offsets_v);
        aie::vector<cint16, vec_factor> oscillator = make_unit_q15_v(cos_estimate_MINIMAX_v(phase_v), sin_estimate_MINIMAX_v(phase_v));

        uint32_t block_phase_step_q = (uint32_t)vec_factor * phase_step_q;
        aie::vector<uint32_t, vec_factor> block_phase_step_v = aie::broadcast<uint32_t, vec_factor>(block_phase_step_q);
        // Advance all oscillator lanes by one 16-sample block.
        aie::vector<cint16, vec_factor> rotate_factor = make_unit_q15_v(
            cos_estimate_MINIMAX_v(block_phase_step_v),
            sin_estimate_MINIMAX_v(block_phase_step_v));

        aie::scoped_mode rnd_mode{aie::rounding_mode::symmetric_inf};
        aie::scoped_mode sat_mode{aie::saturation_mode::saturate};
        while (remaining >= vec_factor) {
            rotate_16(in1 + sample_idx, out + delay_samples + produced_out, oscillator);
            aie::accum<cacc48, vec_factor> next_osc = aie::mul(oscillator, rotate_factor);
            oscillator = next_osc.template to_vector<cint16>(q_shift);
            sample_idx += vec_factor;
            produced_out += vec_factor;
            remaining -= vec_factor;
        }

        // Zero-pad the final SIMD load; only remaining samples are valid.
        if (remaining > 0) {
            alignas(32) cint16 tail_in_i16[vec_factor] = {};

            for (int j = 0; j < remaining; j++) {
                tail_in_i16[j] = in1[sample_idx];
                sample_idx++;
            }

            rotate_16(tail_in_i16, out + delay_samples + produced_out, oscillator);
            produced_out += remaining;
            remaining = 0;
        }
    }

    if (produced_out > 0) {
        for (int j = 0; j < delay_samples; j += vec_factor) {
            copy_16(out + produced_out + j, d_delay + j);
        }
    }

    metadata[0] = produced_out;
    metadata[1] = tag_count;
    aie::store_v(out_meta, aie::load_v<2 + 2 * max_tags_per_tile>(metadata));
}

}
