#ifndef FRAME_EQUALIZER_EQUALIZER_LS_H
#define FRAME_EQUALIZER_EQUALIZER_LS_H

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#include "common.h"

// Least-squares channel estimator, ported from stage34/original/equalizer/ls.cc.
//
// The two repeated long training symbols give a fixed channel estimate for the
// rest of the frame; SIGNAL and payload symbols reuse it unchanged. The public
// interface is shared with the other algorithms:
//   update_channel_estimate(symbol, number, channel, snr)
//   track_channel_estimate(symbol, decisions, modulation, number, channel)
//   reset_channel_estimate(snr)
// kLong, kOccupied, and d_first_long come from common.h.

// The estimate is fixed after the LTS pair, so nothing is tracked per symbol.
static inline void update_channel_estimate(const cint32 *symbol, int number,
                                             cint32 channel[FFT_SIZE],
                                             uint64_t &snr) {
    init_channel_estimate_from_lts(symbol, number, channel, snr);
}

static inline void track_channel_estimate(const cint32 *, const int8_t *, int,
                                          int, cint32 [FFT_SIZE]) {}

// LS derives the SNR from the LTS pair, so a new frame needs no estimator state
// reset beyond the shared frame counters.
static inline void reset_channel_estimate(uint64_t &snr) { (void)snr; }

#endif
