#ifndef FRAME_EQUALIZER_METADATA_H
#define FRAME_EQUALIZER_METADATA_H

#include <stdint.h>
#include <stddef.h>

constexpr int FRAME_EQUALIZER_CSI_SIZE = 64;
constexpr int FRAME_EQUALIZER_MAX_TAGS = 16;
// The FFT kernel emits 64 metadata words per tile:
// words 0/1 are the valid sample count and tag count, followed by 31
// (offset, CFO) pairs. The preprocess kernel copies these words unchanged so
// the equalization kernel can apply the same frame resets independently.
constexpr int FRAME_EQUALIZER_INPUT_METADATA_WORDS = 64;
// A 1024-sample tile contains sixteen 64-bin OFDM symbols. Beta is stored once
// for each symbol in physical tile order, not by frame-relative symbol number.
constexpr int FRAME_EQUALIZER_SYMBOLS_PER_TILE = 16;
// Prepared metadata consists of the copied FFT metadata, one beta per symbol,
// and one center-frequency snapshot for output tag generation.
constexpr int FRAME_EQUALIZER_PREPARED_METADATA_WORDS =
    FRAME_EQUALIZER_INPUT_METADATA_WORDS + FRAME_EQUALIZER_SYMBOLS_PER_TILE + 1;

struct frame_equalizer_prepared_tile_metadata {
    // Original FFT metadata. This is sideband/control information, not FFT
    // sample data; the corrected samples travel through a separate FIFO.
    int32_t input[FRAME_EQUALIZER_INPUT_METADATA_WORDS];
    // Common pilot phase measured by preprocess for each 64-bin symbol.
    int32_t beta[FRAME_EQUALIZER_SYMBOLS_PER_TILE];
    // Snapshot of the center frequency in MHz used by output tag generation.
    int32_t center_frequency_mhz;
};

static_assert(sizeof(frame_equalizer_prepared_tile_metadata) ==
                  FRAME_EQUALIZER_PREPARED_METADATA_WORDS * sizeof(int32_t),
              "frame equalizer prepared metadata ABI changed");
static_assert(offsetof(frame_equalizer_prepared_tile_metadata, beta) ==
                  FRAME_EQUALIZER_INPUT_METADATA_WORDS * sizeof(int32_t),
              "frame equalizer beta metadata ABI changed");

struct frame_equalizer_csi {
    int32_t real; 
    int32_t imag;
};

constexpr int FRAME_EQUALIZER_CHANNEL_WORDS =
    FRAME_EQUALIZER_SYMBOLS_PER_TILE * FRAME_EQUALIZER_CSI_SIZE * 2;
constexpr int FRAME_EQUALIZER_SNR_WORDS = FRAME_EQUALIZER_SYMBOLS_PER_TILE * 2;
constexpr int FRAME_EQUALIZER_ESTIMATED_METADATA_PADDING_WORDS = 7;
constexpr int FRAME_EQUALIZER_ESTIMATED_METADATA_WORDS =
    FRAME_EQUALIZER_CHANNEL_WORDS + FRAME_EQUALIZER_PREPARED_METADATA_WORDS +
    FRAME_EQUALIZER_ESTIMATED_METADATA_PADDING_WORDS + FRAME_EQUALIZER_SNR_WORDS;

struct alignas(32) frame_equalizer_estimated_tile_metadata {
    frame_equalizer_csi
        channel[FRAME_EQUALIZER_SYMBOLS_PER_TILE][FRAME_EQUALIZER_CSI_SIZE];
    frame_equalizer_prepared_tile_metadata prepared;
    int32_t reserved[FRAME_EQUALIZER_ESTIMATED_METADATA_PADDING_WORDS];
    uint64_t snr_linear[FRAME_EQUALIZER_SYMBOLS_PER_TILE];
};

static_assert(alignof(frame_equalizer_estimated_tile_metadata) >= 32,
              "frame equalizer estimated metadata alignment changed");
static_assert(offsetof(frame_equalizer_estimated_tile_metadata, channel) % 32 == 0,
              "frame equalizer channel array alignment changed");
static_assert(offsetof(frame_equalizer_estimated_tile_metadata, snr_linear) % 32 == 0,
              "frame equalizer SNR array alignment changed");
static_assert(sizeof(frame_equalizer_estimated_tile_metadata) ==
                  FRAME_EQUALIZER_ESTIMATED_METADATA_WORDS * sizeof(int32_t),
              "frame equalizer estimated metadata ABI changed");

struct alignas(8) frame_equalizer_tag_metadata {
    // Metadata attached to one output frame/symbol. The CSI array retains the
    // existing host-visible ABI and is not part of the inter-kernel handoff.
    int32_t offset;
    uint32_t frame_start;
    uint64_t frame_bytes;
    uint64_t encoding;
    uint64_t snr_linear; // sum/difference power ratio scaled by 16
    uint32_t center_frequency_mhz;
    int32_t frequency_offset; // CFO in Q3.29 radians/sample
    int32_t beta;              // common phase in Q3.29 radians
    frame_equalizer_csi csi[FRAME_EQUALIZER_CSI_SIZE];
};

struct alignas(8) frame_equalizer_tile_metadata {
    int32_t output_length;
    int32_t tag_count;
    frame_equalizer_tag_metadata tags[FRAME_EQUALIZER_MAX_TAGS];
};

static_assert(sizeof(frame_equalizer_tag_metadata) == 560,
              "frame equalizer tag metadata ABI changed");
static_assert(offsetof(frame_equalizer_tag_metadata, snr_linear) == 24,
              "frame equalizer linear SNR ABI changed");
static_assert(offsetof(frame_equalizer_tag_metadata, center_frequency_mhz) == 32,
              "frame equalizer center frequency ABI changed");
static_assert(offsetof(frame_equalizer_tag_metadata, frequency_offset) == 36,
              "frame equalizer frequency offset ABI changed");
static_assert(offsetof(frame_equalizer_tag_metadata, csi) == 44,
              "frame equalizer CSI ABI changed");
static_assert(sizeof(frame_equalizer_tile_metadata) == 8968,
              "frame equalizer tile metadata ABI changed");

#endif
