// Exercise the production implementation rather than maintaining a second decoder.
// INPUT_OFFSET=8 reproduces the fused graph's non-vector-aligned consumer FIFO.
#ifndef DECODE_MAC_INPUT_OFFSET
#define DECODE_MAC_INPUT_OFFSET 0
#endif

#define decode_mac decode_mac_production
#include "../../kernel/decode_mac.cc"
#undef decode_mac

extern "C" void decode_mac(const int8_t *decisions, const int32_t *in_meta,
                          int8_t *out, int32_t *out_meta,
                          int32_t N, int32_t meta_N) {
    decode_mac_production(decisions + DECODE_MAC_INPUT_OFFSET, in_meta,
                          out, out_meta, N, meta_N);
}
