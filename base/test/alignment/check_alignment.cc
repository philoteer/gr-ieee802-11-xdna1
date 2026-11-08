// Compile for both aie2 and aie2p; no device execution is needed.
#include "aie_trig_luts.h"
#include "frame_equalizer_metadata.h"

#if __AIE_ARCH__ == 20
static_assert(aie::vector_decl_align == 32, "AIE2 alignment changed");
static_assert(kSinLinearLutBankBytes == 16, "AIE2 LUT bank layout changed");
#elif __AIE_ARCH__ == 21
static_assert(aie::vector_decl_align == 64, "AIE2P alignment changed");
static_assert(kSinLinearLutBankBytes == 32, "AIE2P LUT must repeat at 256 bits");
#else
#error "Compile this test for aie2 or aie2p"
#endif

static_assert(__alignof__(kAtanLutQ29) == aie::vector_decl_align,
              "atan LUT must use target vector alignment");
static_assert(__alignof__(kReciprocalSeedQ30) == aie::vector_decl_align,
              "reciprocal LUT must use target vector alignment");
static_assert(__alignof__(kSinLinearLutAb) == aie::vector_decl_align,
              "sine AB LUT must use target vector alignment");
static_assert(__alignof__(kSinLinearLutCd) == aie::vector_decl_align,
              "sine CD LUT must use target vector alignment");
static_assert(sizeof(kSinLinearLutAb) == 4 * kSinLinearLutEntries * sizeof(int16_t),
              "sine AB LUT must contain two copies of each slope/offset pair");
static_assert(sizeof(kSinLinearLutCd) == sizeof(kSinLinearLutAb),
              "sine AB and CD LUT sizes must match");

// Vector alignment must not change the host-visible metadata ABI.
static_assert(alignof(frame_equalizer_tag_metadata) == 8,
              "tag metadata alignment changed");
static_assert(alignof(frame_equalizer_tile_metadata) == 8,
              "tile metadata alignment changed");
