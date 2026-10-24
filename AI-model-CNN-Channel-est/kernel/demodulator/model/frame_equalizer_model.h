#ifndef FRAME_EQUALIZER_MODEL_H
#define FRAME_EQUALIZER_MODEL_H

// Learned-demapper selection is deliberately compile-time, mirroring the
// FRAME_EQUALIZER_ENGINE and FRAME_EQUALIZER_ALGORITHM selectors: each AIE
// image contains exactly one MLP architecture and pays no per-symbol dispatch.
//
// Every architecture lives in its own model/frame_equalizer_model_<name>.h and
// consumes the matching model/frame_equalizer_model_<name>_weights.h generated
// by utils/frame_equalizer_ai. To add one: write the model header, add a macro
// and a dispatch branch here, and list the name in the Makefiles'
// SUPPORTED_MODELS.
#define FRAME_EQUALIZER_MODEL_MLP_16X16 1
#define FRAME_EQUALIZER_MODEL_MLP_16 2

#ifndef FRAME_EQUALIZER_MODEL
#define FRAME_EQUALIZER_MODEL FRAME_EQUALIZER_MODEL_MLP_16X16
#endif

#if FRAME_EQUALIZER_MODEL == FRAME_EQUALIZER_MODEL_MLP_16X16
#include "frame_equalizer_model_mlp_16x16.h"
#elif FRAME_EQUALIZER_MODEL == FRAME_EQUALIZER_MODEL_MLP_16
#include "frame_equalizer_model_mlp_16.h"
#else
#error "Selected FRAME_EQUALIZER_MODEL has no kernel implementation"
#endif

#endif
