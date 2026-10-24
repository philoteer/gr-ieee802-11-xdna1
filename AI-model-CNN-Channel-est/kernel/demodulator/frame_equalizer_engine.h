#ifndef FRAME_EQUALIZER_ENGINE_H
#define FRAME_EQUALIZER_ENGINE_H

// Engine selection is deliberately compile-time, mirroring the
// FRAME_EQUALIZER_ALGORITHM selector for the estimator: each AIE image
// contains exactly one PHY implementation and pays no per-symbol dispatch cost.
#define FRAME_EQUALIZER_ENGINE_CLASSIC 0
#define FRAME_EQUALIZER_ENGINE_AI 1

#ifndef FRAME_EQUALIZER_ENGINE
#define FRAME_EQUALIZER_ENGINE FRAME_EQUALIZER_ENGINE_CLASSIC
#endif

#if FRAME_EQUALIZER_ENGINE == FRAME_EQUALIZER_ENGINE_CLASSIC
#include "frame_equalizer_engine_classic.h"
#elif FRAME_EQUALIZER_ENGINE == FRAME_EQUALIZER_ENGINE_AI
#include "frame_equalizer_engine_ai.h"
#else
#error "Selected FRAME_EQUALIZER_ENGINE has no kernel implementation"
#endif

#endif
