#ifndef FRAME_EQUALIZER_EQUALIZER_SELECT_H
#define FRAME_EQUALIZER_EQUALIZER_SELECT_H

#include "frame_equalizer_algorithm.h"
#include "common.h"

// Pull in exactly the statically selected estimator implementation. Adding an
// algorithm means adding one header to this directory and one branch here.
#if FRAME_EQUALIZER_ALGORITHM == FRAME_EQUALIZER_ALGORITHM_LS
#include "ls.h"
#elif FRAME_EQUALIZER_ALGORITHM == FRAME_EQUALIZER_ALGORITHM_LMS
#include "lms.h"
#elif FRAME_EQUALIZER_ALGORITHM == FRAME_EQUALIZER_ALGORITHM_COMB
#include "comb.h"
#elif FRAME_EQUALIZER_ALGORITHM == FRAME_EQUALIZER_ALGORITHM_STA
#include "sta.h"
#else
#error "Selected FRAME_EQUALIZER_ALGORITHM has no kernel implementation"
#endif

#endif
