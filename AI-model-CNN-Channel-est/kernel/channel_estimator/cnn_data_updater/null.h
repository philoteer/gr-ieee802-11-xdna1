#ifndef FRAME_EQUALIZER_CNN_DATA_UPDATER_NULL_H
#define FRAME_EQUALIZER_CNN_DATA_UPDATER_NULL_H

#include <stdint.h>

#include "../common.h"

namespace {

struct CnnDataUpdater {
    static inline void initialize(const cint32 *) {}
    static inline void forward(const cint32 *, int, uint64_t,
                               cint32 [FFT_SIZE]) {}
};

} // namespace

#endif
