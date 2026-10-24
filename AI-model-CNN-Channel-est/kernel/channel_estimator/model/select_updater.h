#ifndef FRAME_EQUALIZER_MLP_WEIGHTS_SELECT_H
#define FRAME_EQUALIZER_MLP_WEIGHTS_SELECT_H

#include "select.h"

// The updater is trained against its CNN baseline. Use the same selection for
// both components so a build cannot combine different dataset split policies.
#if FRAME_EQUALIZER_CNN_WEIGHTS == FRAME_EQUALIZER_CNN_WEIGHTS_BY_FRAME
#include "frame_equalizer_channel_updater_mlp_weights_by_frame.h"
#elif FRAME_EQUALIZER_CNN_WEIGHTS == FRAME_EQUALIZER_CNN_WEIGHTS_BY_CHANNEL
#include "frame_equalizer_channel_updater_mlp_weights_by_channel.h"
#elif FRAME_EQUALIZER_CNN_WEIGHTS == FRAME_EQUALIZER_CNN_WEIGHTS_BUS_TO_ROOM
#include "frame_equalizer_channel_updater_mlp_weights_bus_to_room.h"
#elif FRAME_EQUALIZER_CNN_WEIGHTS == FRAME_EQUALIZER_CNN_WEIGHTS_ROOM_TO_BUS
#include "frame_equalizer_channel_updater_mlp_weights_room_to_bus.h"
#else
#error "Unsupported MLP weight set"
#endif

#endif
