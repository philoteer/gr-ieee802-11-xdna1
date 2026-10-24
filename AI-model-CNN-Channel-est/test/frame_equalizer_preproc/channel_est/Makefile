KERNEL_NAME := frame_equalizer_channel_estimate
EQUALIZER_ALGORITHM ?= CNN
CNN_DATA_UPDATER ?= NULL
# Keep the two estimator variants available simultaneously for later benchmarks.
BUILD_DIR ?= build/$(if $(filter MLP,${CNN_DATA_UPDATER}),mlp,cnn)
include ../kernel.mk
