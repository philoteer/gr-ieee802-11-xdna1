##===- kernel.mk ----------------------------------------------------------===##
# Shared build rules for the standalone frame-equalizer AIE kernels.
#
# A per-kernel Makefile selects the kernel to build and includes this file:
#
#     KERNEL_NAME := frame_equalizer_preprocess
#     include ../kernel.mk
#
##===----------------------------------------------------------------------===##
.DEFAULT_GOAL := compile
.DELETE_ON_ERROR:
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../../kernel)
include ${srcdir}/../../toolchain.mk
DFRAME_EQUALIZER_OUTPUT_SYMBOLS ?= 0
EQUALIZER_ALGORITHM ?= LS
SUPPORTED_EQUALIZER_ALGORITHMS := LS COMB CNN
ifeq ($(filter ${EQUALIZER_ALGORITHM},${SUPPORTED_EQUALIZER_ALGORITHMS}),)
$(error EQUALIZER_ALGORITHM must be one of: ${SUPPORTED_EQUALIZER_ALGORITHMS})
endif
CNN_DATA_UPDATER ?= NULL
CNN_WEIGHTS ?= BY_FRAME
ENGINE ?= CLASSIC
MODEL ?= MLP_16X16
ifneq ($(filter ${CNN_DATA_UPDATER},NULL MLP),${CNN_DATA_UPDATER})
$(error CNN_DATA_UPDATER must be NULL or MLP)
endif
ifneq ($(filter ${CNN_WEIGHTS},BY_FRAME BY_CHANNEL),${CNN_WEIGHTS})
$(error CNN_WEIGHTS must be BY_FRAME or BY_CHANNEL)
endif
ifneq ($(filter ${ENGINE},CLASSIC AI),${ENGINE})
$(error ENGINE must be CLASSIC or AI)
endif
ifneq ($(filter ${MODEL},MLP_16X16 MLP_16),${MODEL})
$(error MODEL must be MLP_16X16 or MLP_16)
endif
BUILD_DIR ?= build
export AIE_TILES_PER_BATCH ?= 16
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir} -I ${kernel_srcdir}/common \
	-DFRAME_EQUALIZER_OUTPUT_SYMBOLS=${DFRAME_EQUALIZER_OUTPUT_SYMBOLS} \
	-DFRAME_EQUALIZER_ALGORITHM=FRAME_EQUALIZER_ALGORITHM_${EQUALIZER_ALGORITHM} \
	-DFRAME_EQUALIZER_ENGINE=FRAME_EQUALIZER_ENGINE_${ENGINE} \
	-DFRAME_EQUALIZER_MODEL=FRAME_EQUALIZER_MODEL_${MODEL} \
	-DFRAME_EQUALIZER_CNN_WEIGHTS=FRAME_EQUALIZER_CNN_WEIGHTS_${CNN_WEIGHTS} \
	-DFRAME_EQUALIZER_CNN_DATA_UPDATER=FRAME_EQUALIZER_CNN_DATA_UPDATER_${CNN_DATA_UPDATER}

.PHONY: all clean compile FORCE

BUILD_OUTPUTS := ${BUILD_DIR}/final.xclbin ${BUILD_DIR}/insts.bin

# Each standalone kernel is built from its own aie2_memtile_caching.py and a
# symlinked copy of the kernel source; only the matching object is needed.
ifeq ($(filter ${KERNEL_NAME},frame_equalizer_preprocess frame_equalizer_channel_estimate frame_equalizer_equalize),)
$(error KERNEL_NAME is not set to a supported kernel)
endif
KERNEL_DEPS := $(wildcard ${kernel_srcdir}/common/*.h)
ifeq (${KERNEL_NAME},frame_equalizer_channel_estimate)
KERNEL_DEPS += $(wildcard ${kernel_srcdir}/channel_estimator/*.h) \
	$(wildcard ${kernel_srcdir}/channel_estimator/model/*.h) \
	$(wildcard ${kernel_srcdir}/channel_estimator/cnn_data_updater/*.h)
else ifeq (${KERNEL_NAME},frame_equalizer_equalize)
KERNEL_DEPS += $(wildcard ${kernel_srcdir}/demodulator/*.h) \
	$(wildcard ${kernel_srcdir}/demodulator/model/*.h)
endif

KERNEL_OBJECTS := ${BUILD_DIR}/${KERNEL_NAME}.o
BUILD_MODE_MARKER := ${BUILD_DIR}/${KERNEL_NAME}_build_mode.txt
BUILD_CONFIG := ${BUILD_DIR}/.build_config

all: ${BUILD_OUTPUTS} ${BUILD_MODE_MARKER}

# Rebuild when compile flags or graph geometry change, even without make clean.
FORCE:

${BUILD_CONFIG}: FORCE
	@mkdir -p ${BUILD_DIR}
	@printf 'DFRAME_EQUALIZER_OUTPUT_SYMBOLS=%s\nEQUALIZER_ALGORITHM=%s\nCNN_DATA_UPDATER=%s\nCNN_WEIGHTS=%s\nENGINE=%s\nMODEL=%s\nDEVICE=%s\nAIE_TILES_PER_BATCH=%s\n' \
		"${DFRAME_EQUALIZER_OUTPUT_SYMBOLS}" "${EQUALIZER_ALGORITHM}" \
		"${CNN_DATA_UPDATER}" "${CNN_WEIGHTS}" "${ENGINE}" "${MODEL}" \
		"${devicename}" "$(if ${AIE_TILES_PER_BATCH},${AIE_TILES_PER_BATCH},16)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

${BUILD_MODE_MARKER}: ${BUILD_OUTPUTS} ${BUILD_CONFIG}
	@cp ${BUILD_CONFIG} $@

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

${BUILD_DIR}/aie.mlir: ${srcdir}/aie2_memtile_caching.py ${BUILD_CONFIG}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

${BUILD_DIR}/${KERNEL_NAME}.o: ${srcdir}/${KERNEL_NAME}.cc ${KERNEL_DEPS} ${BUILD_CONFIG}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

${BUILD_DIR}/final.xclbin: ${BUILD_DIR}/aie.mlir ${KERNEL_OBJECTS}
	mkdir -p ${@D}
	cd ${@D} && $(AIECC) -j$(COMPILE_JOBS) $(AIECC_FLAGS) --xclbin-name=${@F} \
		--npu-insts-name=insts.bin $(abspath $<)

${BUILD_DIR}/insts.bin: ${BUILD_DIR}/final.xclbin
	@test -f $@

clean:
	rm -rf ${BUILD_DIR}

compile:
	${MAKE} clean
	${MAKE} -j${COMPILE_JOBS} all
