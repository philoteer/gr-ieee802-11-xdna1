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
SUPPORTED_EQUALIZER_ALGORITHMS := LS LMS COMB STA
ifeq ($(filter ${EQUALIZER_ALGORITHM},${SUPPORTED_EQUALIZER_ALGORITHMS}),)
$(error EQUALIZER_ALGORITHM must be one of: ${SUPPORTED_EQUALIZER_ALGORITHMS})
endif
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir} -I ${kernel_srcdir}/common \
	-DFRAME_EQUALIZER_OUTPUT_SYMBOLS=${DFRAME_EQUALIZER_OUTPUT_SYMBOLS} \
	-DFRAME_EQUALIZER_ALGORITHM=FRAME_EQUALIZER_ALGORITHM_${EQUALIZER_ALGORITHM}

.PHONY: all clean compile check-build-mode FORCE

BUILD_OUTPUTS := build/final.xclbin build/insts.bin

# Each standalone kernel is built from its own aie2_memtile_caching.py and a
# symlinked copy of the kernel source; only the matching object is needed.
ifeq (${KERNEL_NAME},frame_equalizer_preprocess)
KERNEL_DEPS := $(wildcard ${kernel_srcdir}/common/*.h)
else ifeq (${KERNEL_NAME},frame_equalizer_equalize)
KERNEL_DEPS := $(wildcard ${kernel_srcdir}/common/*.h ${kernel_srcdir}/channel_estimator/*.h)
else
$(error KERNEL_NAME is not set to a supported kernel)
endif

KERNEL_OBJECTS := build/${KERNEL_NAME}.o
BUILD_MODE_MARKER := build/${KERNEL_NAME}_build_mode.txt
BUILD_CONFIG := build/.build_config

all: ${BUILD_OUTPUTS} ${BUILD_MODE_MARKER}

# Rebuild when compile flags or graph geometry change, even without make clean.
check-build-mode: ${BUILD_CONFIG}

FORCE:

${BUILD_CONFIG}: FORCE
	@mkdir -p ${@D}
	@printf 'DFRAME_EQUALIZER_OUTPUT_SYMBOLS=%s\nEQUALIZER_ALGORITHM=%s\nNPU2=%s\nDEVICE=%s\nAIE_TILES_PER_BATCH=%s\n' \
		"${DFRAME_EQUALIZER_OUTPUT_SYMBOLS}" "${EQUALIZER_ALGORITHM}" "$(if $(filter npu2,${devicename}),1,0)" \
		"${devicename}" "$(if ${AIE_TILES_PER_BATCH},${AIE_TILES_PER_BATCH},16)" > $@.tmp
	@set -e; if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

${BUILD_MODE_MARKER}: ${BUILD_OUTPUTS} ${BUILD_CONFIG}
	@cp ${BUILD_CONFIG} $@

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py ${BUILD_CONFIG}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/${KERNEL_NAME}.o: ${srcdir}/${KERNEL_NAME}.cc ${KERNEL_DEPS} ${BUILD_CONFIG}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir ${KERNEL_OBJECTS}
	mkdir -p ${@D}
	cd ${@D} && $(AIECC) -j$(COMPILE_JOBS) $(AIECC_FLAGS) --xclbin-name=${@F} \
		--npu-insts-name=insts.bin $(<:%=../%)

build/insts.bin: build/final.xclbin
	@test -f $@

clean:
	rm -rf build _build

compile:
	${MAKE} clean
	${MAKE} -j${COMPILE_JOBS} all
