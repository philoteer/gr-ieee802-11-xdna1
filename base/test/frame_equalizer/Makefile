##===- Makefile -----------------------------------------------------------===##
.DEFAULT_GOAL := compile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../kernel)
JOBS ?= 2
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
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir} \
	-DFRAME_EQUALIZER_OUTPUT_SYMBOLS=${DFRAME_EQUALIZER_OUTPUT_SYMBOLS} \
	-DFRAME_EQUALIZER_ALGORITHM=FRAME_EQUALIZER_ALGORITHM_${EQUALIZER_ALGORITHM}

.PHONY: all clean compile test compare

BUILD_OUTPUTS := build/final.xclbin build/insts.bin
KERNEL_OBJECTS := build/frame_equalizer_preprocess.o build/frame_equalizer_equalize.o

# The symbol-comparison tests (test_symbols.py, compare_recorded_symbols.py)
# require DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1: that build emits one normalized
# equalized symbol per payload carrier, which is what those tests compare
# against GNU Radio. The default (CSI) build emits the channel estimate
# instead, which would produce a meaningless multi-hundred-percent EVM. Record
# the active flags in a marker so the test scripts can refuse a mismatched
# build instead of silently comparing the wrong data.
BUILD_MODE_MARKER := build/frame_equalizer_build_mode.txt

all: ${BUILD_OUTPUTS} ${BUILD_MODE_MARKER}

${BUILD_MODE_MARKER}: ${BUILD_OUTPUTS}
	@printf 'DFRAME_EQUALIZER_OUTPUT_SYMBOLS=%s\nEQUALIZER_ALGORITHM=%s\nNPU2=%s\n' \
		"${DFRAME_EQUALIZER_OUTPUT_SYMBOLS}" "${EQUALIZER_ALGORITHM}" "${NPU2}" > $@

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/frame_equalizer_preprocess.o: ${kernel_srcdir}/frame_equalizer_preprocess.cc ${kernel_srcdir}/frame_equalizer_metadata.h ${kernel_srcdir}/aie_kernel_utils.h ${kernel_srcdir}/aie_trig_utils.h ${kernel_srcdir}/aie_atan_lut_q29.h ${kernel_srcdir}/aie_sin_linear_lut_q15.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/frame_equalizer_equalize.o: ${kernel_srcdir}/frame_equalizer_equalize.cc ${kernel_srcdir}/frame_equalizer_algorithm.h ${kernel_srcdir}/frame_equalizer_metadata.h ${kernel_srcdir}/aie_kernel_utils.h ${kernel_srcdir}/equalizer/select.h ${kernel_srcdir}/equalizer/common.h ${kernel_srcdir}/equalizer/constellation.h ${kernel_srcdir}/equalizer/ls.h ${kernel_srcdir}/equalizer/lms.h ${kernel_srcdir}/equalizer/comb.h ${kernel_srcdir}/equalizer/sta.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir ${KERNEL_OBJECTS}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/insts.bin: build/final.xclbin
	@test -f $@

clean:
	rm -rf build _build

compile:
	${MAKE} clean
	${MAKE} -j${JOBS} all

# The symbol tests need the symbols build mode. Force a clean rebuild so the
# flag change actually recompiles the kernels and refreshes the mode marker;
# `all` alone would skip recompilation and leave a stale (CSI-mode) marker.
test:
	${MAKE} DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1 EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} compile
	python3 ${srcdir}/test_symbols.py --algorithm ${EQUALIZER_ALGORITHM}

compare:
	${MAKE} DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1 EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} compile
	python3 ${srcdir}/compare_recorded_symbols.py --algorithm ${EQUALIZER_ALGORITHM}
