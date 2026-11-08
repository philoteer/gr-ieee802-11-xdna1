##===- Makefile -----------------------------------------------------------===##
.DEFAULT_GOAL := compile
.DELETE_ON_ERROR:
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../kernel)
include ${srcdir}/../toolchain.mk
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

.PHONY: all clean compile test compare check-build-mode FORCE

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
BUILD_CONFIG := build/.frame_equalizer_build_mode

all: ${BUILD_OUTPUTS} ${BUILD_MODE_MARKER}

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

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

COMMON_HEADERS := $(wildcard ${kernel_srcdir}/common/*.h)
EQUALIZER_HEADERS := $(wildcard ${kernel_srcdir}/channel_estimator/*.h)

build/frame_equalizer_preprocess.o: ${kernel_srcdir}/frame_equalizer_preprocess.cc ${COMMON_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/frame_equalizer_equalize.o: ${kernel_srcdir}/frame_equalizer_equalize.cc ${COMMON_HEADERS} ${EQUALIZER_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir ${KERNEL_OBJECTS}
	mkdir -p ${@D}
	cd ${@D} && $(AIECC) -j$(COMPILE_JOBS) $(AIECC_FLAGS) --xclbin-name=${@F} \
		--npu-insts-name=insts.bin $(<:%=../%)

build/aie.mlir ${KERNEL_OBJECTS}: ${BUILD_CONFIG}

build/insts.bin: build/final.xclbin
	@test -f $@

clean:
	rm -rf build _build

compile:
	${MAKE} clean
	${MAKE} -j${COMPILE_JOBS} all

# The symbol tests need the symbols build mode. The configuration prerequisite
# also keeps incremental `all` builds from reusing a stale CSI-mode object.
test:
	${MAKE} DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1 EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} compile
	python3 ${srcdir}/test_symbols.py --algorithm ${EQUALIZER_ALGORITHM}

compare:
	${MAKE} DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1 EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} compile
	python3 ${srcdir}/compare_recorded_symbols.py --algorithm ${EQUALIZER_ALGORITHM}
