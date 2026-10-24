##===- Makefile -----------------------------------------------------------===##
.DEFAULT_GOAL := compile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../kernel)
JOBS ?= 2
DFRAME_EQUALIZER_OUTPUT_SYMBOLS ?= 0
EQUALIZER_ALGORITHM ?= LS
SUPPORTED_EQUALIZER_ALGORITHMS := LS LMS COMB STA CNN
ifeq ($(filter ${EQUALIZER_ALGORITHM},${SUPPORTED_EQUALIZER_ALGORITHMS}),)
$(error EQUALIZER_ALGORITHM must be one of: ${SUPPORTED_EQUALIZER_ALGORITHMS})
endif
ENGINE ?= CLASSIC
SUPPORTED_ENGINES := CLASSIC AI
ifeq ($(filter ${ENGINE},${SUPPORTED_ENGINES}),)
$(error ENGINE must be one of: ${SUPPORTED_ENGINES})
endif
MODEL ?= MLP_16X16
SUPPORTED_MODELS := MLP_16X16 MLP_16
ifeq ($(filter ${MODEL},${SUPPORTED_MODELS}),)
$(error MODEL must be one of: ${SUPPORTED_MODELS})
endif
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir} -I ${kernel_srcdir}/common \
	-DFRAME_EQUALIZER_OUTPUT_SYMBOLS=${DFRAME_EQUALIZER_OUTPUT_SYMBOLS} \
	-DFRAME_EQUALIZER_ALGORITHM=FRAME_EQUALIZER_ALGORITHM_${EQUALIZER_ALGORITHM} \
	-DFRAME_EQUALIZER_ENGINE=FRAME_EQUALIZER_ENGINE_${ENGINE} \
	-DFRAME_EQUALIZER_MODEL=FRAME_EQUALIZER_MODEL_${MODEL}

.PHONY: all clean compile test test-grc compare learned-ber test-cnn

BUILD_OUTPUTS := build/final.xclbin build/insts.bin
KERNEL_OBJECTS := build/frame_equalizer_preprocess.o build/frame_equalizer_equalize.o

# Every header the equalizer can include. Engine/model selection is
# compile-time, so globbing lets new model headers participate without
# maintaining a long prerequisite list.
EQUALIZER_HEADERS := \
	$(wildcard ${kernel_srcdir}/common/*.h) \
	$(wildcard ${kernel_srcdir}/demodulator/*.h) \
	$(wildcard ${kernel_srcdir}/demodulator/model/*.h) \
	$(wildcard ${kernel_srcdir}/channel_estimator/*.h) \
	$(wildcard ${kernel_srcdir}/channel_estimator/model/*.h)

# The symbol-comparison tests (test-symbols.grc and compare-recorded-symbols.grc)
# require DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1: that build emits one normalized
# equalized symbol per payload carrier, which is what those tests compare
# against GNU Radio. The default (CSI) build emits the channel estimate
# instead, which would produce a meaningless multi-hundred-percent EVM. Record
# the active flags in a marker so the test scripts can refuse a mismatched
# build instead of silently comparing the wrong data.
BUILD_MODE_MARKER := build/frame_equalizer_build_mode.txt
RECORDED_INPUT ?= /home/piloteer/바탕화면/fft_full.iqdata
RECORDED_HEADER ?= ${RECORDED_INPUT}.hdr
RECORDED_SAMPLES ?= 2097152
BER_BATCHES ?= 10
BER_SEED ?= 39080211
BER_OUTPUT ?=

all: ${BUILD_OUTPUTS} ${BUILD_MODE_MARKER}

${BUILD_MODE_MARKER}: ${BUILD_OUTPUTS}
	@printf 'DFRAME_EQUALIZER_OUTPUT_SYMBOLS=%s\nEQUALIZER_ALGORITHM=%s\nENGINE=%s\nMODEL=%s\nNPU2=%s\n' \
		"${DFRAME_EQUALIZER_OUTPUT_SYMBOLS}" "${EQUALIZER_ALGORITHM}" "${ENGINE}" "${MODEL}" "${NPU2}" > $@

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/frame_equalizer_preprocess.o: ${kernel_srcdir}/frame_equalizer_preprocess.cc ${kernel_srcdir}/common/frame_equalizer_metadata.h ${kernel_srcdir}/common/aie_kernel_utils.h ${kernel_srcdir}/common/aie_trig_utils.h ${kernel_srcdir}/common/aie_trig_luts.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/frame_equalizer_equalize.o: ${kernel_srcdir}/frame_equalizer_equalize.cc ${EQUALIZER_HEADERS}
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
	${MAKE} test-grc EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM}

# Generate and run the GNU Radio Companion differential test.  Keep generation
# after compile because compile intentionally recreates build/.
test-grc:
	${MAKE} DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1 EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} compile
	EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} grcc -o build ${srcdir}/test-symbols.grc
	EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} python3 build/test_symbols_grc.py

compare:
	${MAKE} DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1 EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} compile
	EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} grcc -o build ${srcdir}/compare-recorded-symbols.grc
	EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} \
		RECORDED_INPUT="${RECORDED_INPUT}" RECORDED_HEADER="${RECORDED_HEADER}" \
		RECORDED_SAMPLES=${RECORDED_SAMPLES} python3 build/compare_recorded_symbols_grc.py

# Benchmark the statically selected classic or learned engine on an identical
# seeded corpus. Run this target once per ENGINE to compare the JSON reports.
learned-ber:
	${MAKE} DFRAME_EQUALIZER_OUTPUT_SYMBOLS=0 EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} ENGINE=${ENGINE} MODEL=${MODEL} compile
	grcc -o build ${srcdir}/test-learned-ber-basic.grc
	EQUALIZER_ALGORITHM=${EQUALIZER_ALGORITHM} BER_BATCHES=${BER_BATCHES} BER_SEED=${BER_SEED} BER_OUTPUT="${BER_OUTPUT}" \
		python3 build/test_learned_ber_grc.py

test-cnn:
	${MAKE} -C ${srcdir}/cnn_parity JOBS=${JOBS} test
