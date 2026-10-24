.DEFAULT_GOAL := compile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../../kernel)
JOBS ?= 2

ifneq ($(filter 1 2,$(JOBS)),$(JOBS))
$(error JOBS must be 1 or 2)
endif

export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir} -I ${kernel_srcdir}/common
BUILD_OUTPUTS := build/final.xclbin build/insts.bin
CNN_HEADERS := ${kernel_srcdir}/common/aie_kernel_utils.h \
	${kernel_srcdir}/common/frame_equalizer_algorithm.h \
	${kernel_srcdir}/channel_estimator/common.h \
	${kernel_srcdir}/channel_estimator/select.h \
	${kernel_srcdir}/channel_estimator/cnn.h \
	${kernel_srcdir}/channel_estimator/cnn_common.h \
	${kernel_srcdir}/channel_estimator/model/frame_equalizer_channel_ai_weights.h

.PHONY: all clean compile test

all: ${BUILD_OUTPUTS}

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/cnn_parity_test.o: ${srcdir}/cnn_parity_test.cc ${CNN_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir build/cnn_parity_test.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/insts.bin: build/final.xclbin
	@test -f $@

clean:
	rm -rf build _build __pycache__

compile:
	${MAKE} clean
	${MAKE} -j${JOBS} all

test: all
	python3 ${srcdir}/test_parity.py --xclbin build/final.xclbin --insts build/insts.bin
