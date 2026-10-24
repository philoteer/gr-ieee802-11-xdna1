##===- Makefile -----------------------------------------------------------===##
.DEFAULT_GOAL := compile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../kernel)
JOBS ?= 2
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir}

.PHONY: all clean compile test

BUILD_OUTPUTS := build/final.xclbin build/insts.bin

all: ${BUILD_OUTPUTS}

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/trig_utils_test.o: ${srcdir}/trig_utils_test.cc ${kernel_srcdir}/aie_trig_utils.h ${kernel_srcdir}/aie_atan_lut_q29.h ${kernel_srcdir}/aie_sin_linear_lut_q15.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir build/trig_utils_test.o
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
	python3 ${srcdir}/test_accuracy.py --xclbin build/final.xclbin --insts build/insts.bin
