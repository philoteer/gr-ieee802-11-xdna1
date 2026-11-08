##===- Makefile -----------------------------------------------------------===##
.DEFAULT_GOAL := compile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../kernel)
include ${srcdir}/../toolchain.mk
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir} -I ${kernel_srcdir}/common

.PHONY: all clean compile test

BUILD_OUTPUTS := build/final.xclbin build/insts.bin

all: ${BUILD_OUTPUTS}

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/sync_short_test.o: ${srcdir}/sync_short_test.cc ${kernel_srcdir}/sync_short_cf.cc $(wildcard ${kernel_srcdir}/common/aie_*.h)
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir build/sync_short_test.o
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
