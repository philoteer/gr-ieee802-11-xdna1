.DEFAULT_GOAL := compile
.DELETE_ON_ERROR:
# Keep nested AIE compilation serial as well.
.NOTPARALLEL:
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../toolchain.mk
AIETOOLS_DIR = /opt/vitis_aie_essentials/
export LD_LIBRARY_PATH := ${AIETOOLS_DIR}/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := ${AIETOOLS_DIR}/data
export PATH := ${AIETOOLS_DIR}/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${srcdir}/../../kernel
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all compile clean test check-grc FORCE
all: build/final.xclbin build/insts.bin
compile: all

FORCE:

build/.device: FORCE
	mkdir -p ${@D}
	@printf '%s\n' '${devicename}' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py build/.device
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/decode_mac.o: ${srcdir}/decode_mac.cc ${srcdir}/../../kernel/decode_mac.cc $(wildcard ${srcdir}/../../kernel/common/*.h) build/.device
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir build/decode_mac.o
	cd ${@D} && $(AIECC) -j1 $(AIECC_FLAGS) --xclbin-name=${@F} --npu-insts-name=insts.bin aie.mlir

build/insts.bin: build/final.xclbin
	@test -f $@

test: all
	python3 ${srcdir}/test_decode.py --supplementary

check-grc:
	mkdir -p build/grc
	grcc -o build/grc ${srcdir}/wifi_rx-original-8bit-interleaved.grc

clean:
	rm -rf build _build
