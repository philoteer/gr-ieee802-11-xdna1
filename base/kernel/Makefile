##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
.DEFAULT_GOAL := compile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
JOBS ?= 2
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH) 
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

# mlir-aie/programming_examples/makefile-common
#AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc))/../)
AIETOOLS_DIR = /opt/vitis_aie_essentials/
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib
MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}
CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${srcdir}
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${MLIR_AIE_DIR}/include
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${MLIR_AIE_DIR}/include

TEST_POWERSHELL := $(shell command -v powershell.exe >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(TEST_POWERSHELL),yes)
	powershell = powershell.exe
	getwslpath = wslpath -w
else
	powershell =
	getwslpath = echo
endif

#################
.PHONY: all clean compile

KERNEL_OBJECTS := \
	build/autocorr_cf.o \
	build/magsq_avg_cf.o \
	build/threshold_cf.o \
	build/sync_short_cf.o \
	build/sync_long_cf.o \
	build/fft_cf.o \
	build/frame_equalizer_preprocess.o \
	build/frame_equalizer_equalize.o

BUILD_OUTPUTS := build/final.xclbin build/insts.bin

all: ${BUILD_OUTPUTS}

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

# Compile autocorr_cf.cc to autocorr_cf.o
build/autocorr_cf.o: ${srcdir}/autocorr_cf.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

# Compile magsq_avg_cf.cc to magsq_avg_cf.o
build/magsq_avg_cf.o: ${srcdir}/magsq_avg_cf.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/threshold_cf.o: ${srcdir}/threshold_cf.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/sync_short_cf.o: ${srcdir}/sync_short_cf.cc ${srcdir}/aie_trig_utils.h ${srcdir}/aie_atan_lut_q29.h ${srcdir}/aie_sin_linear_lut_q15.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/sync_long_cf.o: ${srcdir}/sync_long_cf.cc ${srcdir}/aie_trig_utils.h ${srcdir}/aie_atan_lut_q29.h ${srcdir}/aie_sin_linear_lut_q15.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/fft_cf.o: ${srcdir}/fft_cf.cc ${srcdir}/fft64_twiddles.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/frame_equalizer_preprocess.o: ${srcdir}/frame_equalizer_preprocess.cc ${srcdir}/frame_equalizer_metadata.h ${srcdir}/aie_kernel_utils.h ${srcdir}/aie_trig_utils.h ${srcdir}/aie_atan_lut_q29.h ${srcdir}/aie_sin_linear_lut_q15.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/frame_equalizer_equalize.o: ${srcdir}/frame_equalizer_equalize.cc ${srcdir}/frame_equalizer_algorithm.h ${srcdir}/frame_equalizer_metadata.h ${srcdir}/aie_kernel_utils.h ${srcdir}/equalizer/select.h ${srcdir}/equalizer/common.h ${srcdir}/equalizer/constellation.h ${srcdir}/equalizer/ls.h ${srcdir}/equalizer/lms.h ${srcdir}/equalizer/comb.h ${srcdir}/equalizer/sta.h
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
