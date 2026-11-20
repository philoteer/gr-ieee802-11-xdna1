##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
.DEFAULT_GOAL := compile
.DELETE_ON_ERROR:
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
JOBS ?= 1
# Both kernel compilation and aiecc's internal compilation must fit in memory.
COMPILE_JOBS := $(if $(filter 1,$(JOBS)),1,2)
AIECC ?= $(shell command -v aiecc || command -v aiecc.py)
# The Python driver uses AIETOOLS from the environment; only the declarative
# driver accepts --aietools, along with its new output selectors.
AIECC_FLAGS = $(if $(filter aiecc.py,$(notdir $(AIECC))),--aie-generate-xclbin --no-compile-host --aie-generate-npu-insts,--aietools=${AIETOOLS_DIR} --get-xclbin --get-npu-insts --xchesscc --placer=sequential_placer)
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH) 
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

# mlir-aie/programming_examples/makefile-common
#AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc))/../)
AIETOOLS_DIR = /opt/vitis_aie_essentials/
# Also used by wrappers invoked internally by aiecc.
export AIETOOLS := $(AIETOOLS_DIR)
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib
MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}
CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include
CHESSCCWRAP_COMMON_FLAGS = -aietools ${AIETOOLS_DIR} -I ${AIETOOLS_DIR}/include -I ${srcdir} -I ${srcdir}/common
CHESSCCWRAP2_FLAGS = aie2 $(CHESSCCWRAP_COMMON_FLAGS)
CHESSCCWRAP2P_FLAGS = aie2p $(CHESSCCWRAP_COMMON_FLAGS)
# Keep the object architecture consistent with the graph's selected device.
CHESSCCWRAP_FLAGS = $(if $(filter npu2,$(devicename)),$(CHESSCCWRAP2P_FLAGS),$(CHESSCCWRAP2_FLAGS))
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
.PHONY: all clean compile FORCE

KERNEL_OBJECTS := \
	build/autocorr_cf.o \
	build/magsq_avg_cf.o \
	build/threshold_cf.o \
	build/sync_short_cf.o \
	build/sync_long_cf.o \
	build/fft_cf.o \
	build/frame_equalizer_preprocess.o \
	build/frame_equalizer_equalize.o \
	build/decode_mac.o

BUILD_OUTPUTS := build/final.xclbin build/insts.bin

all: ${BUILD_OUTPUTS}

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

FORCE:

# Track graph/device settings so an older equalizer-only xclbin cannot be
# silently reused with the new MAC-output host ABI.
build/.config: FORCE
	@mkdir -p ${@D}
	@printf 'OUTPUT=MAC_V1\nDEVICE=%s\nTILE=%s\nTILES=%s\nFIFO_DEPTH=%s\nFIFO_DEPTHS=%s\n' \
		'${devicename}' '$(if ${AIE_TILE_SIZE},${AIE_TILE_SIZE},1024)' \
		'$(if ${AIE_TILES_PER_BATCH},${AIE_TILES_PER_BATCH},16)' \
		'$(if ${AIE_FIFO_DEPTH},${AIE_FIFO_DEPTH},2)' '${AIE_FIFO_DEPTHS}' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

build/aie.mlir: ${srcdir}/aie2_memtile_caching.py build/.config
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

# Compile autocorr_cf.cc to autocorr_cf.o
build/autocorr_cf.o: ${srcdir}/autocorr_cf.cc ${srcdir}/common/aie_kernel_utils.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

# Compile magsq_avg_cf.cc to magsq_avg_cf.o
build/magsq_avg_cf.o: ${srcdir}/magsq_avg_cf.cc ${srcdir}/common/aie_kernel_utils.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/threshold_cf.o: ${srcdir}/threshold_cf.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

TRIG_HEADERS := $(wildcard ${srcdir}/common/aie_*.h)
EQUALIZER_HEADERS := $(wildcard ${srcdir}/common/*.h ${srcdir}/channel_estimator/*.h)

build/sync_short_cf.o: ${srcdir}/sync_short_cf.cc ${TRIG_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/sync_long_cf.o: ${srcdir}/sync_long_cf.cc ${TRIG_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/fft_cf.o: ${srcdir}/fft_cf.cc ${srcdir}/common/fft64_twiddles.h ${srcdir}/common/aie_kernel_utils.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/frame_equalizer_preprocess.o: ${srcdir}/frame_equalizer_preprocess.cc ${srcdir}/common/frame_equalizer_metadata.h ${TRIG_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/frame_equalizer_equalize.o: ${srcdir}/frame_equalizer_equalize.cc ${EQUALIZER_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/decode_mac.o: ${srcdir}/decode_mac.cc ${srcdir}/common/decode_mac.h ${srcdir}/common/frame_equalizer_metadata.h ${srcdir}/common/viterbi_tables.h ${srcdir}/common/aie_kernel_utils.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

${KERNEL_OBJECTS}: build/.config

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
