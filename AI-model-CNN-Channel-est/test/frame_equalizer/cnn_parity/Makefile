.DEFAULT_GOAL := compile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
kernel_srcdir := $(realpath ${srcdir}/../../../kernel)
CNN_WEIGHTS ?= BY_FRAME
ifeq ($(filter ${CNN_WEIGHTS},BY_FRAME BY_CHANNEL BUS_TO_ROOM ROOM_TO_BUS),)
$(error CNN_WEIGHTS must be BY_FRAME, BY_CHANNEL, BUS_TO_ROOM or ROOM_TO_BUS)
endif
split_mode := $(shell printf '%s' '${CNN_WEIGHTS}' | tr A-Z a-z)
channel_ai_header := ${kernel_srcdir}/channel_estimator/model/frame_equalizer_channel_ai_weights_${split_mode}.h
UPDATER_BUILD := build_updater_${split_mode}
CNN_BUILD := build_${split_mode}
include ${srcdir}/../../toolchain.mk

ifneq ($(filter 1 2,$(JOBS)),$(JOBS))
$(error JOBS must be 1 or 2)
endif

export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include -I ${kernel_srcdir} -I ${kernel_srcdir}/common \
	-DFRAME_EQUALIZER_CNN_WEIGHTS=FRAME_EQUALIZER_CNN_WEIGHTS_${CNN_WEIGHTS}
BUILD_OUTPUTS := ${CNN_BUILD}/final.xclbin ${CNN_BUILD}/insts.bin
CNN_HEADERS := ${kernel_srcdir}/common/aie_kernel_utils.h \
	${kernel_srcdir}/common/frame_equalizer_algorithm.h \
	${kernel_srcdir}/channel_estimator/common.h \
	${kernel_srcdir}/channel_estimator/select.h \
	${kernel_srcdir}/channel_estimator/cnn.h \
	${kernel_srcdir}/channel_estimator/cnn_common.h \
	${kernel_srcdir}/channel_estimator/model/select.h \
	${kernel_srcdir}/channel_estimator/model/frame_equalizer_channel_ai_weights_${split_mode}.h

.PHONY: all clean compile test generate-reference updater-test

all: ${BUILD_OUTPUTS}

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

${CNN_BUILD}/aie.mlir: ${srcdir}/aie2_memtile_caching.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

${CNN_BUILD}/cnn_parity_test.o: ${srcdir}/cnn_parity_test.cc ${CNN_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

${CNN_BUILD}/final.xclbin: ${CNN_BUILD}/aie.mlir ${CNN_BUILD}/cnn_parity_test.o
	mkdir -p ${@D}
	cd ${@D} && $(AIECC) -j$(COMPILE_JOBS) $(AIECC_FLAGS) --xclbin-name=${@F} \
		--npu-insts-name=insts.bin $(<:%=../%)

${CNN_BUILD}/insts.bin: ${CNN_BUILD}/final.xclbin
	@test -f $@

clean:
	rm -rf build build_by_frame build_by_channel build_bus_to_room build_room_to_bus build_updater build_updater_by_frame build_updater_by_channel build_updater_bus_to_room build_updater_room_to_bus _build __pycache__

compile:
	${MAKE} clean
	${MAKE} -j${COMPILE_JOBS} all

# The NumPy reference reads the exact checked-in Q7 weights compiled above;
# no private training archive or retraining is required for a parity check.
generate-reference: ${channel_ai_header}

test: generate-reference
	${MAKE} all
	python3 ${srcdir}/test_parity.py --xclbin ${CNN_BUILD}/final.xclbin --insts ${CNN_BUILD}/insts.bin --split-mode ${split_mode}

${UPDATER_BUILD}/aie.mlir: ${srcdir}/aie2_updater.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

${UPDATER_BUILD}/updater_parity_test.o: ${srcdir}/updater_parity_test.cc ${CNN_HEADERS} ${kernel_srcdir}/channel_estimator/cnn_data_updater/mlp.h ${kernel_srcdir}/channel_estimator/model/select_updater.h ${kernel_srcdir}/channel_estimator/model/frame_equalizer_channel_updater_mlp_weights_${split_mode}.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP_FLAGS} -c $< -o ${@F}

${UPDATER_BUILD}/final.xclbin: ${UPDATER_BUILD}/aie.mlir ${UPDATER_BUILD}/updater_parity_test.o
	cd ${@D} && $(AIECC) -j$(COMPILE_JOBS) $(AIECC_FLAGS) --xclbin-name=${@F} \
		--npu-insts-name=insts.bin $(<:%=../%)

updater-test: ${UPDATER_BUILD}/final.xclbin
	python3 ${srcdir}/test_updater.py --xclbin ${UPDATER_BUILD}/final.xclbin --insts ${UPDATER_BUILD}/insts.bin --split-mode ${split_mode}
