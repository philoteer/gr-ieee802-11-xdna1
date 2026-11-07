# Shared compiler compatibility for component-test builds.
JOBS ?= 1
COMPILE_JOBS := $(if $(filter 1,$(JOBS)),1,2)
AIECC ?= $(shell command -v aiecc || command -v aiecc.py)
AIECC_FLAGS = $(if $(filter aiecc.py,$(notdir $(AIECC))),--aie-generate-xclbin --no-compile-host --aie-generate-npu-insts,--aietools=${AIETOOLS_DIR} --get-xclbin --get-npu-insts --xchesscc --placer=sequential_placer)
export AIETOOLS = $(AIETOOLS_DIR)
# Preserve every test's include paths and model defines for either target.
CHESSCCWRAP_FLAGS = $(if $(filter npu2,$(devicename)),aie2p,aie2) -aietools ${AIETOOLS_DIR} $(filter-out aie2,$(CHESSCCWRAP2_FLAGS))
