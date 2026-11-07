# AI-model-basic toolchain and alignment checks

From the repository root:

```sh
python3 -m unittest discover -s AI-model-basic/test/alignment -v
```

The generator tests reproduce the combined `kernel/common/aie_trig_luts.h`.
The build-command tests cover the production and all component-test builds,
including both compiler drivers and preservation of AI engine/model selection.

Compile `check_alignment.cc` with `xchesscc_wrapper` for both `aie2` and `aie2p`,
using `-I <repo>/AI-model-basic/kernel/common`. Use separate output directories
for each architecture, and run compiler jobs serially under local memory limits.
The LUT header includes the API itself, so the test needs no trig-utils include.

Vector storage uses `aie::vector_decl_align` (32 bytes on AIE2, 64 on AIE2P).
The four-access sine LUT also needs target-specific bank repetition: 128 bits on
AIE2, 256 bits on AIE2P. Metadata's explicit 8-byte alignment is a host ABI and
must not be changed. Narrow vector stores (e.g. 16 int8 decisions) still use their
actual width's alignment; they do not require a 64-byte stride on AIE2P.

The tag word at byte offset 4 is the host's `frame_start` flag, not padding.
Symbol-accuracy tests alone do not prove that the full receiver emits the packet
tags needed by `decode_mac`. See [`../packet_decode`](../packet_decode/README.md)
for a recorded-IQ, CRC-valid packet regression through the complete pipeline.

For numerical validation, build `test/trig_utils` with `make -j1 JOBS=1 NPU2=1`,
then run `python3 test_accuracy.py --exhaustive-linear` there. This checks all
262,144 quarter-wave coordinates across all quadrants against scalar arithmetic.
