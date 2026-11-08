# Baseline alignment regression checks

Run the host-only generator, build-command, and graph checks from the repository
root (graph checks require the MLIR-AIE Python bindings):

```sh
python3 -m unittest discover -s base/test/alignment -v
```

Compile `check_alignment.cc` with `xchesscc_wrapper` for **both** `aie2` and
`aie2p`, using `-I <repo>/base/kernel -I <repo>/base/kernel/common`.
It checks the real target API's LUT
alignment and preserves the metadata ABI. The LUT headers deliberately include
the API themselves so they also work without first including `aie_trig_utils.h`.

Build kernel objects in separate directories for the two targets. Run compiler
jobs serially (`make -j1 JOBS=1` and `aiecc -j1`) under the local memory limits.
The synchronization kernels exercise both static and stack vector buffers;
`frame_equalizer_preprocess.cc` and `trig_utils_test.cc` exercise the shared LUTs.
For numerical validation of the target-specific LUT repetition width, see
[`../trig_utils/README.md`](../trig_utils/README.md).

Vector storage uses `aie::vector_decl_align` (32 bytes on AIE2, 64 on AIE2P).
The explicit 8-byte alignment of host-visible frame metadata is an ABI
requirement, not a vector alignment, and must not be replaced with this constant.
Narrower aligned loads/stores still use the requirements of their actual vector
width; e.g. 16-byte decision stores do not require a 64-byte stride on AIE2P.
