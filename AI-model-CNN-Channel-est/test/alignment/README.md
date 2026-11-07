# Toolchain and AIE2/AIE2P regression checks

From the repository root:

```sh
python3 -m unittest discover -s AI-model-CNN-Channel-est/test/alignment -v
```

The generator and make dry-run tests need no NPU. Graph checks additionally
require MLIR-AIE Python bindings and generate every production/component graph
for both devices, including the single-column standalone stages and MLP updater.
The make checks cover modern `aiecc` and legacy `aiecc.py` flags without running
either compiler.

Compile `check_alignment.cc` with `xchesscc_wrapper` for **both** `aie2` and
`aie2p`, in separate output directories, using:

```sh
xchesscc_wrapper aie2p -aietools /opt/vitis_aie_essentials \
  -I /opt/vitis_aie_essentials/include \
  -I <repo>/AI-model-CNN-Channel-est/kernel \
  -I <repo>/AI-model-CNN-Channel-est/kernel/common \
  -c <repo>/AI-model-CNN-Channel-est/test/alignment/check_alignment.cc \
  -o check_alignment.o
```

The checks preserve the 8-byte host metadata ABI and the 2168-word estimated
metadata handoff. Do not replace these ABI alignments with vector alignment.
Narrower SIMD operations retain their own alignment requirements: a 16-byte
decision store does not require a 64-byte stride on AIE2P.

## Production build

In the MLIR-AIE environment, build serially on this machine:

```sh
make -C AI-model-CNN-Channel-est/kernel -j1 JOBS=1 NPU2=1 \
  EQUALIZER_ALGORITHM=CNN CNN_DATA_UPDATER=MLP all
```

Omit `NPU2=1` for AIE2/XDNA1; use `CNN_DATA_UPDATER=NULL` for CNN without the
payload updater. Existing algorithm, engine, model, and weight defaults are
unchanged. The production build marker tracks the effective `devicename`, so
explicit device changes also invalidate stale objects. For component tests,
clean before switching architectures in the same output directory.

The existing optional `ENGINE=AI` demapper still requires
`EQUALIZER_ALGORITHM=LS`; CNN builds use the default `ENGINE=CLASSIC`. This
toolchain port does not expand that engine/algorithm support matrix.

Use a separate build-only cgroup with `MemoryMax=9G`, `MemorySwapMax=512M`, and
`OOMPolicy=kill` on the local 15 GiB machine. Both make and aiecc default to one
compiler job. Never overlap independent compiler builds.

## Why a direct baseline port was insufficient

In addition to the baseline driver/runtime, pilot-correlation, vector-alignment,
and target-specific sine-LUT fixes, the CNN needs a target-dependent halo.
Aligning its array bases is not enough: the old eight-int32 data offset is only
32 bytes. AIE2P's 16-lane int32 stores require 64-byte alignment. `kCnnHalo` and
`kCnnRowStride` now derive from the target API; AIE2 keeps its original layout.

The CNN parity tests read the checked-in Q7 headers instead of requiring missing
private NPZ training archives. The shared integer MLP reference is NumPy-only,
so parity validation does not require PyTorch or retraining.

See [`../trig_utils/README.md`](../trig_utils/README.md) for exhaustive numerical
LUT validation and `../frame_equalizer/cnn_parity/` for CNN/updater parity tests.
Recorded build and hardware results are in [`RESULTS.md`](RESULTS.md).
