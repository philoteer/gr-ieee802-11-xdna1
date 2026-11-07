# Full receiver packet-decoding regression

`test_recorded.py` exercises recorded int16 IQ through synchronization, CFO/phase
correction, FFT, the selected NPU demapper, and GNU Radio's MAC decoder. It uses
the same preprocessing, frequency-tag injector, batch size, and two-slot host as
`Experiments/quality_frames_rx/NPU/NPU.grc`. It does not bypass the NPU or feed
already-equalized symbols to the demapper.

Run from the repository root with GNU Radio's interpreter (on this machine,
`/usr/bin/python3`; the MLIR-AIE virtualenv has an incompatible NumPy ABI):

```sh
# Run these sequentially. Each output directory must be new.
/usr/bin/python3 AI-model-basic/test/packet_decode/test_recorded.py \
    --backend cpu --output /tmp/opencode/packets-cpu
/usr/bin/python3 AI-model-basic/test/packet_decode/test_recorded.py \
    --reference /tmp/opencode/packets-cpu/report.json \
    --output /tmp/opencode/packets-ai
```

The default input is the first 327,680 complex samples of the shared
`stress_test.sigmf-data` recording. `--capture`, `--metadata`, `--start`, and
`--samples` select another window. All offsets/counts are complex samples, not
int16 components. Counts must be multiples of 16,384 to avoid silently dropping
an incomplete NPU batch. The active SigMF frequency at the start and subsequent
tuning changes are passed through the normal host's `rx_freq` tags/RTP updates.

`--kernel-dir` selects another paired `final.xclbin`/`insts.bin`; the default is
`AI-model-basic/kernel/build`. To rebuild, use the activated compiler environment:

```sh
make -C AI-model-basic/kernel -j1 JOBS=1 NPU2=1 ENGINE=AI MODEL=MLP_16X16 all
```

Use a memory-limited build-only cgroup on memory-constrained hosts. Do not run
other NPU workloads concurrently. Changing model/target flags also requires a
clean or separate build directory; Make does not track command-line settings.

Outputs are `packets.pcap`, `report.json`, and `decisions.npy`. PCAP contains raw
802.11 MAC frames (DLT 105), after the decoder has verified and removed FCS.
Its timestamps are packet ordinals, not measured capture times. Reports include
input-window/metadata/artifact hashes, frame-tag counts, CRC-valid packet counts,
encoding counts, and per-packet hashes. `--reference` requires byte-identical
packet multisets, including duplicate counts. Do not require identical recovery
on arbitrary noisy windows: different demappers can legitimately recover
different packets. A positive regression fails if no packets decode; use
`--min-packets 0` only for explicitly idle/negative-control windows.

## Packet-start metadata fix

The current host emits `frame bytes` and `encoding` only when bit zero of the
metadata word at byte offset 4 is set. The AI implementation still called this
word `reserved` and always wrote zero. Thus it could pass symbol-decision tests
but its normal full-PHY host could never start the MAC decoder.

The kernel now implements the same `frame_start` contract as `base/` (introduced
there by `cb62da9`): set the bit on frame-relative symbol 3, the first payload
symbol, and clear it on subsequent per-symbol debug tags. Structure sizes and
all other field offsets are unchanged. No host installation changes or fallback
to the classic demapper are required.
