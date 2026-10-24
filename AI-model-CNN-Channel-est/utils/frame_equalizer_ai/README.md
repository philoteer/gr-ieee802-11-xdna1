# Frame Equalizer AI Training

This directory contains a deterministic GNU Radio/PyTorch pipeline for four
separate bias-free demappers: BPSK, QPSK, 16-QAM, and 64-QAM. Each network is
an MLP `3 -> hidden... -> B` with ReLU after every hidden layer and LSB-first
coded-bit logits. Architectures are selected with `--model`; the shipped ones
are `mlp_16x16` (`3 -> 16 -> 16 -> B`) and the smaller `mlp_16`
(`3 -> 16 -> B`). The chosen name determines both the trained topology and the
kernel weights header written under `kernel/demodulator/model/`. The installed
`ieee802_11` bindings supply the exact point order; the source-compatible
fallback exists only so exported checkpoints remain usable on machines without
GNU Radio.

## Commands

Run from this directory. The default run is intended to finish on a normal CPU
in minutes:

```bash
python3 cli.py train --model mlp_16x16
python3 cli.py validate --checkpoint artifacts/frame_equalizer_ai_mlp_16x16.pt
python3 cli.py export --checkpoint artifacts/frame_equalizer_ai_mlp_16x16.pt
```

`train` derives its output paths from the architecture: the checkpoint,
report, and artifact prefix land in `artifacts/frame_equalizer_ai_<model>*`
and the header is written directly to
`kernel/demodulator/model/frame_equalizer_model_<model>_weights.h`. `export` and
`validate` read the checkpoint passed with `--checkpoint` and default the
header and artifact prefix the same way. Override either with `--header` and
`--artifact-prefix`.

`data.grc` is the dataset-generator source of truth. `dataset.py` invokes
`grcc -o build data.grc`, runs the generated `build/data.py` in an isolated
process, validates its NPZ schema, and removes the temporary dataset after
loading it. Both `build/` and `artifacts/` are ignored; generated Python and
datasets must not be committed.
Run `grcc` and the generated Python under the same GNU Radio environment; the
tested environment is GNU Radio 3.10.9.2.

To inspect the generated flowgraph directly:

```bash
grcc -o build data.grc
DATASET_MODULATION=qam64 DATASET_FRAMES=8 \
  DATASET_SYMBOLS_PER_FRAME=4 DATASET_SEED=33055 \
  DATASET_OUTPUT=/tmp/qam64.npz python3 build/data.py
```

Train, validation, and test corpora are generated in separate GNU Radio runs
with fixed, non-overlapping seed bases. Frames and all 48 of their data carriers
therefore remain in one split. `model.py` applies a fresh
deterministic permutation to the training samples each epoch; validation and
test order remains stable. `dataset.py` restarts the flowgraph every 128 frames
with a derived seed, providing multiple independent fading realizations without
implementing another channel model in custom Python.

A fast end-to-end smoke run is:

```bash
python3 cli.py train --model mlp_16 --train-frames 24 --validation-frames 8 \
  --test-frames 12 --symbols-per-frame 2 --epochs 2 --batch-size 512
python3 cli.py validate --checkpoint artifacts/frame_equalizer_ai_mlp_16.pt \
  --frames 4 --symbols-per-frame 2
```

`train` writes a PyTorch checkpoint, JSON report, standalone C++ header, NPZ,
and JSON weight artifact. `--model` selects the MLP architecture from
`model.ARCHITECTURES`; the architecture fixes the hidden layers, and the
generated header carries those dimensions into the AIE build. Adding a model
means adding an `Architecture` entry, an
`demodulator/model/frame_equalizer_model_<name>.h` kernel, a selector branch in
`demodulator/model/frame_equalizer_model.h`, and a Makefile `SUPPORTED_MODELS`
entry.

The checked-in weights were trained with the current defaults and the GRC
corpus. The generated report and checkpoint are intentionally ignored; run
`train` to reproduce them before exporting a new header.

The deployed headers were generated with seed 39, 1,500/320/600 frames,
five payload symbols per frame, and 100 epochs. Their held-out provenance is:

| Model | BPSK | QPSK | 16-QAM | 64-QAM |
| --- | ---: | ---: | ---: | ---: |
| `mlp_16x16` quantized BER | 2.7743% | 3.9861% | 8.4885% | 4.4770% |
| `mlp_16` quantized BER | 2.7653% | 3.9674% | 8.4917% | 4.6171% |

The test digests are identical for both architectures because they share the
seeded corpus: `57ff0a90...` (BPSK), `4b2c258f...` (QPSK), `0b62f385...`
(16-QAM), and `88c4e97d...` (64-QAM).

## NPU BER Comparison

`test/frame_equalizer/test-learned-ber-basic.grc` uses visible GNU Radio blocks for
seeded labels, IEEE 802.11 mapping, IFFT/CP processing, hardware impairments,
multipath fading, AWGN, CP removal, and FFT processing. It generates the same
seeded corpus for either compiled engine. Its combined metric counts a failed
SIGNAL decode as a fully errored payload. This is the authoritative classic
baseline: the training dataset deliberately carries no Python zero-forcing
decisions, because those would neither use the NPU nor the `gr-ieee802-11`
receiver. Run both engines through separate builds from `test/frame_equalizer`:

```bash
make learned-ber ENGINE=CLASSIC BER_BATCHES=10 BER_OUTPUT=/tmp/classic.json
make learned-ber ENGINE=AI MODEL=MLP_16X16 BER_BATCHES=10 BER_OUTPUT=/tmp/ai.json
make learned-ber ENGINE=AI MODEL=MLP_16 BER_BATCHES=10 BER_OUTPUT=/tmp/ai-mlp16.json
```

Each modulation reports BER/SER over received packets, combined BER/SER with
every missed packet counted as fully erroneous, and the packet miss rate.

On the checked-in seed and ten batches, the GRC-trained learned engine versus
classic BER was 2.4935% versus 2.4544% for BPSK, 2.1126% versus 2.1419% for
QPSK, 2.0557% versus 2.0654% for 16-QAM, and 3.2346% versus 3.1890% for
64-QAM. Neither engine dropped a SIGNAL field. Relative to the previous learned
weights, QPSK improved from 2.1615%, 16-QAM from 2.1191%, and 64-QAM from
3.6046%; BPSK increased from 2.4414% by eight bit errors out of 15,360.

The smaller `mlp_16` architecture reaches comparable accuracy on the same
benchmark: 2.4479% for BPSK, 2.1517% for QPSK, 2.0540% for 16-QAM, and 3.3626%
for 64-QAM, again without dropping a SIGNAL field. It is both a working
compile-time alternative and the minimal fixture for exercising the model
selection path.

## Data And Integer Contract

`data.grc` constructs each waveform with standard GNU Radio blocks: a seeded
random label source, packet tags, the `gr-ieee802-11` constellation mapper and
SIGNAL formatter, OFDM carrier allocation, IFFT, cyclic prefix, hardware
impairments, selective fading, modulation-specific AWGN, cyclic-prefix removal,
and FFT. Every frame contains two L-LTF symbols, a valid SIGNAL field, and the
requested payload symbols. Only payload symbols become training samples.

The small Embedded Python feature extractor applies the production
preprocessor's pilot-derived common-phase correction to both L-LTF and data
symbols, averages the corrected L-LTF pair into an LS channel estimate, and forms
`Re(Y*conj(H_est)), Im(Y*conj(H_est)), |H_est|^2`. One common maximum-absolute
scale is calculated over all 48 feature values in each aligned 16-bin block, so
pilot, guard, and DC lanes participate before data lanes are selected.

Exported normalized inputs use signed int32 Q23 (`kInputFractionalBits = 23`
fractional bits, so `2^23` represents one).
Training and AIE both scale each 16-lane block up or down to the same
power-of-two range before inference. The reference then performs one
int8 W, acc64, ReLU, and right shift stage per hidden layer, followed by an
unshifted int8 output layer and `logit >= 0`. Hidden-layer conversions use
floor rounding and saturating int32 conversion, and the per-block input scale
truncates as well, matching the int64 reference's arithmetic `>>` and
`quantize_inputs`. Calibration feature maxima choose shifts that retain about
20 activation bits while keeping subsequent MACs safely in int64. Output bits
compose an LSB-first constellation class. Arrays are row-major with one
`[4][hidden][in]` matrix per hidden layer (the first input width is 3) plus a
padded `[4][6][hidden]` output matrix; unused output rows are zero, and the
header's `kHiddenLayers` identifies the architecture.

## Simulation Limits

The waveform starts at the OFDM mapper and therefore does not model packet
detection, CFO acquisition, timing slips, analog filtering, ADC clipping, or
interference. The Embedded Python block exists only because GNU Radio's frame
equalizer does not expose the matched-filter numerator and channel-power values
consumed by the learned AIE engine. It does not reproduce the AIE kernel's
fixed-point preprocessing errors; sampling-error tracking is omitted because
this flowgraph does not introduce CFO or SFO. Validate exported weights with
`test/frame_equalizer/test-symbols.grc` and `test-learned-ber-basic.grc` before
deployment.
