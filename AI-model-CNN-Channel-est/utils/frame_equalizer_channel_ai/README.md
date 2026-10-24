# frame_equalizer_channel_ai(1)

## NAME

frame_equalizer_channel_ai — generate channel-estimation data, train and export
quantized CNN weights, then evaluate the exported model and select an LS gate.

## SYNOPSIS

Run these commands from `AI-model-CNN-Channel-est/`. The three entry points are independent:

```text
python3 utils/frame_equalizer_channel_ai/dataset_generator/generate_dataset.py [options]
python3 utils/frame_equalizer_channel_ai/training/cli.py {train,export} [options]
python3 utils/frame_equalizer_channel_ai/verification/cli.py {gate,evaluate} [options]
```

Use `<program> <command> --help` for the full list of options (or `<program>
--help` for the dataset generator).

## EXAMPLE

### Cross-domain CNN/MLP pairs: bus to room and room to bus

Reproduce both source-validated cross-domain runs from `AI-model-CNN-Channel-est/`:

```bash
python3 utils/frame_equalizer_channel_ai/training/train_cross_domain.py
```

To reuse CNNs trained with this policy and train only their matching MLP updaters:

```bash
python3 utils/frame_equalizer_channel_ai/training/train_cross_domain.py --updater-only
```

Both CNN and MLP use the saved **by-frame** partitions with this reassignment:

| Experiment split | Collection and saved partitions | Purpose |
| --- | --- | --- |
| Training | Source domain `train + test` | Fit weights |
| Validation | Source domain `val` | Select epochs and float/Q7 fallback gates |
| Final test | Target domain `train + val + test` | Report out-of-domain performance only |

Thus bus→room validates on bus and tests on room; room→bus validates on room
and tests on bus. The original source partition named `test` is deliberately
repurposed for training. Source validation frames remain disjoint from fitting
frames, and each target frame appears once. Only one partition representation
is used; channels are not reassigned. Model selection never uses target inputs,
labels, losses, or gate scores. CNN test NPZ loading and MLP target-data assembly
are deferred until weights and gates are fixed. Regression tests change target
observations/labels and check that selected weights, epochs, and gates do not change.

CNN training uses seed 40, batch size 256, learning rate 0.002, noise augmentation
0.02, up to 60 epochs and patience 8. Final out-of-domain test results:

| Train → test | CNN train / validation / test examples | Best epoch / epochs trained | LS NMSE | NPU Q7 NMSE | CPU float NMSE |
| --- | --- | --- | --- | --- | --- |
| Bus → room | 37,401 / 4,191 / 44,334 | 48 / 56 | 0.015692 | 0.008754 | 0.008399 |
| Room → bus | 39,612 / 4,722 / 41,592 | 57 / 60 | 0.079603 | 0.036731 | 0.036509 |

NMSE is linear (lower is better). The two test populations differ, so these
rows alone do not isolate a domain penalty against an in-domain model. The
source-selected CNN gate (SNR-ratio×16) is disabled (`UINT64_MAX`) for bus→room
and 32,000 for room→bus, for both arithmetic modes. Reports, checkpoints, and
exports are in `training/artifacts/{bus_to_room,room_to_bus}/`. Dataset manifests
and NPZ files are in `data/datasets/channel_estimator/` under those names. They
record the exact partition policy, frame identities, sample counts, and digests;
CNN checkpoint metadata also records the policy and split digests.

Each MLP is fitted only on its source domain against the matching exported
quantized CNN baseline. It uses the four `61 -> 16 -> 26` group networks and
training settings described below for the recorded-split updaters. It follows
the same source `train + test` / source `val` / target-all policy as its CNN.
Updater targets exclude the baseline pair and current symbol. Final test results:

| Train → test | MLP train / validation / test examples | Best epochs (four groups) | Q7 residual MSE / CNN-only baseline | Float MLP residual MSE / baseline |
| --- | --- | --- | --- | --- |
| Bus → room | 35,865 / 4,020 / 43,017 | 35, 17, 34, 24 | 0.866766 (13.32% reduction) | 0.863433 |
| Room → bus | 38,429 / 4,588 / 39,885 | 31, 39, 39, 39 | 0.995017 (0.50% reduction) | 0.993342 |

These are normalized residual-MSE test metrics, not the CNN NMSE above.
Both Q7 and float MLP metrics use the quantized CNN baseline used for training.
For comparison, Q7 residual-MSE ratios on source validation are 0.767610
(bus→room) and 0.795228 (room→bus). The room-trained updater's in-domain benefit
largely disappears on bus. Updater gates (SNR-ratio×16) are 7,164 for bus→room
and 22,654 for room→bus, identical for Q7 and float. `channel_updater.pt` and `updater_report.json`
are saved alongside each CNN's artifacts; generated artifacts remain ignored
by Git. Reports include data digests, collection paths, and the matching CNN
header's SHA-256. These source-selected runs supersede the earlier experiments
that selected epochs/gates on the target domain. `--updater-only` rejects CNN
checkpoints lacking the new source-only policy; old updater checkpoints must
also be retrained before re-exporting under this policy.

NPU build selection (run these commands sequentially; the last build is active):

```bash
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=BUS_TO_ROOM CNN_DATA_UPDATER=NULL
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=ROOM_TO_BUS CNN_DATA_UPDATER=NULL
# Enable each CNN's matching MLP updater:
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=BUS_TO_ROOM CNN_DATA_UPDATER=MLP
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=ROOM_TO_BUS CNN_DATA_UPDATER=MLP
```

GNU Radio's regular and debug Algorithm dropdowns include
`ieee802_11.CNN_BUS_TO_ROOM` (14), `CNN_ROOM_TO_BUS` (15),
`CNN_BUS_TO_ROOM_INT8` (16), and `CNN_ROOM_TO_BUS_INT8` (17).
The first two use floating-point weights; the INT8 variants use the NPU Q7
weights. Switching models through `set_algorithm()` requires no recompilation
after rebuilding/installing the updated GNU Radio library and bindings.
The matching CNN+MLP variants are `CNN_BUS_TO_ROOM_MLP` (18),
`CNN_ROOM_TO_BUS_MLP` (19), `CNN_BUS_TO_ROOM_MLP_INT8` (20), and
`CNN_ROOM_TO_BUS_MLP_INT8` (21), also in the `ieee802_11` namespace.

Verify both NPU exports against the integer reference on hardware:

```bash
make -C test/frame_equalizer/cnn_parity -j1 test CNN_WEIGHTS=BUS_TO_ROOM
make -C test/frame_equalizer/cnn_parity -j1 test CNN_WEIGHTS=ROOM_TO_BUS
make -C test/frame_equalizer/cnn_parity -j1 updater-test CNN_WEIGHTS=BUS_TO_ROOM
make -C test/frame_equalizer/cnn_parity -j1 updater-test CNN_WEIGHTS=ROOM_TO_BUS
```

The `-j1` make invocations above keep CHESS object compilation serial.
Makefile defaults remain `JOBS ?= 2`; `aiecc.py` uses its default parallelism.
Run only one NPU make invocation at a time.

Re-evaluate an exported updater on its entire target domain with:

```bash
python3 utils/frame_equalizer_channel_ai/verification/updater.py --split-mode bus_to_room --split test
python3 utils/frame_equalizer_channel_ai/verification/updater.py --split-mode room_to_bus --split test
```

`--split train` selects source `train + test`, `--split val` selects source `val`,
and `--split test` selects the entire target domain. Reports explicitly record
`test_is_validation: false`. Default collection and model paths follow the direction.

### Recorded frame-split and channel-split CNN/MLP pairs

Each split policy has its own CNN and MLP updater, both trained on the combined
room and bus collections using only that policy's saved train partition.
The updater uses the matching quantized CNN to construct its baseline and
residual targets. Validation selects the best epochs; fallback gates are
selected independently for integer NPU and floating-point CPU inference.
Test partitions are used only for reporting. Reproduce conversion, training,
and both exports with:

```bash
python3 utils/frame_equalizer_channel_ai/training/train_presplit_cnn_models.py
```

This wrapper uses the existing SigMF partitions; it does not split the dataset
again. It regenerates their NPZ representations before training both CNN/MLP pairs.
CNN training uses seed 40, up to 60 epochs, batch size 256, Adam learning rate 0.002,
noise augmentation 0.02, and early stopping patience 8. Converted datasets are
saved under `data/datasets/channel_estimator/{collection}_{by_frame,by_channel}/`.
The general training CLI also accepts comma-separated source names in `--dataset`.

To keep the saved CNNs and train only their matching MLP updaters:

```bash
python3 utils/frame_equalizer_channel_ai/training/train_presplit_cnn_models.py --updater-only
```

MLP training uses four bias-free `61 -> 16 -> 26` group networks, a clipped
ReLU hidden layer, clipped linear outputs, Adam with constant learning rate
0.001, batch size 256, up to 40 epochs, patience 6, minimum validation-MSE
improvement `1e-7`, and seed 40. Training-only Gaussian augmentation has a
per-example standard deviation sampled uniformly from `[0, 0.02]` on the first
60 features; the SNR feature is unchanged. Inputs/targets are normalized Q15
values and the loss is residual MSE. Averaged reference targets exclude the
two baseline symbols and the current pilot-bearing symbol, using radius 8
and at least 4 independent reference symbols.

The MLP reads the saved SigMF partitions directly because it also needs the
following payload symbol and its pilots. `training/updater.py --recordings`
accepts multiple collection directories. Its default CNN, output headers,
checkpoint and report paths all follow `--split-mode by_frame|by_channel`.

Saved checkpoints, losses, sample counts, test digests, exported NPZ/JSON weights,
and metrics are in `training/artifacts/by_frame/` and `training/artifacts/by_channel/`
relative to this README. Deployment headers are:

* NPU: `kernel/channel_estimator/model/frame_equalizer_channel_ai_weights_by_frame.h`
  and `frame_equalizer_channel_ai_weights_by_channel.h`.
* GNU Radio: `gr-ieee802-11-modified/lib/equalizer/model/channel_cnn_weights_by_frame.h`
  and `channel_cnn_weights_by_channel.h` (relative to the repository root).
* MLP NPU: `kernel/channel_estimator/model/frame_equalizer_channel_updater_mlp_weights_by_frame.h`
  and `frame_equalizer_channel_updater_mlp_weights_by_channel.h`.
* MLP GNU Radio: `gr-ieee802-11-modified/lib/equalizer/model/channel_updater_mlp_weights_by_frame.h`
  and `channel_updater_mlp_weights_by_channel.h` (relative to the repository root).

Each artifact directory also contains `channel_updater.pt` and
`updater_report.json`. The reports record the dataset policy, collection paths,
training settings, per-group validation histories, best epochs, gates, data
digests, and the SHA-256 of the exact CNN header used to construct the baseline.

Held-out test results from the saved reports (linear NMSE, lower is better):

| Split | Train / validation / test examples | Best epoch | LS | NPU Q7 CNN | CPU float CNN |
| --- | --- | --- | --- | --- | --- |
| Frame | 69,321 / 8,913 / 7,692 | 45 (53 trained) | 0.050843 | 0.022443 | 0.022384 |
| Channel | 43,365 / 27,439 / 15,122 | 22 (30 trained) | 0.089984 | 0.044875 | 0.044966 |

Channel-split training excludes channels 1 and 13; validation uses channel 1
and testing uses channel 13. Frame-split partitions contain disjoint frames.
The two test populations differ, so these rows are not a head-to-head comparison.

MLP held-out results, relative to the matching CNN without an updater:

| Split | Train / validation / test MLP examples | Best epochs (four groups) | Quantized MLP residual MSE / baseline |
| --- | --- | --- | --- |
| Frame | 66,906 / 8,608 / 7,388 | 34, 21, 34, 38 | 0.845998 (15.40% reduction) |
| Channel | 42,163 / 26,644 / 14,095 | 11, 7, 10, 18 | 0.919420 (8.06% reduction) |

MLP example counts differ from CNN counts because an updater example requires
a following payload symbol and independent surrounding reference symbols.
These are normalized residual-MSE metrics, not the CNN NMSE values above.

#### NPU compile-time selection

From `AI-model-CNN-Channel-est/`:

```bash
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=BY_FRAME
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=BY_CHANNEL
# Enable the matching MLP updater as well:
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=BY_FRAME CNN_DATA_UPDATER=MLP
make -C kernel -j1 all EQUALIZER_ALGORITHM=CNN CNN_WEIGHTS=BY_CHANNEL CNN_DATA_UPDATER=MLP
```

`CNN_WEIGHTS` supports `BY_FRAME` (the default), `BY_CHANNEL`, `BUS_TO_ROOM`,
and `ROOM_TO_BUS`.
The build records the selection in `kernel/build/frame_equalizer_build_mode.txt`
and automatically rebuilds when it changes. The underlying C++ macro is
`FRAME_EQUALIZER_CNN_WEIGHTS`, with values `FRAME_EQUALIZER_CNN_WEIGHTS_BY_FRAME`,
and `FRAME_EQUALIZER_CNN_WEIGHTS_BY_CHANNEL`.
`CNN_DATA_UPDATER=NULL|MLP` enables/disables payload updates. With `MLP`,
`CNN_WEIGHTS` selects the matching CNN and MLP together; their split policies
cannot be selected independently.

#### GNU Radio post-build selection

Build/install `gr-ieee802-11-modified` once. Both the regular and debug frame
equalizer blocks expose these methods in their Algorithm dropdown:

| Python enum | Value | Model |
| --- | --- | --- |
| `ieee802_11.CNN_FRAME` | 6 | Frame-split CNN |
| `ieee802_11.CNN_CHANNEL` | 7 | Channel-split CNN |
| `ieee802_11.CNN_FRAME_MLP` | 8 | Frame-split CNN + frame-split MLP updater |
| `ieee802_11.CNN_CHANNEL_MLP` | 9 | Channel-split CNN + channel-split MLP updater |

Select at block construction or call, for example,
`block.set_algorithm(ieee802_11.CNN_CHANNEL)` between frames. Both sets are
included in the same library, so switching requires no recompilation.
Each method selects both CNN and MLP weights and their respective gates as one
pair. Both split-specific MLPs are included in the same library.

From the repository root, build and check the GNU Radio integration with:

```bash
cmake -S gr-ieee802-11-modified -B gr-ieee802-11-modified/build -DCMAKE_BUILD_TYPE=Release
cmake --build gr-ieee802-11-modified/build -j2
ctest --test-dir gr-ieee802-11-modified/build --output-on-failure
python3 gr-ieee802-11-modified/utils/test_channel_ai_grc.py
python3 gr-ieee802-11-modified/utils/test_channel_ai_grc.py --debug-block
```

On an NPU-equipped system, verify bit-exact CNN+MLP parity against the Python
integer reference from `AI-model-CNN-Channel-est/`:

```bash
make -C test/frame_equalizer/cnn_parity -j1 updater-test CNN_WEIGHTS=BY_FRAME
make -C test/frame_equalizer/cnn_parity -j1 updater-test CNN_WEIGHTS=BY_CHANNEL
```

### Synthetic CNN

Generate both synthetic and GNU Radio splits, train with the same dataset, pick
the gate on the validation split, re-export the weights with that gate, and
evaluate on the held-out test split:

```bash
python3 utils/frame_equalizer_channel_ai/dataset_generator/generate_dataset.py --dataset both
python3 utils/frame_equalizer_channel_ai/training/cli.py train --dataset both
python3 utils/frame_equalizer_channel_ai/verification/cli.py gate --dataset both
# Replace the example value with gate_snr_ratio_x16 from the preceding JSON output.
GATE=20191
python3 utils/frame_equalizer_channel_ai/training/cli.py export --gate-snr-ratio-x16 "$GATE"
python3 utils/frame_equalizer_channel_ai/verification/cli.py evaluate --dataset both --checkpoint utils/frame_equalizer_channel_ai/training/artifacts/channel_estimator.pt
```

For an original-only run, omit `--dataset both` throughout. Gate
selection does not modify the weights: the second export embeds the selected
threshold in the NPZ, JSON, and C++ header. The initial training export uses
an effectively disabled gate. Export writes the deployed header to
`kernel/channel_estimator/model/frame_equalizer_channel_ai_weights_by_frame.h` by
default; rebuilding the kernel is required to deploy new weights.

## INPUTS AND OUTPUTS

The room and bus real-symbol collections are fixed SigMF splits under
`data/datasets/channel_estimator/{20260927_room,20260928_bus}/by_frame/` and
`by_channel/`, each with `train/`, `val/`, `test/` and a checksummed manifest.
The unsplit originals are archived at
`data/datasets/channel_estimator_sources_archive/`; see `data/README.md` for
the split policy and regeneration command. Convert only the saved partitions:

```bash
python3 utils/frame_equalizer_channel_ai/dataset_generator/convert_sigmf.py --recordings data/datasets/channel_estimator/20260927_room --split-mode by_frame
python3 utils/frame_equalizer_channel_ai/dataset_generator/convert_sigmf.py --recordings data/datasets/channel_estimator/20260928_bus --split-mode by_channel
python3 utils/frame_equalizer_channel_ai/training/updater.py --recordings data/datasets/channel_estimator/20260927_room --split-mode by_frame
python3 utils/frame_equalizer_channel_ai/verification/updater.py --recordings data/datasets/channel_estimator/20260927_room --split-mode by_frame --split test
```

The converter writes `train.npz`, `validation.npz`, and `test.npz`; its split
source is always the saved SigMF version. Updater training uses saved train/val
for fitting and gate selection; verification defaults to saved test.

The dataset generator writes `train.npz`, `validation.npz`, and `test.npz` under
`data/datasets/channel_estimator/synthetic_original/` and/or
`data/datasets/channel_estimator/synthetic_gnuradio/`, with a manifest for each
source. `--dataset both` combines the matching splits when training or
verifying. The default generation sizes are 12,000 training, 2,000 validation,
and 3,000 test samples total. Use the same `--dataset-dir` with all three
programs if storing splits elsewhere. To convert a recorded GNU Radio archive,
use `generate_dataset.py --convert RAW_NPZ --output PATH`, then place the
converted split at `DATASET_DIR/SOURCE/SPLIT.npz`.

Training writes `training/artifacts/channel_estimator.pt` (checkpoint),
`channel_estimator_report.json` (settings and loss), and
`channel_estimator_weights.npz`/`.json` (exported integer weights and gate).
Verification consumes the NPZ; it does not require the training checkpoint for
`gate` or ordinary `evaluate`. `gate` defaults to the validation split;
`evaluate` defaults to the test split and reports LS, ungated quantized, and
gated NMSE/EVM plus per-SNR results. Pass `--checkpoint` to `evaluate` on the
full test split to check its digest against the training checkpoint. Use
`--report PATH` to save verification output as JSON.

## TESTS

```bash
python3 -m pytest -q utils/frame_equalizer_channel_ai
```
