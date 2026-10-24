# Stage 42 data catalog

This directory contains channel-estimator datasets, SDR captures, and temporary
outputs used by the Stage 42 flowgraphs and utilities. Paths below are relative
to `stage42/`; run commands that use them from that directory unless a utility
says otherwise.

## Storage and layout

| Path | Contents |
| --- | --- |
| `data/datasets/channel_estimator/` | Training-ready synthetic splits and SigMF recordings with decoded Wi-Fi symbols and transmitted reference symbols. |
| `data/raw_capture/` | Raw SDR I/Q recordings and a few auxiliary binary captures. |
| `data/tmp_out/` | Disposable PCAP, JSONL, and other outputs produced by test/diagnostic flowgraphs. Do not use these as source datasets. |

`data/datasets/` and `data/raw_capture/` are repository symlinks. In the
current setup they resolve to `/mnt/SUB/datasets/` and `/mnt/SUB/raw_capture/`;
the large recording payloads live there rather than in the Git worktree. Keep
using the logical `data/...` paths in commands, and check that the targets are
mounted and readable before running a job in another environment. Do not replace
the symlinks with copies. Generated files in `data/tmp_out/` are ignored by
Git; `.gitkeep` preserves the directory.

## Channel-estimator datasets

### Synthetic corpora

`data/datasets/channel_estimator/synthetic_original/` and
`data/datasets/channel_estimator/synthetic_gnuradio/` each contain
`train.npz`, `validation.npz`, `test.npz`, and `manifest.json`. The manifests
identify the generator and record split sizes, digests, seed, and SNR range.
The currently present corpora each have 6,000 training, 1,000 validation, and
1,500 test examples (seed 40; SNR 2–40 dB).

Use the channel-estimator training and verification workflow described in
`utils/frame_equalizer_channel_ai/README.md`. Its dataset generator can recreate
the synthetic splits; the training and verification CLIs consume these NPZ
splits directly.

### Real-world symbol corpora

| Directory | Contents and intended use |
| --- | --- |
| `data/datasets/channel_estimator/20260927_room/` | Room/lab collection from 2026-09-27, pre-split as annotated two-channel `cf32_le` SigMF decoded Wi-Fi symbols and transmitted references. Covers Wi-Fi channels 1–13. |
| `data/datasets/channel_estimator/20260928_bus/` | Moving bus collection, pre-split as annotated two-channel `cf32_le` SigMF decoded Wi-Fi symbols and transmitted references. Route 206 was outbound and route 207 was the return trip. |

Each collection has `by_frame/` and `by_channel/`, each containing `train/`,
`val/`, `test/` SigMF pairs and a `manifest.json` with original frame offsets
and SHA-256 checksums. `by_frame` assigns whole frames deterministically
(seed 40, 80/10/10); `by_channel` holds out channel 1 for validation and
channel 13 for testing, and trains on all other channels. Both include every
annotated frame exactly once per version. Output metadata rebases annotations
and captures to the concatenated samples; symbol and reference bytes and
per-frame frequency metadata are preserved. Split validation checks checksums,
duplicate frame identities and channel isolation before conversion or training.

The unpartitioned recordings and their `.txt`/`doc` notes were moved outside
the training dataset tree to
`data/datasets/channel_estimator_sources_archive/{20260927_room,20260928_bus}/`.
Do not point consumers at this archive. To regenerate, remove both split
versions from each collection and run from this stage directory:

```bash
python3 utils/frame_equalizer_channel_ai/dataset_generator/split_sigmf.py
```

This fails rather than overwriting an existing split. The converter
`utils/frame_equalizer_channel_ai/dataset_generator/convert_sigmf.py` reads
only a selected pre-split version (`--recordings` collection directory and
`--split-mode by_frame|by_channel`). The updater training and verification
commands use the same options; verification also selects `--split test` by
default. These recordings contain decoded payload symbols, not raw LTS or SDR
I/Q. The raw SDR files described below must first be decoded into annotated
symbol recordings.

## Raw SDR captures

`data/raw_capture/` contains these SigMF recordings:

| Basename | Description |
| --- | --- |
| `room_20260927` | Raw room recording, tuned across Wi-Fi channels 1–13. |
| `206` | Raw bus recording, tuned across Wi-Fi channels 1–13. This is distinct from the decoded-symbol `20260928_bus/206` corpus. |
| `stress_test` | Lab capture dated 2026-09-29 at 2412 MHz (channel 1), approximately two minutes with iperf3 traffic at about 20 Mbit/s; see `data/raw_capture/doc.txt`. |

Each has a `.sigmf-data` payload and `.sigmf-meta` sidecar. The sidecars specify
`ci16_le` complex I/Q, 20 MS/s, and USRP B210 capture metadata. For the room and
bus files, `captures` entries mark sample offsets where the tuned frequency
changes; channel 1 (2412 MHz) through channel 13 (2472 MHz) are swept, so do not
interpret the entire file as a single fixed-frequency capture. Use the
sidecar's sample offsets/frequencies when selecting channel-specific data.
These raw recordings are not the same representation as the `cf32_le`
channel-estimator symbol corpora.

The sidecars currently set `core:dataset` to the generic name
`out.sigmf-data`, although each actual adjacent payload has a capture-specific
basename. When using SigMF readers, confirm they resolve the intended sibling
payload; do not rename or rewrite the capture based only on that metadata field.

Additional binary files in this directory include `wifi_iq_hackrf.bin` and
`output_head_short.bin`. They have no SigMF sidecars here. The
`utils/compare_channel_estimators.py` utility defaults to
`wifi_iq_hackrf.bin` and expects interleaved-byte IQ input; check the producing
flowgraph before interpreting other standalone binary files.

## Collector and output notes

`utils/dataset_collector/` contains the GNU Radio collection flowgraphs and
`dataset_stats.py`. The latter summarizes annotated SigMF Wi-Fi frames and
channel metadata; it is not a decoder for unannotated raw I/Q.

Flowgraphs may read captures through paths such as
`data/raw_capture/stress_test.sigmf-data` and write packet captures or
diagnostics under `data/tmp_out/`. Treat files there as regenerable test output;
use a separate dataset/capture path for inputs that must be kept.
