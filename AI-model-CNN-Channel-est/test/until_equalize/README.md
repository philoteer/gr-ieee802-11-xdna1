# Receiver tests through frame equalization

This is the CNN counterpart of `base/test/until_equalize/`. It is a separate
nine-worker test graph, not a mode of the production ten-worker MAC receiver.
Its Makefile compiles the C++ sources under `../../../kernel/`; no kernel source
copies or production decoder switches are needed.

## Build

From this directory, using the MLIR-AIE environment:

```sh
# CSI, coded BER, SNR, and equalizer-only throughput tests:
make -C kernel -j1 JOBS=1 all EQUALIZER_ALGORITHM=CNN
# Per-symbol EVM:
make -C kernel -j1 JOBS=1 all EQUALIZER_ALGORITHM=CNN \
  DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1
```

Keep compilation serial and memory-limited as described in the receiver
[README](../../README.md). `NPU2=1`, model/weight/updater selections, FIFO
settings, and incremental configuration tracking work as in production.
The two builds have independent output directories.

## Flowgraphs

Run these from this directory so `kernel/build/` and `../../data/` resolve:

| Flowgraphs | Test build |
| --- | --- |
| `replaced-diagnostic.grc`, `replaced-diagnostic-cui.grc` | `DFRAME_EQUALIZER_OUTPUT_SYMBOLS=0` |
| `replaced-diagnostic-cui-snr.grc` | `DFRAME_EQUALIZER_OUTPUT_SYMBOLS=0` |
| `replaced-diagnostic-evm*.grc` | `DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1` |
| `speedtest.grc` | `DFRAME_EQUALIZER_OUTPUT_SYMBOLS=0` |
| `replaced.grc` (CPU MAC decoding after the NPU equalizer) | `DFRAME_EQUALIZER_OUTPUT_SYMBOLS=0` |
| `original-diagnostic*.grc`, `speedtest_802_11.grc`, `speedtest_80211.grc` | CPU reference implementations |

The diagnostic host/CPU decoder still receives packed carrier decisions and
per-frame or per-symbol metadata. This preserves the existing pre-FEC coded
BER and EVM measurements; decoded MAC bytes alone cannot reconstruct those
observations. Output metadata contains `OUTPUT_ABI=EQUALIZER_V1` in the build
marker. Runtime diagnostic checks reject production `MAC_V1` images and stale
markers. Archived images must be rebuilt with this test Makefile.

`speedtest_80211.grc` additionally requires the optional external
`gr-ieee80211`/`presiso` modules and their GRC block definitions.

The utility entry points remain in `../../utils/`:

```sh
/usr/bin/python3 ../../utils/benchmark_receivers.py --list-cases
/usr/bin/python3 ../../utils/compare_channel_estimators.py --help
```

Both resolve this directory's graphs and kernel, not the production image.
The estimator comparison accepts recorded `ci16_le`, 20 MS/s SigMF input.
Its build jobs default to one.

See [RESULTS.md](RESULTS.md) for graph-generation and real-hardware BER/EVM checks.
