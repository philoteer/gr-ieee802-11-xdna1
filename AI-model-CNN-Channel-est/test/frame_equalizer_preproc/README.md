# Standalone frame-equalizer stages

The newer kernels split processing into three independent stages:
preprocess → channel estimation → equalization. Each stage builds its own NPU
image. Channel estimation has two retained builds: CNN alone, and CNN with the
MLP payload updater supplied by `channel_estimator/cnn_data_updater/mlp.h`.
The supplied estimator selector has no separate MLP-only algorithm.

## Build the three stages

From this directory, with `xchesscc_wrapper`, `aiecc` (or legacy `aiecc.py`), and the AIE Python
package available:

```bash
make -C preprocess all -j1 JOBS=1 NPU2=1
make -C channel_est all -j1 JOBS=1 NPU2=1 CNN_DATA_UPDATER=NULL
make -C channel_est all -j1 JOBS=1 NPU2=1 CNN_DATA_UPDATER=MLP
make -C equalize all -j1 JOBS=1 NPU2=1
```

Each directory below contains `final.xclbin`, `insts.bin`, `aie.mlir`, the kernel
object, and a `frame_equalizer_*_build_mode.txt` recording its configuration:

| Stage | Output directory |
| --- | --- |
| Preprocess | `preprocess/build/` |
| CNN channel estimation | `channel_est/build/cnn/` |
| CNN + MLP payload updater | `channel_est/build/mlp/` |
| Equalization | `equalize/build/` |

Omit `NPU2=1` for AIE2/XDNA1. Do not overlap builds; use the memory limits in
[`../alignment/README.md`](../alignment/README.md) on the local machine.

Channel estimation defaults to `EQUALIZER_ALGORITHM=CNN`,
`CNN_WEIGHTS=BY_FRAME`, and `CNN_DATA_UPDATER=NULL`. Both estimator builds above
use the same CNN weights; the second additionally enables the MLP updater.
`CNN_WEIGHTS=BY_CHANNEL` selects the alternative supplied weight set. Equalize
defaults to `ENGINE=CLASSIC`; it consumes the estimates rather than running an
estimator internally. Compile flags and `AIE_TILES_PER_BATCH` are recorded and
automatically invalidate stale builds when changed. Batch size defaults to 16
tiles of 1024 complex samples.

### Standalone kernel interfaces

Buffer sizes below are **per tile**; runtime buffers contain 16 tiles by default.
Samples are interleaved `int32` I/Q (2048 words per tile).

| Kernel | Inputs | Outputs |
| --- | --- | --- |
| Preprocess | Samples; 64-word FFT metadata | Prepared samples; 81-word prepared metadata |
| Channel estimation | Prepared samples; 81-word prepared metadata | 2168-word estimated metadata |
| Equalization | Prepared samples; 2168-word estimated metadata | Decision bytes; 2248-word padded frame metadata |

Estimated metadata contains 2048 CSI words, 81 prepared words, seven alignment
padding words, and 32 words of SNR values. Channel estimation does not copy the
prepared sample stream to an output: retain that stream for equalization.

Compilation and GRC receive-chain validation of both estimator variants have
been verified on `stress_test.sigmf`. The standalone graphs reserve one NPU
column each so the three stages can also run in the live receive-test graph.
The numerical kernel implementations are unchanged.

## Global installation

Build and install the updated native blocks (including Python bindings and GRC
descriptions) before running these flowgraphs normally:

```bash
cmake --build /home/piloteer/xdna_phy/gr-mlir_aie/gr-mlir_aie/build -j2
sudo cmake --install /home/piloteer/xdna_phy/gr-mlir_aie/gr-mlir_aie/build
```

The updated global bindings include the standalone channel-estimation and
estimated-input equalization interfaces. Project flowgraphs use ordinary global
imports and can be generated directly with `grcc`.

The flowgraphs use the globally installed `gnuradio.mlir_aie` Python module and
GRC block descriptions. No local Python path, launcher, or `LD_LIBRARY_PATH`
override is required.

The existing upstream FFT pipeline at `../until_fft/kernel/build/` must also be
built. The kernels retain their 1024-sample tiles, 16 tiles per invocation, and
existing preprocess/equalize source implementations.

## Three-stage GRC flowgraphs

| Flowgraph | Generated script | Input / output |
| --- | --- | --- |
| `until_fft.grc` | `fft_dump.py` | `--dataset SIGMF_PREFIX --dump-path FFT_FILE` |
| `until_fft_test.grc` | `until_fft_test.py` | FFT file → native equalizer → decoder → PCAP |
| `preprocess_speedtest.grc` | `preprocess_speedtest.py` | FFT file → preprocess → null sink + input-rate log |
| `preprocess_dump.grc` | `preprocess_dump.py` | `--dump-path FFT_FILE --output-path PREPARED_FILE` |
| `channel_est_speedtest.grc` | `channel_est_speedtest.py` | Prepared file → channel estimation → null sink + input-rate log |
| `channel_est_dump.grc` | `channel_est_dump.py` | Prepared file → channel estimation → estimated file |
| `channel_est_test.grc` | `channel_est_test.py` | FFT file → preprocess → channel estimation → equalize → decoder → PCAP |
| `until_preproc_test.grc` | `until_preproc_test.py` | Prepared file → channel estimation → equalize → decoder → PCAP |
| `equalize_speedtest.grc` | `equalize_speedtest.py` | Estimated file → standalone equalize → null sink + input-rate log |
| `equalize_test.grc` | `equalize_test.py` | Estimated file → standalone equalize → decoder → PCAP |

All receive-test scripts accept `--dump-path`, `--freq`, and `--pcap-path`.
Preprocess scripts also accept `--freq`. All sources run once through EOF.
The runner reads the capture frequency from SigMF metadata.
Channel-estimation graphs and `until_preproc_test.grc` accept `--estimator cnn`
or `--estimator mlp`. The dump graph additionally accepts `--output-path`.
All streaming, metadata conversion, and NPU execution are native blocks; `.py`
flowgraphs are generated by `grcc`, not maintained as hand-written DSP graphs.

Generate scripts using the global block descriptions:

```bash
grcc -o . preprocess_speedtest.grc
python3 preprocess_speedtest.py --dump-path tmp_data/fft.iqdata --freq 2.412e9
```

## Metadata handoffs

The opt-in `preprocess=True` mode on
`mlir_aie_cpp_tagged_int64_to_int64` supplies the missing 64-word input metadata
buffer, patches the two frequency RTP markers, and reads 81 output metadata words
per tile. At each physical 1024-sample tile boundary it emits a
`frame_equalizer_prepared` PMT s32vector containing:

* 64 original FFT metadata words (length, count, and 31 offset/CFO pairs);
* 16 Q3.29 pilot-phase values, indexed by physical symbol position;
* one center-frequency snapshot in MHz.

`wifi_start` tags are recreated too. The opt-in `channel_estimate=True` mode on
the same native block consumes the prepared tag, invokes the estimator's
three-buffer ABI, and passes the samples through unchanged while attaching a
`frame_equalizer_estimated` PMT s32vector of **2168 words** at every tile boundary.
The native conversion blocks and detached metadata files preserve both integer
vector tags. File samples remain numerical complex64, using the existing Q16.15
conversion blocks.

The opt-in `estimated_input=True` mode on `mlir_aie_cpp_equalizer_test` consumes
the estimated tag, validates its shape, and supplies it to the equalize kernel.
It bypasses frequency-RTP patching and uses the **2248-word padded output stride**
for the 2242-word payload. All new modes default off; the earlier generic,
preprocess, prepared-input, and full-equalizer interfaces remain available.

## Validation and later measurements

```bash
python3 run_speedtests.py --validate
# Select just one estimator or a different capture:
python3 run_speedtests.py --validate --estimator mlp --dataset 206.sigmf
```

`--validate` defaults to **only `stress_test.sigmf`**, with both CNN and CNN+MLP.
It runs all ten graphs, including the three standalone rate graphs. It checks
PCAP/radiotap framing, requires CRC-accepted packets, and reports decoded
MAC-frame overlap between live, prepared-file, and estimated-file receive paths.
Small kernel-level nondeterminism is reported without attempts to fix it in
the test harness. Learned-estimator differences from native LS are informational.
Missing/invalid PCAPs, subprocess failures, or replay paths with no common decoded
MAC frames remain failures. All temporary captures, headers, and PCAPs are removed
on completion or failure. Stage logs and `results.json` remain in
`validation_logs/<dataset>/`.

Verified stress-test packet counts:

| Path | CNN | CNN + MLP |
| --- | ---: | ---: |
| Live three-stage receive chain | 485403 | 485410 |
| Prepared-file replay | 485405 | 485407 |
| Estimated-file replay | 485404 | 485409 |

Native LS decoded 485399 packets. These counts are a validation result, not an
accuracy equivalence requirement for different learned estimators.

When ready to measure all three stages separately:

```bash
python3 run_speedtests.py
# Or select one capture:
python3 run_speedtests.py --dataset stress_test.sigmf
```

Benchmark logs go to `speedtest_logs/preprocess/`,
`speedtest_logs/channel_est/{cnn,mlp}/`, and `speedtest_logs/equalize/{cnn,mlp}/`.
For each dataset the sequence is FFT dump → preprocess speedtest → preprocess
dump → channel-estimation speedtest → channel-estimation dump → equalize
speedtest. The last three steps repeat for each selected estimator, then all
temporary files are deleted before the next capture. Preparation output is
console-only. The FFT dump is removed after preprocessing; each estimated dump
is removed after its equalize run, and the prepared file after both variants.
`--work-dir` selects the disk-backed temporary directory. Subprocesses use
blocking waits without timeouts, and the runner does not monitor disk space.
