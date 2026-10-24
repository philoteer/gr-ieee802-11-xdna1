# Standalone frame-equalizer stages

This keeps the existing file-separated design: prepare FFT input, run preprocess,
save its output, then run equalize independently. Stream conversion, metadata
adaptation, NPU execution, and tag recreation are C++ blocks. Python only launches
flowgraphs and examines validation results after processing has completed.

## Global installation

Install the modified `gr-mlir_aie` tree globally, then build the standalone
kernels. From this directory:

```bash
make -C preprocess all
make -C equalize all
python3 run_speedtests.py --validate
```

The flowgraphs use the globally installed `gnuradio.mlir_aie` Python module and
GRC block descriptions. No local Python path, launcher, or `LD_LIBRARY_PATH`
override is required.

The existing upstream FFT pipeline at `../until_fft/kernel/build/` must also be
built. The kernels retain their 1024-sample tiles, 16 tiles per invocation, and
existing preprocess/equalize source implementations.

## Flowgraphs

| Flowgraph | Generated script | Input / output |
| --- | --- | --- |
| `until_fft.grc` | `fft_dump.py` | `--dataset SIGMF_PREFIX --dump-path FFT_FILE` |
| `until_fft_test.grc` | `until_fft_test.py` | FFT file → native equalizer → decoder → PCAP |
| `preprocess_speedtest.grc` | `preprocess_speedtest.py` | FFT file → preprocess → null sink + input-rate log |
| `preprocess_dump.grc` | `preprocess_dump.py` | `--dump-path FFT_FILE --output-path PREPARED_FILE` |
| `until_preproc_test.grc` | `until_preproc_test.py` | Prepared file → standalone equalize → decoder → PCAP |
| `equalize_speedtest.grc` | `equalize_speedtest.py` | Prepared file → standalone equalize → null sink + input-rate log |
| `equalize_test.grc` | `equalize_test.py` | Prepared file → standalone equalize → decoder → PCAP |

All receive-test scripts accept `--dump-path`, `--freq`, and `--pcap-path`.
Preprocess scripts also accept `--freq`. All sources run once through EOF.
The runner reads the capture frequency from SigMF metadata.

Generate scripts using the global block descriptions:

```bash
grcc -o . preprocess_speedtest.grc
python3 preprocess_speedtest.py --dump-path tmp_data/fft.iqdata --freq 2.412e9
```

## Prepared metadata contract

The opt-in `preprocess=True` mode on
`mlir_aie_cpp_tagged_int64_to_int64` supplies the missing 64-word input metadata
buffer, patches the two frequency RTP markers, and reads 81 output metadata words
per tile. At each physical 1024-sample tile boundary it emits a
`frame_equalizer_prepared` PMT s32vector containing:

* 64 original FFT metadata words (length, count, and 31 offset/CFO pairs);
* 16 Q3.29 pilot-phase values, indexed by physical symbol position;
* one center-frequency snapshot in MHz.

`wifi_start` tags are recreated too. The native conversion blocks and GNU Radio
detached metadata files preserve the integer-vector tag. The opt-in
`prepared_input=True` mode on `mlir_aie_cpp_equalizer_test` requires it at each
tile boundary and copies it verbatim into the standalone equalize kernel's input.
That mode does not try to patch nonexistent RTP markers in the equalize-only
binary. It reads the standalone output's **2248-word padded metadata stride**,
while the payload still occupies 2242 words. Both modes default to off, retaining
the original generic/full-equalizer interfaces.

## Validation and later measurements

`--validate` defaults to **only `stress_test.sigmf`**. It checks PCAP/radiotap
framing, requires received packets, and compares decoded MAC-frame contents with
the native receiver. The observed one-packet variation is allowed (up to two
unmatched occurrences for a replacement or one missing packet in each replay).
No streaming Python diagnostics are added. All captures, headers, and validation
PCAPs are deleted when validation finishes, including after an exception.

When ready to measure both stages separately:

```bash
python3 run_speedtests.py
# Or select one capture:
python3 run_speedtests.py --dataset stress_test.sigmf
```

Benchmark logs go to `speedtest_logs/preprocess/` and `speedtest_logs/equalize/`.
Preparation output is console-only. The upstream FFT file is removed once its
prepared replacement is ready; the prepared file is removed after equalize.
`--work-dir` selects the disk-backed temporary directory. Subprocesses use
blocking waits without timeouts, and the runner does not monitor disk space.
