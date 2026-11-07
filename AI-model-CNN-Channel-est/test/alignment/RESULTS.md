# Compatibility validation — 2026-11-07

Hardware: Ryzen AI 5 340, `RyzenAI-npu6` / AIE2P; XRT 2.25.00, firmware
1.1.2.64. Builds used the installed modern MLIR-AIE `aiecc`, Vitis AIE
Essentials, and one compiler job in a separate systemd service capped at
9 GiB RAM and 512 MiB swap. Successful AIE2P builds peaked at about 8 GiB
with no swap use. No simulator, retraining, or host-module replacement was used.

## Checks completed

* All 11 host regression tests pass, including all graph/device combinations,
  modern/legacy driver command selection, model/weight defines, effective-device
  build invalidation, generated LUT reproducibility, and integer MLP references
  for all four checked-in weight sets.
* `check_alignment.cc` compiles for both AIE2 and AIE2P, checking the real target
  API, CNN halo geometry, LUT layout, and unchanged metadata ABIs.
* The full nine-worker AIE2P receiver builds with
  `EQUALIZER_ALGORITHM=CNN CNN_DATA_UPDATER=MLP CNN_WEIGHTS=BY_FRAME ENGINE=CLASSIC`.
* The full AIE2/XDNA1 receiver also builds with that configuration, producing
  both `final.xclbin` and `insts.bin` in a separate directory. That build peaked
  at 5.4 GiB with no swap use; the AIE2P production artifacts remain untouched.
* The isolated AIE2P CNN matches the CPU integer reference exactly for all 64
  complex channel bins across eight scale/sign/SNR cases. The MLP updater also
  matches exactly across repeated symbols, pilot-polarity changes, normalization
  scales, and its high-SNR gate. Hardware parity uses the `BY_FRAME` weights.
* The exhaustive AIE2P sine/cosine test is bit-exact at all 262,144 shuffled
  phases against the independent scalar reference. Maximum absolute error is
  `3.71195291e-5`, below `5e-5`. Both atan2 implementations pass their bounds.

## Real recorded-IQ replay

The production CNN+MLP image was tested with the existing GNU Radio runner:

```sh
/usr/bin/python3 AI-model-basic/test/packet_decode/test_recorded.py \
  --kernel-dir AI-model-CNN-Channel-est/kernel/build \
  --samples 327680 --output <new-report-directory>
```

Input: the original 20 MHz `ci16_le` `stress_test.sigmf-data`, starting at sample
zero, with its SigMF frequency metadata. Results count only MAC-decoder PDUs
accepted after CRC validation.

| Complex samples | CNN+MLP CRC-valid packets | CPU LS packets | Packet comparison |
| --- | ---: | ---: | --- |
| 327,680, two host slots | 27 | 27 | Byte-identical packet multiset |
| 327,680, one host slot | 27 | 27 | Byte-identical packet multiset |
| 20,004,864 (~1 second), two slots | 5,166 | 5,166 | 5,165 matching occurrences; one missing and one additional |

The longer replay covers encoding IDs 0, 4, 5, 6, and 7. Learned channel
estimation is not required to reproduce LS decisions or its exact packet
multiset; isolated CPU/NPU parity is the numerical correctness check.

The AIE2P production `final.xclbin` SHA-256 is
`0fef3125ff9e003899227162fc1a1f046be94c08f20ca2e11609eacd2f7c354c`;
`insts.bin` is
`de156b99e0a342a51bd158ba131d63dc795feccc0bdeae3b13132c2deed9ee67`.
Logs, report JSON, decision dumps, and PCAPs are retained under
`/home/piloteer/xdna_phy/build-logs/cnn-*20261107*` (the production build log is
`cnn-aie2p-main-20261106.log`, started before midnight).

## Additional fixes and scope

The baseline port alone is insufficient: the CNN's old eight-int32 halo puts
aligned AIE2P stores at a 32-byte offset. The halo and row stride now use the
target's actual aligned-store requirement. No weight values or fixed-point
arithmetic were changed. Parity infrastructure was also made independent of
private NPZ training archives and PyTorch.

Legacy `aiecc.py` is covered by command-generation tests, not an installed-driver
build. The existing optional `ENGINE=AI` restriction to
`EQUALIZER_ALGORITHM=LS` remains unchanged; CNN validation uses `ENGINE=CLASSIC`.
No XDNA1 hardware execution was performed. Chess-linked ELF files still
produce the toolchain's existing warning that automatic stack-size validation
cannot reconstruct their call graphs; stack reservations were not reduced or
validation bypassed. These bounded tests are not full-recording, all-weight-set
hardware, or real-time throughput claims.
