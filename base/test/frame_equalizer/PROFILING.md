# Split Frame Equalizer Profiling

This file records the August 2026 AIE cycle profile and the temporary method
used to collect it. Profiling code is intentionally not kept in the kernels so
the production sources and metadata ABI remain readable.

## Last Result

The production CSI configuration was re-profiled on four 1024-sample tiles.
Each tile contained one 130-byte frequency-selective 16-QAM frame. Fifty
serialized NPU runs produced these averages. The counters are instrumented
measurements; shares are comparable within this build. The preprocess figures
below include both the interpolated `atan2` LUT and the linear sine/cosine LUT;
the equalization figures are unchanged from the prior measurement.

| Kernel or region | Cycles/tile | Share |
| --- | ---: | ---: |
| `frame_equalizer_preprocess` | 22,810 | 100.00% |
| Metadata copy | 28 | 0.12% |
| Sampling-ramp rotation | 6,787 | 29.76% |
| Pilot beta | 2,399 | 10.52% |
| Sampling-error update | 3,626 | 15.89% |
| Common-phase rotation | 5,829 | 25.56% |
| Data copies | 1,308 | 5.73% |
| Other preprocess work | 2,833 | 12.42% |
| `frame_equalizer_equalize` | 22,443 | 100.00% |
| LTS/channel/SNR | 8,901 | 39.66% |
| Equalization/channel power | 1,508 | 6.72% |
| Hard decisions | 5,494 | 24.48% |
| SIGNAL decode | 3,528 | 15.72% |
| Payload output | 1,122 | 5.00% |
| Control/tag state | 683 | 3.04% |
| Other equalization work | 1,207 | 5.38% |

Preprocess is now 1.02 times the equalization cost (down from 2.16x with
CORDIC). The two trigonometric rotations consume 55.32% of preprocess; pilot
beta and sampling-error tracking together consume 26.41%.

## Linear Sine/Cosine LUT

Only `frame_equalizer_preprocess` uses the quarter-wave linear approximation;
`sync_short` and `sync_long` keep the minimax sine/cosine implementation. The
table has 256 segments with signed Q15 coefficients, `step_bits=8`, and
`bias=128`. Four-access bank duplication occupies 4 KiB. One
`SinCosLinearLutQ15` object is constructed per tile invocation and reused for
all sampling-ramp and common-phase evaluations. Cosine uses the same sine LUT
with a quarter-turn phase offset.

The coefficient generator locally optimizes each segment over every integer
remainder. A standalone AIE sweep gave:

| Estimator | Maximum error | RMS error | Norm maximum | Cycles/pair |
| --- | ---: | ---: | ---: | ---: |
| Minimax | 3.05e-5 | 8.90e-6 | 6.10e-5 | 447.68 |
| 256-segment linear LUT | 3.97e-5 | 1.12e-5 | 7.52e-5 | 281.00 |

The sweep combines uniform, random, quadrant-edge, and coordinate-rounding-edge
phases; an exhaustive model check over all 65,536 quarter-wave coordinates is
bounded to one Q15 LSB. The LUT is 1.59 times faster in the standalone
benchmark. In the production profile it reduced sampling-ramp rotation from
9,348 to 6,787 cycles/tile and
common-phase rotation from 8,368 to 5,829 cycles/tile. Total preprocess cost
fell from 27,755 to 22,810 cycles/tile, a 17.8% reduction.

Synthetic flat-channel EVM remained 0.00595%. Frequency-selective EVM moved
from 0.00842% to 0.00898%, with identical hard decisions. Recorded-data RMS EVM
moved from 0.01023% to 0.01103%; maximum error moved from 0.0006069 to 0.0006312
and the same two hard-decision disagreements remained. A residual correction
was therefore not warranted.

The linked preprocess DM usage increased by exactly 4 KiB, from 6,459 to
10,555 bytes. PM moved from 8,944 to 9,008 bytes, maximum stack usage from 480
to 928 bytes, and allocated stack remained 2,560 bytes. The production graph
links successfully, and the sync kernels' linked PM/DM are unchanged.

## Interpolated Atan LUT

The `atan2` used by `pilot_beta` and `update_sampling_error` was swapped from the
24-stage CORDIC to a 512-entry interpolated LUT (`atan2_estimate_lut_q29`);
`sync_short` and `sync_long` keep the original CORDIC. The LUT holds 512 Q3.29
`atan(i/512)` entries plus a 256-entry Q2.30 reciprocal seed table, for 3 KiB;
`atan(i/512)` is linearly interpolated with Q15 slope and the reciprocal seed is
refined twice with Newton iteration to avoid scalar division.

A standalone AIE microbenchmark over a quadrant sweep, random full-range
`int32_t` pairs, signed endpoints, axes, diagonals, and `(0,0)` gave:

| Estimator | Maximum error | RMS error | Cycles/call |
| --- | ---: | ---: | ---: |
| 24-stage CORDIC | 1.24e-7 rad | 6.87e-8 rad | 1,186.8 |
| 512-entry interpolated LUT | 3.30e-7 rad | 1.79e-7 rad | 309.9 |

The LUT is 3.83 times faster per call while staying more than 15 times below the
existing 5e-6 rad acceptance limit. The same 50-run coarse-profile A/B that
produced the pre-sine-LUT preprocess figures compared both implementations on
identical binaries, counters, and the four-tile 130-byte 16-QAM workload:

| Region | CORDIC cycles/tile | LUT cycles/tile | Reduction |
| --- | ---: | ---: | ---: |
| `frame_equalizer_preprocess` | 48,454 | 27,755 | 42.7% |
| `pilot_beta` | 13,584 | 2,399 | 82.3% |
| `update_sampling_error` | 12,404 | 3,609 | 70.9% |

Synthetic EVM was unchanged: 0.00595% on the all-rate flat-channel test and
0.00842% on the frequency-selective tracked frame. Recorded-data RMS EVM moved
from 0.01012% to 0.01023%; maximum error stayed 0.0006069 and the same two
hard-decision disagreements remained. The LUT tables add 3 KiB to linked DM
(3,483 to 6,459 bytes) and 640 bytes to PM (8,304 to 8,944 bytes); the linker
discards them from `sync_short` and `sync_long`, whose CORDIC code and linked DM
are unchanged.

## Fine Preprocess Profile

An earlier temporary fine-grained run used the same workload and 50 serialized
runs before the sampling-ramp recurrence optimization. The extra cycle reads
were placed inside the 16-lane loops, so these values include measurement
overhead and are retained as optimization history, not current production
timing:

| Region | Instrumented cycles/tile | Share of instrumented preprocess |
| --- | ---: | ---: |
| Sampling-ramp rotation | 30,454 | 41.31% |
| Sampling-ramp oscillator generation | 24,961 | 33.86% |
| Sampling-ramp multiply/store | 1,858 | 2.52% |
| Pilot beta | 14,498 | 19.66% |
| Pilot preparation | 1,248 | 1.69% |
| Pilot `atan2` | 12,578 | 17.06% |
| Sampling-error update | 13,109 | 17.78% |
| Update correlation/extraction | 704 | 0.95% |
| Update phase/scaling | 11,682 | 15.85% |
| Common-phase rotation | 10,225 | 13.87% |
| Common oscillator generation | 6,881 | 9.33% |
| Common-phase multiply/store | 1,856 | 2.52% |

The optimization target was therefore the trigonometric/phase-estimation path,
not the vector complex multiply. The sampling-ramp oscillator was subsequently
changed to one trigonometric evaluation plus a 16-bin Q15 recurrence.

After replacing four absolute sampling-ramp oscillator evaluations with one
evaluation plus 16-bin Q15 recurrence, the same fine-profile method measured
`rotate_sampling_ramp` at 10,866 instrumented cycles/tile, down from 30,454
(about 64%). The optimized breakdown was 6,274 cycles for trigonometry, 833 for
recurrence, 1,536 for sample rotation/store, and 976 for setup. The corresponding
pre-sine-LUT full-profile sampling-ramp region was 10,581 cycles/tile.
Recorded-data RMS EVM
increased slightly from 0.00912% to 0.01012%.

### Historical Fine Preprocess Internals

The four preprocess functions were re-instrumented with cycle reads at region
boundaries (not inside the 16-lane loops) and measured with the same 50-run
four-tile 130-byte 16-QAM workload. Two independent 50-run sets agreed to
within one cycle per region.

| Region | Instrumented cycles/tile | Share of preprocess |
| --- | ---: | ---: |
| `frame_equalizer_preprocess` total | 52,569 | 100.00% |
| FFT metadata copy | 22 | 0.04% |
| `rotate_sampling_ramp` total | 10,869 | 20.68% |
| Sampling-ramp setup | 1,056 | 2.01% |
| Sampling-ramp trig evaluation | 6,240 | 11.87% |
| Sampling-ramp recurrence | 817 | 1.55% |
| Sampling-ramp rotate/store | 1,541 | 2.93% |
| `pilot_beta` total | 14,448 | 27.48% |
| Pilot preparation | 1,230 | 2.34% |
| Pilot `atan2` | 12,578 | 23.93% |
| `update_sampling_error` total | 12,982 | 24.70% |
| Update correlation/extraction | 704 | 1.34% |
| Update phase/scaling | 11,220 | 21.34% |
| `rotate_common_phase` total | 9,088 | 17.29% |
| Common oscillator generation | 6,880 | 13.09% |
| Common rotate/store | 1,552 | 2.95% |

The trigonometric/phase-estimation paths remain the whole story: pilot `atan2`
(12,578), sampling-update phase/scaling (11,220), sampling-ramp trig (6,240),
and common oscillator generation (6,880) together account for about 70% of the
instrumented preprocess total. The recurrence trick already shrank the
sampling-ramp trig work from four evaluations to one (24,961 -> 6,240); further
gains must attack `atan2`, the Q29 scaling division, or the common-oscillator
evaluation rather than the vector complex multiplies.

Do not benchmark with `DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1` when comparing the
production kernels. That test mode performs scalar LS normalization for every
payload symbol and makes payload output dominate the equalization profile.

## Repeating The Profile

Use `aie::tile::current().cycles()` and accumulate `uint64_t` deltas. Keep the
instrumentation temporary and measure one build after another; do not run NPU
programs concurrently.

Recommended preprocess regions:

1. Whole function
2. FFT metadata copy
3. `rotate_sampling_ramp`
4. `pilot_beta`
5. `update_sampling_error`
6. `rotate_common_phase`
7. Input/output symbol copies

Recommended equalization regions:

1. Whole function
2. LTS capture, channel estimate, and SNR
3. Equalization and channel-power generation
4. `hard_decision`
5. `decode_signal` plus `parse_signal`
6. Payload metadata and output stores
7. Tag traversal and skipped-symbol control flow

The profiling run temporarily appended eight preprocess counter words to
`frame_equalizer_prepared_tile_metadata`. The equalization kernel copied those
eight words plus its own eight counters into CSI entries 52 through 59 of
`metadata->tags[0]`. Each CSI entry contains two `int32_t` words, giving sixteen
host-visible counters without changing the output allocation. The fine
preprocess-internals run above instead reserved all sixteen words for the
preprocess regions listed under "Historical Fine Preprocess Internals".

For a new run:

1. Temporarily enlarge the prepared metadata type in both C++ and the two IRON
   graphs by sixteen `int32_t` words.
2. Bracket the regions above with `aie::tile::current().cycles()`.
3. Store preprocess counters in the temporary prepared-metadata tail.
4. Store both kernels' counters in CSI entries 52 through 59.
5. Build the standalone graph in production CSI mode with `JOBS=2`.
6. Generate a deterministic frame using helpers in `test_symbols.py` and invoke
   `DefaultNPURuntime.run()` serially at least 20 times.
7. Parse CSI entries 52 through 59 as sixteen signed `int32_t` values, average
   across runs and tiles, and report each region as a percentage of its kernel.
8. Remove all instrumentation and restore the 81-word prepared metadata ABI.
9. Rebuild and run `test_symbols.py` plus `compare_recorded_symbols.py` in its
   required symbol-output configuration.

Cycle reads and added stores perturb scheduling, so compare regions within the
same instrumented build. Use a coarse whole-kernel-only build if an accurate
absolute cycle count is needed.
