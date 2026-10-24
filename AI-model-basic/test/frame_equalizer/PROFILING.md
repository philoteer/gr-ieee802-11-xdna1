# Split Frame Equalizer Profiling

This file records the current AI-engine profile results. The method used to
collect them now lives at `../../../AGENTS.md` ("Profiling kernels"). Profiling
code is intentionally not kept in the kernels so the production sources and
metadata ABI remain readable.

## Current Engine (September 2026)

The deployed `ENGINE=AI` equalizer replaces the classic hard demapper with one
bias-free learned MLP per modulation (`3 -> 16 -> 16 -> bits`, ReLU after both
hidden layers, LSB-first bit logits). It requires the LS estimator and is
stateless after the LTS-derived LS estimate; decision-directed tracking is not
implemented yet.

Each 16-bin block is scaled to signed int32 Q23 (`kInputFractionalBits = 23`)
with one shared power-of-two factor. The input scale and both hidden-layer
shifts truncate, and the dense layers run under `rounding_mode::floor` with
`saturation_mode::saturate`. The exported shifts are `[[11, 9], [12, 8],
[12, 8], [12, 7]]` for BPSK, QPSK, 16-QAM, and 64-QAM. The 16x16 inner loop is
unrolled by 3 and the output-layer inner loop by 5; every other loop stays
rolled under `[[chess::no_unroll]]`.

Per-payload-symbol cycles on a deterministic frequency-selective frame with
eight payload symbols, mean of 20 serial NPU runs, instrumentation bracketing
only `equalize_symbol()`:

| Modulation | Classic | Learned AI (deployed) |
| --- | ---: | ---: |
| BPSK | 312 | 8,421 |
| QPSK | 308 | 8,837 |
| 16-QAM | 897 | 9,669 |
| 64-QAM | 1,007 | 10,501 |

The production (non-instrumented) `test/frame_equalizer` image links:

| Kernel | PM | DM | Maximum stack |
| --- | ---: | ---: | ---: |
| `frame_equalizer_preprocess` | 8,880 | 10,555 | 928 |
| `frame_equalizer_equalize` | 11,104 | 11,899 | 1,152 |

DM counts the linked data sections plus the allocated stack region (2,560 bytes
for preprocess and 4,096 bytes for equalize); maximum stack is the deepest call
chain in the linked ELF map.

## Loop Unroll Study (September 2026)

The learned model first shipped with `[[chess::no_unroll]]` on every dense-layer
loop. Removing those pragmas changed neither cycles nor linked PM, so they only
documented the compiler's default schedule. The alternative
`[[chess::unroll_loop(n)]]`, `[[chess::unroll_loop()]]`, and
`chess_flatten_loop` annotations from `aie_kernel_utils.h` were then swept with
the same deterministic frequency-selective eight-payload-symbol workload,
20 serial NPU runs, and the same temporary `equalize_symbol()` instrumentation:

| 16x16 inner | Output inner | BPSK | QPSK | 16-QAM | 64-QAM | PM | Stack |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rolled | rolled | 10,536 | 11,080 | 12,168 | 13,256 | 11,120 | 1,312 |
| 3 | 3 | 8,447 | 8,871 | 9,719 | 10,567 | 11,488 | 1,312 |
| 3 | 5 (deployed) | 8,421 | 8,837 | 9,669 | 10,501 | 11,536 | 1,312 |
| full | full | 10,418 | 10,858 | 11,738 | 12,618 | 13,056 | 2,816 |

PM and stack are from the linked instrumented image, so they are comparable
within the table but shifted by the counters. Findings:

1. Partial inner-loop unrolling is the only annotation that pays. Factor 3 on
   the 16x16 layer and 5 on the output layer cut the per-payload cost by about
   20% (BPSK 10,536 to 8,421 cycles, 64-QAM 13,256 to 10,501). Factor 2 hides
   too little latency, factor 4 leaves a remainder, and factors 5 and 15 on the
   16x16 layer increase accumulator and register pressure; full unrolling is
   slower than the deployed build and more than doubles the deepest stack.
2. `chess_flatten_loop` must sit between the loop header and its body and
   requires a manifest trip count, so it cannot apply to the runtime-bounded
   bit loop at all. Where it did apply it inflated PM to 15,856 bytes (96.8%)
   while still being slower than the rolled baseline.
3. Unrolling the first 3-to-16 layer or the outer neuron loops never helped, so
   those loops stay rolled with an explicit `[[chess::no_unroll]]`.

The deployed configuration is `AIE_LOOP_UNROLL(3)` on the 16x16 inner loop and
`AIE_LOOP_UNROLL(5)` on the output-layer inner loop, with every outer loop
rolled. At that revision the unrolling added 368 bytes of production PM (11,328
to 11,696) for the roughly 20% payload-cycle reduction. The all-rate symbol test
passes with the same transmitted-label decisions as the rolled build, and the
impaired-channel BER benchmark passes unchanged.

## Rounding And Saturation Mode Check (September 2026)

The dense layers run under `rounding_mode::floor` with
`saturation_mode::saturate`, and both the input block scale and the exported
hidden-layer shifts truncate. Python studies on the current 144,000-decision
test split (600 frames x 5 symbols x 48 data bins per modulation) and hardware
A/B builds measured whether those choices matter.

Re-running the integer reference with symmetric half-away-from-zero shifts
changed 0 BPSK, 1 QPSK, 4 16-QAM, and 15 64-QAM decisions out of 144,000 each,
with BER moving by less than 0.001 percentage points in either direction.
Converting with wrapping instead of saturating cannot change an outcome: the
largest shifted first- and second-layer activations stay below 2^20 and the
largest logit stays below 2^29, far from the int32 limits. Two hardware builds
confirmed the study on the impaired-channel benchmark (10 batches, four
modulations, including dropped SIGNAL frames); the `symmetric_inf` and
non-saturating variants produced byte-identical BER reports.

The input block scaling was studied the same way, because truncation there
would let `ai_scale_to_input()` use `aie::upshift`/`aie::downshift` and remove
`ai_round_shift()` and its scoped rounding mode entirely. A Python emulation of
rounding block scaling changed 0 BPSK, 0 QPSK, 1 16-QAM, and 1 64-QAM decision
out of 144,000, with 16-QAM BER moving from 2.9512% to 2.9514%. The simplified
hardware build passed the all-rate symbol test and produced a byte-identical
impaired-channel BER report.

Both stages now truncate. `quantize_inputs()` uses floor semantics so the
exported integer reference models the hardware, and the model was retrained and
re-exported: the calibration shifts are unchanged (`[[11, 9], [12, 8], [12, 8],
[12, 7]]`), validation passes, and quantized held-out BER is at or below the
classic hard demapper for 16-QAM and 64-QAM (2.9512% vs 3.0717% and 4.8433% vs
4.9865%). The simplifications removed 272 bytes of PM (11,696 to 11,424) with
DM 11,899 and the 1,280-byte maximum stack unchanged.

An interim input path also repaired nonzero lanes that truncated to zero
(`ai_preserve_nonzero_sign` plus a channel-power zero fix). Temporary hardware
counters over three impaired batches (16 tiles each, four modulations) counted
43,262 nonzero matched-filter lanes and zero repair events: raw values are
always large enough relative to their block maximum that truncating shifts do
not erase them. Removing the repair, its power correction, and the helper
produced a byte-identical impaired-channel BER report and passed the all-rate
symbol test, and the Python `quantize_inputs()` dropped the equivalent
correction. PM fell from 11,424 to 11,104 bytes and the maximum stack from
1,280 to 1,152, leaving DM at 11,899.

Rounding and saturation modes are therefore not an accuracy lever on the
deployed corpus. Floor is kept because it matches the int64 reference's
arithmetic shifts, so the exported model and the validated metrics stay
bit-exact, and `saturate` is kept as a guard: the shifts are calibrated on the
validation split, so activations from out-of-distribution frames can exceed the
calibration maxima and drive a conversion past int32, where saturating
preserves the sign that wrapping would corrupt.

## Repeating The Profile

The instrumentation, region list, counter ABI, and step-by-step procedure moved
to `../../../AGENTS.md` ("Profiling kernels"); keep that guide in sync when the
profiling method changes.
