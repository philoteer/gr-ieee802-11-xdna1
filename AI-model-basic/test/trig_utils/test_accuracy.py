#!/usr/bin/env python3
"""Compare fixed-point sine/cosine and atan2 implementations on the NPU."""

import argparse
from pathlib import Path
import runpy

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel
import numpy as np

SAMPLE_COUNT = 4096
TILE_COUNT = 16
TILE_SIZE = SAMPLE_COUNT // TILE_COUNT
Q29_SCALE = float(1 << 29)
Q15_SCALE = float(1 << 15)


def error_stats(actual, expected):
    error = actual - expected
    return float(np.max(np.abs(error))), float(np.sqrt(np.mean(error * error)))


def linear_reference_q15(phase, coefficients):
    """Scalar LUT arithmetic, independent of physical duplication and AIE API."""
    phase = phase.astype(np.int64)
    quadrant = phase >> 30
    offset = phase & ((1 << 30) - 1)
    folded = np.where(quadrant & 1, (1 << 30) - offset, offset)
    coordinate = np.clip((folded - (1 << 29) + (1 << 13)) >> 14,
                         -32768, 32767)
    step_bits = 16 - (len(coefficients).bit_length() - 1)
    index = (coordinate >> step_bits) + len(coefficients) // 2
    remainder = coordinate & ((1 << step_bits) - 1)
    slope = coefficients[index, 0]
    intercept = coefficients[index, 1]
    magnitude = np.clip(
        (intercept * (1 << step_bits) + slope * remainder
         + (1 << (step_bits - 1))) >> step_bits, -32768, 32767)
    return np.where(quadrant & 2, -magnitude, magnitude)


def check_exhaustive_linear(handle):
    generator = runpy.run_path(str(
        Path(__file__).resolve().parents[2] / "utils/gen_sin_linear_lut_q15.py"))
    coefficients = np.array(generator["optimize_coefficients"](256), dtype=np.int64)
    # Every int16 quarter-wave coordinate, across all quadrants. Shuffle lanes
    # to exercise unrelated LUT addresses simultaneously, not just broadcasts.
    phase = (np.arange(4 * 65536, dtype=np.uint64) << 14).astype(np.uint32)
    np.random.default_rng(80211).shuffle(phase)
    max_error = 0.0
    for start in range(0, len(phase), SAMPLE_COUNT):
        batch = phase[start:start + SAMPLE_COUNT]
        packed_in = np.zeros((TILE_COUNT, 3, TILE_SIZE), dtype=np.int32)
        packed_in[:, 0, :] = batch.view(np.int32).reshape(TILE_COUNT, TILE_SIZE)
        packed_out = iron.zeros(10 * SAMPLE_COUNT, dtype=np.int32)
        DefaultNPURuntime.run(handle, [
            iron.tensor(packed_in.reshape(-1), dtype=np.int32), packed_out])
        result = packed_out.numpy().reshape(TILE_COUNT, 10, TILE_SIZE)
        for channel, quarter_turn in ((2, 0), (3, 1 << 30)):
            shifted = ((batch.astype(np.uint64) + quarter_turn)
                       & ((1 << 32) - 1)).astype(np.uint32)
            actual = result[:, channel, :].reshape(-1).astype(np.int64)
            expected = linear_reference_q15(shifted, coefficients)
            if not np.array_equal(actual, expected):
                mismatch = np.flatnonzero(actual != expected)[0]
                raise SystemExit(
                    f"linear LUT mismatch at phase={int(batch[mismatch]):#010x}, "
                    f"channel={channel}: actual={actual[mismatch]}, "
                    f"expected={expected[mismatch]}")
            angle = shifted.astype(np.float64) * (2.0 * np.pi / (1 << 32))
            max_error = max(max_error, float(np.max(np.abs(
                actual / Q15_SCALE - np.sin(angle)))))
    print(f"EXHAUSTIVE LINEAR: {len(phase)} phases, sin and cos bit-exact; "
          f"max_abs={max_error:.9g}")
    if max_error > 5.0e-5:
        raise SystemExit("exhaustive linear sine/cosine exceeded its acceptance bound")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xclbin", type=Path, default=Path("build/final.xclbin"))
    parser.add_argument("--insts", type=Path, default=Path("build/insts.bin"))
    parser.add_argument("--exhaustive-linear", action="store_true",
                        help="also check every int16 LUT coordinate in all quadrants")
    args = parser.parse_args()

    # Half the vectors sweep every quadrant at fixed magnitude. The rest cover
    # arbitrary scales plus signed endpoints and axis-adjacent cases.
    sweep_count = SAMPLE_COUNT // 2
    angle = np.linspace(-np.pi, np.pi, sweep_count, endpoint=True)
    magnitude = 1 << 30
    y = np.rint(magnitude * np.sin(angle)).astype(np.int32)
    x = np.rint(magnitude * np.cos(angle)).astype(np.int32)
    rng = np.random.default_rng(80211)
    random_count = SAMPLE_COUNT - sweep_count
    y = np.concatenate((y, rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max,
        random_count, dtype=np.int32)))
    x = np.concatenate((x, rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max,
        random_count, dtype=np.int32)))
    edges = np.array([
        [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0],
        [1, 1], [-1, 1], [1, -1], [-1, -1],
        [np.iinfo(np.int32).min, 1],
        [1, np.iinfo(np.int32).min],
        [np.iinfo(np.int32).min, np.iinfo(np.int32).min],
    ], dtype=np.int32)
    y[-len(edges):] = edges[:, 0]
    x[-len(edges):] = edges[:, 1]

    phase_sweep = (np.arange(sweep_count, dtype=np.uint64) *
                   (1 << 32) // sweep_count).astype(np.uint32)
    phase = np.concatenate((phase_sweep, rng.integers(
        0, 1 << 32, random_count, dtype=np.uint32)))
    # Exercise quadrant folds and Q30 -> Q16 coordinate rounding boundaries,
    # in addition to random interpolation remainders.
    phase_edges = np.array([
        0, 1, (1 << 13) - 1, 1 << 13, (1 << 13) + 1,
        (1 << 14) - 1, 1 << 14, (1 << 14) + 1,
        (1 << 30) - 1, 1 << 30, (1 << 30) + 1,
        (1 << 31) - 1, 1 << 31, (1 << 31) + 1,
        (3 << 30) - 1, 3 << 30, (3 << 30) + 1,
        (1 << 32) - 1,
    ], dtype=np.uint32)
    phase[-len(phase_edges):] = phase_edges
    packed_in = np.stack((phase.view(np.int32), y, x)).reshape(
        3, TILE_COUNT, TILE_SIZE)
    packed_in = packed_in.transpose(1, 0, 2).reshape(-1)
    packed_out = iron.zeros(10 * SAMPLE_COUNT, dtype=np.int32)
    handle = DefaultNPURuntime.load(NPUKernel(
        str(args.xclbin), str(args.insts), kernel_name="MLIR_AIE"))
    DefaultNPURuntime.run(handle, [
        iron.tensor(packed_in, dtype=np.int32), packed_out])
    result = packed_out.numpy().reshape(
        TILE_COUNT, 10, TILE_SIZE).transpose(1, 0, 2)
    sin_minimax = result[0].reshape(-1).astype(np.float64) / Q15_SCALE
    cos_minimax = result[1].reshape(-1).astype(np.float64) / Q15_SCALE
    sin_linear = result[2].reshape(-1).astype(np.float64) / Q15_SCALE
    cos_linear = result[3].reshape(-1).astype(np.float64) / Q15_SCALE
    cordic = result[4].reshape(-1).astype(np.float64) / Q29_SCALE
    lut = result[5].reshape(-1).astype(np.float64) / Q29_SCALE
    expected = np.arctan2(y.astype(np.float64), x.astype(np.float64))
    expected[(x == 0) & (y == 0)] = 0.0
    cordic_max, cordic_rms = error_stats(cordic, expected)
    lut_max, lut_rms = error_stats(lut, expected)
    reference_angle = phase.astype(np.float64) * (2.0 * np.pi / (1 << 32))
    reference_sin = np.sin(reference_angle)
    reference_cos = np.cos(reference_angle)
    minimax_sin_max, minimax_sin_rms = error_stats(sin_minimax, reference_sin)
    minimax_cos_max, minimax_cos_rms = error_stats(cos_minimax, reference_cos)
    linear_sin_max, linear_sin_rms = error_stats(sin_linear, reference_sin)
    linear_cos_max, linear_cos_rms = error_stats(cos_linear, reference_cos)
    minimax_norm = np.max(np.abs(sin_minimax**2 + cos_minimax**2 - 1.0))
    linear_norm = np.max(np.abs(sin_linear**2 + cos_linear**2 - 1.0))
    vector_calls = TILE_SIZE / 16
    minimax_cycles = result[6, :, 0].astype(np.float64) / vector_calls
    linear_cycles = result[7, :, 0].astype(np.float64) / vector_calls
    cordic_cycles = result[8, :, 0].astype(np.float64) / TILE_SIZE
    lut_cycles = result[9, :, 0].astype(np.float64) / TILE_SIZE
    print(f"MINIMAX sin/cos: max_abs={max(minimax_sin_max, minimax_cos_max):.9g} "
          f"rms={max(minimax_sin_rms, minimax_cos_rms):.9g} "
          f"norm_max={minimax_norm:.9g} cycles/pair={minimax_cycles.mean():.2f}")
    print(f"LINEAR  sin/cos: max_abs={max(linear_sin_max, linear_cos_max):.9g} "
          f"rms={max(linear_sin_rms, linear_cos_rms):.9g} "
          f"norm_max={linear_norm:.9g} cycles/pair={linear_cycles.mean():.2f}")
    print(f"sin/cos speedup: {minimax_cycles.mean() / linear_cycles.mean():.3f}x")
    print(f"CORDIC: max_abs={cordic_max:.9g} rad, rms={cordic_rms:.9g} rad, "
          f"cycles/call={cordic_cycles.mean():.2f}")
    print(f"LUT:    max_abs={lut_max:.9g} rad, rms={lut_rms:.9g} rad, "
          f"cycles/call={lut_cycles.mean():.2f}")
    print(f"speedup: {cordic_cycles.mean() / lut_cycles.mean():.3f}x")
    if minimax_sin_max > 5.0e-5 or minimax_cos_max > 5.0e-5:
        raise SystemExit("minimax sine/cosine exceeded its acceptance bound")
    if linear_sin_max > 5.0e-5 or linear_cos_max > 5.0e-5:
        raise SystemExit("linear-LUT sine/cosine exceeded its acceptance bound")
    if cordic_max > 5.0e-6 or lut_max > 5.0e-6:
        raise SystemExit("atan2 accuracy exceeded its acceptance bound")
    if args.exhaustive_linear:
        check_exhaustive_linear(handle)


if __name__ == "__main__":
    main()
