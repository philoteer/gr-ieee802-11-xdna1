#!/usr/bin/env python3
"""Generate the four-copy int16 LUT used by AIE linear_approx sine.

The input is a signed 16-bit quarter-wave coordinate. A 256-segment table uses
step_bits=8 and bias=128, covering the complete [-32768, 32767] range without
discarding a phase bit. Each logical entry stores an integer segment slope and
a Q15 offset. The generator searches nearby integer coefficient pairs and
chooses the pair with the smallest maximum Q15 error over the segment.

AMD's four-access LUT intrinsic requires four physical copies: two interleaved
copies in each of the AB and CD arrays. Thus an N-entry int16 slope/offset table
occupies N * 16 bytes (4 KiB for the default 256 entries).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


Q15_SCALE = 1 << 15


def q15_sin(coordinate: int) -> int:
    fraction = (coordinate + (1 << 15)) / float(1 << 16)
    value = round(math.sin(0.5 * math.pi * fraction) * Q15_SCALE)
    return min(32767, max(-32768, value))


def interpolate(offset: int, slope: int, remainder: int, step_bits: int) -> int:
    # All first-quadrant values are nonnegative; positive-infinity rounding is
    # equivalent to adding half an output unit before the fixed-point shift.
    value = (offset * (1 << step_bits) + slope * remainder
             + (1 << (step_bits - 1))) >> step_bits
    return min(32767, max(-32768, value))


def optimize_coefficients(entries: int) -> list[tuple[int, int]]:
    log_entries = entries.bit_length() - 1
    step_bits = 16 - log_entries
    step = 1 << step_bits
    coefficients = []
    for index in range(entries):
        start = -(1 << 15) + index * step
        expected = [q15_sin(start + remainder) for remainder in range(step)]
        nominal_offset = expected[0]
        next_value = q15_sin(min((1 << 15) - 1, start + step))
        nominal_slope = next_value - nominal_offset

        best = None
        for offset in range(nominal_offset - 2, nominal_offset + 3):
            for slope in range(nominal_slope - 3, nominal_slope + 4):
                errors = [
                    interpolate(offset, slope, remainder, step_bits) - target
                    for remainder, target in enumerate(expected)
                ]
                score = (max(abs(error) for error in errors),
                         sum(error * error for error in errors),
                         abs(offset - nominal_offset) + abs(slope - nominal_slope))
                if best is None or score < best[0]:
                    best = (score, slope, offset)
        assert best is not None
        _, slope, offset = best
        if not (-32768 <= slope <= 32767 and -32768 <= offset <= 32767):
            raise ValueError(f"coefficient overflow at entry {index}")
        coefficients.append((slope, offset))
    return coefficients


def duplicate_for_four_accesses(coefficients: list[tuple[int, int]]) -> list[int]:
    if len(coefficients) % 4:
        raise ValueError("entry count must be divisible by four")
    values = []
    for base in range(0, len(coefficients), 4):
        bank_width = [value for pair in coefficients[base:base + 4] for value in pair]
        values.extend(bank_width)
        values.extend(bank_width)
    return values


def format_array(name: str, values: list[int]) -> list[str]:
    lines = [f"alignas(32) static const int16_t {name}[{len(values)}] = {{"]
    for base in range(0, len(values), 12):
        lines.append("    " + ", ".join(str(value) for value in values[base:base + 12]) + ",")
    lines.append("};")
    return lines


def generate_header(entries: int) -> str:
    if entries < 16 or entries > 8192 or entries & (entries - 1):
        raise ValueError("entries must be a power of two in [16, 8192]")
    step_bits = 16 - (entries.bit_length() - 1)
    if step_bits < 3:
        raise ValueError("int16 linear_approx requires step_bits >= 3")
    coefficients = optimize_coefficients(entries)
    duplicated = duplicate_for_four_accesses(coefficients)
    physical_bytes = 2 * len(duplicated) * 2

    lines = [
        "#ifndef AIE_SIN_LINEAR_LUT_Q15_H",
        "#define AIE_SIN_LINEAR_LUT_Q15_H",
        "",
        "#include <stdint.h>",
        "",
        f"constexpr unsigned kSinLinearLutEntries = {entries};",
        f"constexpr unsigned kSinLinearLutStepBits = {step_bits};",
        f"constexpr int kSinLinearLutBias = {entries // 2};",
        "",
        f"// Q15 sine offsets and integer segment slopes for {entries} quarter-wave",
        f"// segments. Four-access bank duplication makes the physical table {physical_bytes} bytes.",
        *format_array("kSinLinearLutAb", duplicated),
        "",
        *format_array("kSinLinearLutCd", duplicated),
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=256)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    header = generate_header(args.entries)
    if args.output:
        args.output.write_text(header)
    else:
        print(header, end="")


if __name__ == "__main__":
    main()
