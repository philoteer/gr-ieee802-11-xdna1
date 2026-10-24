#!/usr/bin/env python3
"""Generate the aie_atan_lut_q29.h header from configurable table parameters.

Produces a C header containing:
  - kAtanLutQ29: signed Q3.29 atan(i/N) for i = 0..N-1
  - kReciprocalSeedQ30: Q2.30 reciprocal seeds for 2^B index bits

Examples:
    # Default 512-entry atan, 256-entry reciprocal seed (matches current header)
    python3 gen_atan_lut.py > ../kernel/aie_atan_lut_q29.h

    # Smaller table: 256-entry atan, 128-entry seed
    python3 gen_atan_lut.py --atan-size 256 --seed-bits 7 > /tmp/test.h

    # Larger table: 1024-entry atan, 512-entry seed
    python3 gen_atan_lut.py --atan-size 1024 --seed-bits 9
"""

import argparse
import math
import sys
from pathlib import Path

Q29 = 1 << 29
Q30 = 1 << 30


def gen_atan_table(size: int) -> list[int]:
    """atan(i / size) in signed Q3.29 radians, i = 0..size-1."""
    return [round(math.atan(i / size) * Q29) for i in range(size)]


def gen_reciprocal_seed_table(seed_bits: int) -> list[int]:
    """Q2.30 reciprocal seeds for normalized unsigned Q1.31 denominators.

    For each index j, the seed approximates 1/major in Q2.30, where major
    (a normalized Q1.31 value) has its top (seed_bits + 1) bits equal to
    (1 << seed_bits) + j.  Two Newton refinements recover full accuracy.
    """
    count = 1 << seed_bits
    return [round(2 ** (31 + seed_bits) / (2 ** (seed_bits + 1) + 2 * j + 1))
            for j in range(count)]


def format_table(name: str, values: list[int], signed: bool = True) -> list[str]:
    """Format a C array declaration with alignas(32)."""
    max_val = max(abs(v) for v in values)
    if signed and max_val < (1 << 31):
        c_type = "int32_t"
    elif not signed and max_val < (1 << 32):
        c_type = "uint32_t"
    else:
        raise ValueError(f"values out of range for {'int32_t' if signed else 'uint32_t'}")

    lines = [f"alignas(32) static const {c_type} {name}[{len(values)}] = {{"]
    row: list[str] = []
    formatted: list[str] = []
    for v in values:
        row.append(str(v))
        if len(row) == 8:
            formatted.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        formatted.append("    " + ", ".join(row) + ",")
    lines.extend(formatted)
    lines.append("};")
    return lines


def generate_header(atan_size: int, seed_bits: int) -> str:
    atan_table = gen_atan_table(atan_size)
    seed_table = gen_reciprocal_seed_table(seed_bits)

    atan_bytes = atan_size * 4
    seed_bytes = len(seed_table) * 4
    total_bytes = atan_bytes + seed_bytes

    # Spell out small numbers in the seed comment to match the reference header.
    word = {8: "eight", 7: "seven", 9: "nine", 6: "six"}.get(seed_bits, str(seed_bits))

    parts = [
        "#ifndef AIE_ATAN_LUT_Q29_H",
        "#define AIE_ATAN_LUT_Q29_H",
        "",
        "#include <stdint.h>",
        "",
        f"// atan(i / {atan_size}) in signed Q3.29 radians for i=0..{atan_size - 1}. "
        "atan(1) is kept as",
        f"// the existing pi/4 constant, giving {atan_size} interpolation "
        f"intervals with a",
        f"// {atan_bytes // 1024} KiB table.",
        *format_table("kAtanLutQ29", atan_table, signed=True),
        "",
        "// Q2.30 reciprocal seeds for normalized unsigned Q1.31 denominators in",
        f"// [1, 2). The top {word} fractional bits select a midpoint seed; "
        "two Newton",
        "// refinements in atan2_estimate_lut_q29 recover the Q24 slope accurately.",
        *format_table("kReciprocalSeedQ30", seed_table, signed=False),
        "",
        "#endif",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aie_atan_lut_q29.h with configurable table sizes.",
    )
    parser.add_argument(
        "--atan-size", type=int, default=512,
        help="Number of atan LUT entries (default: 512)",
    )
    parser.add_argument(
        "--seed-bits", type=int, default=8,
        help="Index bits for reciprocal seed table, giving 2^B entries (default: 8)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    if args.atan_size < 1:
        parser.error("--atan-size must be >= 1")
    if args.seed_bits < 1:
        parser.error("--seed-bits must be >= 1")

    header = generate_header(args.atan_size, args.seed_bits)

    if args.output:
        Path(args.output).write_text(header)
        atan_bytes = args.atan_size * 4
        seed_bytes = (1 << args.seed_bits) * 4
        print(f"Wrote {args.output}: atan={args.atan_size} entries "
              f"({atan_bytes} B), seed={1 << args.seed_bits} entries "
              f"({seed_bytes} B), total={atan_bytes + seed_bytes} B",
              file=sys.stderr)
    else:
        sys.stdout.write(header)


if __name__ == "__main__":
    main()
