#!/usr/bin/env python3
    """Generate the atan/reciprocal section of kernel/common/aie_trig_luts.h.

Produces the atan portion of the combined trig LUT header, without an include
guard or includes, so the output can replace that section in
    kernel/common/aie_trig_luts.h.  Two tables are emitted:

    - ``kAtanLutQ29``: signed Q3.29 ``atan(i/N)`` for i = 0..N-1.  The table
      covers the first octant only; ``atan(1)`` is kept as the separate
      ``pi/4`` constant in the kernel.
    - ``kReciprocalSeedQ30``: Q2.30 reciprocal seeds for the top ``B`` index
      bits of a normalized Q1.31 denominator, used by the kernel's atan2
      estimator before Newton refinement.

The generated text is byte-for-byte reproducible so it can be diffed against
the checked-in header (see ``replicate_frame_equalizer_constants.py`` checks).

Examples:
    # Default 512-entry atan, 256-entry reciprocal seed (matches current header)
    python3 gen_atan_lut.py

    # Smaller table: 256-entry atan, 128-entry seed
    python3 gen_atan_lut.py --atan-size 256 --seed-bits 7

    # Larger table: 1024-entry atan, 512-entry seed
    python3 gen_atan_lut.py --atan-size 1024 --seed-bits 9

    # Write directly to a file (progress summary goes to stderr)
    python3 gen_atan_lut.py -o /tmp/atan_section.h
"""

import argparse
import math
import sys
from pathlib import Path


# Fixed-point scale factors: one raw unit equals 2**-29 (atan) or 2**-30
# (reciprocal seed) of the represented real value.
Q29 = 1 << 29
Q30 = 1 << 30


def gen_atan_table(size: int) -> list[int]:
    """Build the signed Q3.29 first-octant atan lookup table.

    Entry ``i`` holds ``round(atan(i / size) * 2**29)``.  The sampling grid
    runs over i = 0..size-1, so the largest tabulated angle is just below
    ``atan(1) = pi/4``; the kernel handles the endpoint with its existing
    ``pi/4`` constant and interpolates (or reflects) for the full circle.
    The unit slope of atan near zero means the sample spacing (1/size) also
    bounds the interpolation error as roughly ``1 / (2 * size**2)`` radians.

    Args:
        size: Number of table entries, one per interpolation interval.  The
            kernel expects a power of two (512 by default) so the index math
            can use shifts.

    Returns:
        A list of ``size`` signed integers in the Q3.29 range; ``atan`` on
        this grid spans ``[0, pi/4)``, well inside the ``int32_t`` limits.
    """
    return [round(math.atan(i / size) * Q29) for i in range(size)]


def gen_reciprocal_seed_table(seed_bits: int) -> list[int]:
    """Build Q2.30 reciprocal seeds for normalized unsigned Q1.31 denominators.

    For each index j, the seed approximates ``1/major`` in Q2.30, where
    ``major`` (the top ``seed_bits + 1`` bits of a normalized Q1.31 value,
    i.e. the raw value right-shifted by ``31 - seed_bits``) equals
    ``(1 << seed_bits) + j``.  The seed targets the segment *midpoint*
    ``major + 0.5`` rather than ``major`` itself, because 1/x is convex and
    the midpoint minimizes the maximum error over the segment:

        seed = round(2**(31 + seed_bits) / (2**(seed_bits + 1) + 2*j + 1))

    The result is only a seed: two Newton refinements in
    ``atan2_estimate_lut_q29`` recover the full Q24 slope accuracy.

    Args:
        seed_bits: Number of fractional index bits; the table has
            ``1 << seed_bits`` entries (8 by default, i.e. 256 entries).
            Larger values tighten the initial guess and reduce Newton
            iterations at the cost of 4 bytes per entry.

    Returns:
        A list of ``1 << seed_bits`` non-negative integers in Q2.30.  The
        reciprocal of a denominator in [1, 2) lies in (0.5, 1], so all seeds
        are positive and fit comfortably in ``uint32_t``.
    """
    count = 1 << seed_bits
    return [round(2 ** (31 + seed_bits) / (2 ** (seed_bits + 1) + 2 * j + 1))
            for j in range(count)]


def format_table(name: str, values: list[int], signed: bool = True) -> list[str]:
    """Format a scalar table as an ``alignas(32)`` C array definition.

    The smallest sufficient 32-bit element type is selected from the values,
    matching the AIE vector load requirements (32-byte alignment, 8 int32
    lanes per vector).  Eight values per source line keep the generated
    header readable and diffable.

    Args:
        name: C identifier for the generated array.
        values: Integer values to emit, in table order.
        signed: When true emit ``int32_t`` (Q3.29-style signed tables);
            when false emit ``uint32_t`` (the reciprocal seeds).

    Returns:
        The C declaration as a list of lines, from ``alignas(32) ... = {``
        through the closing ``};``.  The list carries no trailing newline.

    Raises:
        ValueError: If any value does not fit in the selected 32-bit type.
    """
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


def generate_section(atan_size: int, seed_bits: int) -> str:
    """Render the complete atan/reciprocal section of ``aie_trig_luts.h``.

    Assembles the explanatory comments and both table definitions into the
    exact text currently checked in under ``kernel/common/aie_trig_luts.h``.  No
    include guard or ``#include`` lines are emitted: the caller is expected
    to splice this section into the combined header.

    Args:
        atan_size: Number of atan entries, as accepted by
            :func:`gen_atan_table`.
        seed_bits: Reciprocal seed index bits, as accepted by
            :func:`gen_reciprocal_seed_table`.

    Returns:
        The generated C section as a single newline-joined string, ending in
        a trailing blank line.
    """
    atan_table = gen_atan_table(atan_size)
    seed_table = gen_reciprocal_seed_table(seed_bits)

    atan_bytes = atan_size * 4
    seed_bytes = len(seed_table) * 4
    total_bytes = atan_bytes + seed_bytes

    # Spell out small numbers in the seed comment to match the reference header.
    word = {8: "eight", 7: "seven", 9: "nine", 6: "six"}.get(seed_bits, str(seed_bits))

    parts = [
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
    ]
    return "\n".join(parts)


def main() -> None:
    """Command-line entry point for the atan LUT generator.

    Parses ``--atan-size`` and ``--seed-bits``, renders the section, and
    writes it to ``--output`` or stdout.  When writing to a file, a one-line
    size summary (entry counts and byte totals) is printed to stderr so that
    stdout stays reserved for the generated C text.
    """
    parser = argparse.ArgumentParser(
        description="Generate the atan section for aie_trig_luts.h with "
                    "configurable table sizes.",
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

    section = generate_section(args.atan_size, args.seed_bits)

    if args.output:
        Path(args.output).write_text(section)
        atan_bytes = args.atan_size * 4
        seed_bytes = (1 << args.seed_bits) * 4
        print(f"Wrote {args.output}: atan={args.atan_size} entries "
              f"({atan_bytes} B), seed={1 << args.seed_bits} entries "
              f"({seed_bytes} B), total={atan_bytes + seed_bytes} B",
              file=sys.stderr)
    else:
        # Write without an extra newline: generate_section already ends the
        # text with a blank line.
        sys.stdout.write(section)


if __name__ == "__main__":
    main()
