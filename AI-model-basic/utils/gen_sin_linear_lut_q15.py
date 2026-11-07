#!/usr/bin/env python3
"""Generate the sine section of kernel/common/aie_trig_luts.h.

Emits the four-copy int16 LUT used by the AIE ``linear_approx`` sine, without
an include guard or includes, so the output can replace that section in
kernel/common/aie_trig_luts.h.

Table model:
    The input is a signed 16-bit quarter-wave coordinate spanning
    [-32768, 32767], where the segment index and the intra-segment remainder
    double as an unsigned phase.  A 256-segment table uses ``step_bits=8`` and
    ``bias=128``, covering the complete range without discarding a phase bit.
    Each logical entry stores an integer segment slope and a Q15 offset; the
    hardware evaluates ``offset + slope * remainder / 2**step_bits``.

    The generator searches nearby integer coefficient pairs and chooses the
    pair with the smallest maximum Q15 error over the segment, breaking ties
    by total squared error and then by coefficient perturbation.  This keeps
    the worst-case error near half an LSB instead of the larger error a naive
    slope from the segment endpoints would give.

Physical layout:
    AMD's four-access LUT intrinsic requires four physical copies: two
    interleaved copies in each of the ``AB`` and ``CD`` arrays. Repetition is
    at the memory-bank width: 128 bits on AIE2, 256 bits on AIE2P. Thus an
    N-entry int16 slope/offset table occupies ``N * 16`` bytes (4 KiB for the
    default 256 entries).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


# Scale of the Q15 fixed-point sine samples (one raw unit = 2**-15).
Q15_SCALE = 1 << 15


def q15_sin(coordinate: int) -> int:
    """Evaluate the ideal sine of a signed 16-bit quarter-wave coordinate.

    The signed coordinate ``c`` in [-32768, 32767] is first biased into the
    unsigned range [0, 65536) and then normalized to the fraction
    ``(c + 32768) / 65536`` in [0, 1).  That fraction indexes a quarter
    period, so the ideal value is ``sin(pi/2 * fraction)``, rounded to Q15.
    The result is clamped to the signed int16 range because the endpoint
    rounds up to 32768, which is not representable.

    Args:
        coordinate: Signed 16-bit quarter-wave coordinate.

    Returns:
        The ideal ``sin(pi/2 * fraction)`` value in Q15, clamped to
        [-32768, 32767].
    """
    fraction = (coordinate + (1 << 15)) / float(1 << 16)
    value = round(math.sin(0.5 * math.pi * fraction) * Q15_SCALE)
    return min(32767, max(-32768, value))


def interpolate(offset: int, slope: int, remainder: int, step_bits: int) -> int:
    """Evaluate one table segment exactly as the kernel does.

    Computes ``(offset * 2**step_bits + slope * remainder) / 2**step_bits``
    with round-half-up behavior and clamps the result to the signed int16
    range.  The rounding bias is added before the shift; this is the standard
    positive-infinity rounding idiom and is exact here because all
    first-quadrant sine values are nonnegative.

    Args:
        offset: Segment start value (the table's Q15 offset coefficient).
        slope: Integer change in value across the whole segment.
        remainder: Position within the segment, in [0, ``2**step_bits``).
        step_bits: Number of coordinate bits per segment; the segment width
            is ``2**step_bits``.

    Returns:
        The interpolated sample, clamped to [-32768, 32767].
    """
    # All first-quadrant values are nonnegative; positive-infinity rounding is
    # equivalent to adding half an output unit before the fixed-point shift.
    value = (offset * (1 << step_bits) + slope * remainder
             + (1 << (step_bits - 1))) >> step_bits
    return min(32767, max(-32768, value))


def optimize_coefficients(entries: int) -> list[tuple[int, int]]:
    """Fit int16 (slope, offset) pairs for every quarter-wave segment.

    The signed 16-bit coordinate range is divided into ``entries`` equal
    segments of ``step = 2**step_bits`` coordinates, with
    ``step_bits = 16 - log2(entries)``.  For each segment, ideal Q15 samples
    are compared against :func:`interpolate` over the full set of remainders,
    and the best integer pair within a small neighborhood of the nominal
    (endpoint-derived) coefficients is returned.  Searching lets the fitter
    trade a little offset for a better slope (or vice versa) and typically
    halves the worst-case error.

    Selection priority (lexicographic):
        1. smallest maximum absolute error over the segment,
        2. smallest sum of squared errors,
        3. smallest perturbation from the nominal coefficients (stability).

    Args:
        entries: Number of segments; must be a power of two dividing 65536
            evenly.  :func:`generate_section` validates the usable range.

    Returns:
        A list of ``entries`` ``(slope, offset)`` tuples in segment order,
        each fitting in a signed 16-bit integer.

    Raises:
        ValueError: If an optimized coefficient does not fit in int16.
    """
    log_entries = entries.bit_length() - 1
    step_bits = 16 - log_entries
    step = 1 << step_bits
    coefficients = []
    for index in range(entries):
        start = -(1 << 15) + index * step
        # Ideal values for the whole segment, used as the fitting target.
        expected = [q15_sin(start + remainder) for remainder in range(step)]
        nominal_offset = expected[0]
        # The end sample is clamped for the final segment, where start + step
        # would be 32768 (outside the signed coordinate range).
        next_value = q15_sin(min((1 << 15) - 1, start + step))
        nominal_slope = next_value - nominal_offset

        # Exhaustively search a +-2 / +-3 box around the nominal pair; the
        # score tuple orders by (max error, squared error, perturbation).
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


def duplicate_for_four_accesses(coefficients: list[tuple[int, int]],
                                bank_bytes: int = 16) -> list[int]:
    """Flatten coefficients and replicate each target-width bank for the LUT copies.

    The four-access intrinsic reads the table through four independent ports,
    so the hardware needs duplicate physical copies.  This function groups
    the logical ``(slope, offset)`` pairs into banks of four (AIE2) or eight
    (AIE2P), flattens each bank in ``(slope0, offset0, slope1, offset1, ...)``
    order, and emits the bank twice.  The caller therefore writes the result
    into *both* the AB and CD arrays, yielding four physical copies of every
    logical entry.

    Args:
        coefficients: Logical ``(slope, offset)`` pairs in segment order;
            the length must fill complete banks.
        bank_bytes: Memory-bank width, 16 for AIE2 or 32 for AIE2P.

    Returns:
        The interleaved, twice-duplicated int16 value list.

    Raises:
        ValueError: If the bank width is unsupported or a bank is incomplete.
    """
    if bank_bytes not in (16, 32):
        raise ValueError("bank width must be 16 or 32 bytes")
    entries_per_bank = bank_bytes // 4  # int16 slope + int16 offset
    if len(coefficients) % entries_per_bank:
        raise ValueError("entry count must fill complete memory banks")
    values = []
    for base in range(0, len(coefficients), entries_per_bank):
        bank_width = [value for pair in coefficients[base:base + entries_per_bank]
                      for value in pair]
        values.extend(bank_width)
        values.extend(bank_width)
    return values


def format_array(name: str, values: list[int]) -> list[str]:
    """Format an int16 table using the target's vector alignment.

    Twelve values per source line matches the width used by the checked-in
    header, keeping regenerated output diffable against it.

    Args:
        name: C identifier for the generated array.
        values: int16 values to emit, in hardware table order.

    Returns:
        The C declaration as a list of lines, from ``alignas(...) ... = {``
        through the closing ``};``.
    """
    lines = [f"alignas(aie::vector_decl_align) static const int16_t {name}[{len(values)}] = {{"]
    for base in range(0, len(values), 12):
        lines.append("    " + ", ".join(str(value) for value in values[base:base + 12]) + ",")
    lines.append("};")
    return lines


def generate_section(entries: int) -> str:
    """Render the complete sine section of ``aie_trig_luts.h``.

    Validates the entry count, derives the segment geometry, fits the
    coefficients, duplicates them for the four-access layout, and assembles
    the ``constexpr`` geometry constants plus the AB and CD array definitions.
    No include guard or ``#include`` lines are emitted: the caller splices
    this section into the combined header.

    Args:
        entries: Number of quarter-wave segments; must be a power of two in
            [16, 8192] so that ``step_bits`` stays in [3, 12].  The lower
            bound guarantees at least eight coordinates per segment (required
            by the int16 linear approximation), and the upper bound keeps the
            segment count addressable.

    Returns:
        The generated C section as a single newline-joined string, ending in
        a trailing blank line.

    Raises:
        ValueError: If ``entries`` is not a power of two in the supported
            range, or if the derived ``step_bits`` is below three.
    """
    if entries < 16 or entries > 8192 or entries & (entries - 1):
        raise ValueError("entries must be a power of two in [16, 8192]")
    step_bits = 16 - (entries.bit_length() - 1)
    if step_bits < 3:
        raise ValueError("int16 linear_approx requires step_bits >= 3")
    coefficients = optimize_coefficients(entries)
    duplicated_aie2 = duplicate_for_four_accesses(coefficients, bank_bytes=16)
    duplicated_aie2p = duplicate_for_four_accesses(coefficients, bank_bytes=32)
    physical_bytes = 2 * len(duplicated_aie2) * 2

    lines = [
        f"constexpr unsigned kSinLinearLutEntries = {entries};",
        f"constexpr unsigned kSinLinearLutStepBits = {step_bits};",
        f"constexpr int kSinLinearLutBias = {entries // 2};",
        "",
        f"// Q15 sine offsets and integer segment slopes for {entries} quarter-wave",
        f"// segments. Four-access bank duplication makes the physical table {physical_bytes} bytes.",
        "#if __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22",
        "// AIE2P: repeat eight slope/offset pairs per 256-bit memory bank.",
        "constexpr unsigned kSinLinearLutBankBytes = 32;",
        *format_array("kSinLinearLutAb", duplicated_aie2p),
        "",
        *format_array("kSinLinearLutCd", duplicated_aie2p),
        "#else",
        "// AIE2: retain four slope/offset pairs per 128-bit memory bank.",
        "constexpr unsigned kSinLinearLutBankBytes = 16;",
        *format_array("kSinLinearLutAb", duplicated_aie2),
        "",
        *format_array("kSinLinearLutCd", duplicated_aie2),
        "#endif",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point for the sine LUT generator.

    Parses ``--entries`` and the optional ``-o/--output`` path, renders the
    section, and writes it to the file or to stdout.  The parser uses this
    module's docstring as its description, so ``--help`` documents the table
    model and physical layout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=256)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    section = generate_section(args.entries)
    if args.output:
        args.output.write_text(section)
    else:
        # print() with end="" matches write_text: no extra trailing newline.
        print(section, end="")


if __name__ == "__main__":
    main()
