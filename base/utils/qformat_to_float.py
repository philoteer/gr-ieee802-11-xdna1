#!/usr/bin/env python3
"""Convert signed fixed-point integers to floating-point values.

Examples:
    qformat_to_float.py --q-format Q0.15 16384 -16384
    printf '0x40000000\n-8388608\n' | qformat_to_float.py --q-format Q7.24

The Q notation used here follows the convention in this project: the first
number counts integer bits excluding the sign bit.  Therefore Q0.15 is a
signed 16-bit value, while Q7.24 is a signed 32-bit value.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable


Q_FORMAT_RE = re.compile(r"^[Qq](\d+)\.(\d+)$")


def parse_q_format(value: str) -> tuple[int, int]:
    """Return ``(bit_width, fractional_bits)`` for a Q-format string."""
    match = Q_FORMAT_RE.fullmatch(value)
    if match is None:
        raise ValueError("Q format must look like Q7.24")

    integer_bits, fractional_bits = (int(part) for part in match.groups())
    bit_width = integer_bits + fractional_bits + 1
    if bit_width < 2:
        raise ValueError("Q format must contain at least two bits")
    return bit_width, fractional_bits


def q_to_float(
    raw_value: int,
    fractional_bits: int,
    *,
    bit_width: int | None = None,
    signed: bool = True,
) -> float:
    """Convert one fixed-point integer to a float.

    ``bit_width`` is needed when the input is an unsigned representation of a
    signed value, or when the raw value needs sign extension.  If omitted,
    ``raw_value`` is assumed to already be a correctly signed Python integer.
    """
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    if bit_width is not None and bit_width <= fractional_bits:
        raise ValueError("bit_width must be greater than fractional_bits")

    if bit_width is not None:
        raw_value &= (1 << bit_width) - 1
        if signed and raw_value & (1 << (bit_width - 1)):
            raw_value -= 1 << bit_width

    return raw_value / (1 << fractional_bits)


def _parse_raw_values(values: Iterable[str]) -> list[int]:
    """Parse decimal or ``0x``/``0o``/``0b`` integer tokens."""
    try:
        return [int(value, 0) for value in values]
    except ValueError as error:
        raise ValueError(f"invalid raw integer: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert signed fixed-point integers to floating-point values."
    )
    parser.add_argument(
        "values",
        nargs="*",
        help="raw integer values (decimal or 0x/0o/0b); read stdin if omitted",
    )
    parser.add_argument(
        "-q",
        "--q-format",
        required=True,
        metavar="Qm.n",
        help="Q format, for example Q7.24 or Q0.15",
    )
    parser.add_argument(
        "--unsigned",
        action="store_true",
        help="treat values as unsigned instead of signed",
    )
    parser.add_argument(
        "--bits",
        type=int,
        metavar="N",
        help="override the inferred storage width",
    )
    args = parser.parse_args()

    try:
        inferred_width, fractional_bits = parse_q_format(args.q_format)
        bit_width = args.bits if args.bits is not None else inferred_width
        tokens = args.values or sys.stdin.read().split()
        raw_values = _parse_raw_values(tokens)
        converted = (
            q_to_float(
                raw_value,
                fractional_bits,
                bit_width=bit_width,
                signed=not args.unsigned,
            )
            for raw_value in raw_values
        )
        for value in converted:
            print(value)
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
