#!/usr/bin/env python3
"""Convert signed fixed-point integers to floating-point values.

Fixed-point DSP code stores fractional numbers as scaled integers; this tool
maps those integers back to the real values they encode so that LUT contents,
kernel constants, and captured data can be inspected from the command line.

Q notation convention (this project):
    ``Qm.n`` describes a two's-complement fixed-point number with ``m``
    integer bits *excluding* the sign bit, ``n`` fractional bits, and hence a
    total storage width of ``m + n + 1`` bits.  The encoded real value is
    ``raw / 2**n``.  For example, ``Q0.15`` is a signed 16-bit value and
    ``Q7.24`` is a signed 32-bit value.

Input handling:
    Raw values may be given as positional arguments or read from stdin (one
    or more whitespace-separated tokens) when no arguments are supplied.
    Tokens may be decimal or use the ``0x``/``0o``/``0b`` prefixes.  Each
    input is masked to the storage width and sign-extended when the sign bit
    is set, so unsigned representations of negative numbers (for example
    ``0xFF800000`` for ``-0.5`` in Q7.24) convert correctly.  One float is
    printed per line in input order.

Exit status:
    ``0`` on success; ``2`` on invalid arguments or unparsable input values
    (reported through ``argparse``'s error path).

Examples:
    qformat_to_float.py --q-format Q0.15 16384 -16384
    printf '0x40000000\\n-8388608\\n' | qformat_to_float.py --q-format Q7.24
    qformat_to_float.py --q-format Q7.24 --unsigned 0xC0000000
    qformat_to_float.py --q-format Q1.6 --bits 16 0xFFFF
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable


# ``Q<integer bits>.<fractional bits>``, e.g. Q0.15, q7.24.
Q_FORMAT_RE = re.compile(r"^[Qq](\d+)\.(\d+)$")


def parse_q_format(value: str) -> tuple[int, int]:
    """Parse a Q-format string into a storage width and fractional-bit count.

    The accepted syntax is ``Qm.n`` (case-insensitive), where ``m`` counts
    integer bits excluding the sign bit and ``n`` counts fractional bits.
    The total two's-complement storage width is therefore ``m + n + 1``.

    Args:
        value: Q-format string such as ``"Q7.24"`` or ``"q0.15"``.

    Returns:
        A ``(bit_width, fractional_bits)`` tuple.  For example, ``"Q0.15"``
        returns ``(16, 15)`` and ``"Q7.24"`` returns ``(32, 24)``.

    Raises:
        ValueError: If ``value`` does not match the ``Qm.n`` syntax, or if
            the implied storage width is smaller than two bits (which could
            not hold a sign bit plus at least one magnitude bit).
    """
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
    """Convert one fixed-point integer to its floating-point real value.

    The result is ``raw_value / 2**fractional_bits`` after the optional
    two's-complement normalization described below.  Supplying ``bit_width``
    lets the function accept unsigned or out-of-range representations of a
    negative number: the value is first truncated to ``bit_width`` bits and
    then sign-extended when the sign bit is set and ``signed`` is true.

    Args:
        raw_value: Scaled integer to convert.
        fractional_bits: Number of fractional bits ``n`` of the Qm.n format.

    Keyword Args:
        bit_width: Storage width in bits used to mask and sign-extend
            ``raw_value``.  When ``None``, ``raw_value`` must already be a
            correctly signed Python integer.
        signed: Interpret the masked value as two's-complement when true;
            otherwise treat it as unsigned.

    Returns:
        The real value ``raw_value / 2**fractional_bits`` as a float.

    Raises:
        ValueError: If ``fractional_bits`` is negative, or if ``bit_width``
            is given but is not greater than ``fractional_bits`` (a value
            needs at least one bit of integer magnitude).
    """
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    if bit_width is not None and bit_width <= fractional_bits:
        raise ValueError("bit_width must be greater than fractional_bits")

    if bit_width is not None:
        # Truncate to the storage width first, then sign-extend if needed.
        raw_value &= (1 << bit_width) - 1
        if signed and raw_value & (1 << (bit_width - 1)):
            raw_value -= 1 << bit_width

    return raw_value / (1 << fractional_bits)


def _parse_raw_values(values: Iterable[str]) -> list[int]:
    """Parse decimal or base-prefixed integer tokens into Python integers.

    Python's automatic base detection is used (``int(token, 0)``), so tokens
    may be plain decimal (``-42``) or carry an explicit ``0x``/``0X``,
    ``0o``/``0O``, or ``0b``/``0B`` prefix.

    Args:
        values: Iterable of raw integer token strings.

    Returns:
        The parsed integers, in input order.

    Raises:
        ValueError: If any token is not a valid integer literal.
    """
    try:
        return [int(value, 0) for value in values]
    except ValueError as error:
        raise ValueError(f"invalid raw integer: {error}") from error


def main() -> int:
    """Run the command-line fixed-point-to-float converter.

    Parses the arguments, collects input tokens from the positional arguments
    or from stdin when none were supplied, converts each token with
    :func:`q_to_float`, and prints one float per line.  Argument and parsing
    errors are reported through ``parser.error``.

    Returns:
        ``0`` on success.  Invalid input terminates the process from inside
        ``argparse`` with status ``2`` rather than returning a value.
    """
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
        # The Q format gives the default storage width; --bits overrides it
        # for cases where the input width differs from the table declaration.
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
