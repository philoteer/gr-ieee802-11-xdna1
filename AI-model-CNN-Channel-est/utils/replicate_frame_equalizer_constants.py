#!/usr/bin/env python3
"""Regenerate and verify fixed tables and constants used by the equalizer kernels.

The frame equalizer kernels hard-code a number of lookup tables (subcarrier
occupancy, constellation symbol inverses, Viterbi parity, comb-interpolation
weights, decision copies, ...) that were originally derived from GNU Radio's
``ieee802_11`` equalizer.  This script re-derives every one of them from
first principles in plain Python and checks the kernel sources against the
results, so the tables can be debugged, regenerated, and regression-checked
without an AIE compile.

Two modes of operation:
    * Verification (default): parse the kernel sources (or the files given
      with ``--source``), compare each table and literal against the Python
      derivation, print ``PASS``/``FAIL`` lines, and exit nonzero if any check
      fails.  This is the mode to run after touching the equalizer kernels.
    * Generation (``--print``): after verification, dump the regenerated
      tables and copy lists to stdout in C-array form so a changed table can
      be pasted (or diffed) against the kernel sources.

Reference tables:
    ``kLong`` (64-bin LTS frequency-domain sequence) and ``kPolarity``
    (scrambler/parity sequence) cannot be derived cheaply, so they are parsed
    from the frozen stage34 snapshot of the original GNU Radio equalizer
    (``--reference``).  All other checks are independent derivations.

Examples:
    python3 stage41/utils/replicate_frame_equalizer_constants.py
    python3 stage41/utils/replicate_frame_equalizer_constants.py --print
    python3 stage41/utils/replicate_frame_equalizer_constants.py \\
        --source kernel/frame_equalizer_preprocess.cc \\
        --source kernel/frame_equalizer_equalize.cc
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


# OFDM geometry shared with the kernels.
FFT_SIZE = 64
VECTOR_SIZE = 16
Q15 = 1 << 15


def extract_initializer(text: str, name: str) -> str:
    """Return the brace-delimited body of a C array/object definition.

    Searches for ``name`` followed by an optional ``[...]`` extent and an
    ``= { ... }`` initializer, capturing everything between the outer braces.
    The match is non-greedy, so it stops at the first ``};`` at the top level
    of plain (non-nested) brace lists, which is what the kernels use.

    Args:
        text: C/C++ source text to search.
        name: Identifier of the array or object whose initializer is wanted.

    Returns:
        The initializer body, without the surrounding braces.

    Raises:
        ValueError: If no ``= { ... };`` initializer for ``name`` is found.
    """
    pattern = re.compile(
        rf"\b{re.escape(name)}\s*(?:\[[^\]]*\])?\s*=\s*\{{(?P<body>.*?)\}}\s*;",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"could not find initializer for {name}")
    return match.group("body")


def parse_int_array(text: str, name: str) -> list[int]:
    """Parse the first integer-array definition for ``name`` as Python ints.

    Decimal and ``0x``-prefixed literals are accepted via ``int(token, 0)``.
    The token regex uses a negative lookbehind so digits embedded in
    identifiers (for example a stray ``Q15`` suffix) are not mistaken for
    values.

    Args:
        text: C/C++ source text to search.
        name: Identifier of the array to parse.

    Returns:
        The array values in definition order.

    Raises:
        ValueError: If no initializer for ``name`` is found.
    """
    body = extract_initializer(text, name)
    tokens = re.findall(r"(?<![A-Za-z_])-?(?:0[xX][0-9A-Fa-f]+|\d+)", body)
    return [int(token, 0) for token in tokens]


def parse_int_arrays(text: str, name: str) -> list[list[int]]:
    """Parse every definition of an array, not just the first one.

    Shared tables can legitimately appear in more than one translation unit
    (for example kPolarity in preprocess and in the equalizer headers), so the
    verifier must not stop at the first match.

    Args:
        text: C/C++ source text (possibly several concatenated files) to
            search.
        name: Identifier of the array to parse.

    Returns:
        One list of parsed values per definition found, in text order.  An
        empty list means the definition was not present at all.
    """
    pattern = re.compile(
        rf"\b{re.escape(name)}\s*(?:\[[^\]]*\])?\s*=\s*\{{(?P<body>.*?)\}}\s*;",
        re.DOTALL,
    )
    return [
        [
            int(token, 0)
            for token in re.findall(
                r"(?<![A-Za-z_])-?(?:0[xX][0-9A-Fa-f]+|\d+)", match.group("body")
            )
        ]
        for match in pattern.finditer(text)
    ]


def parse_copy_array(text: str, name: str) -> list[tuple[int, int, int]]:
    """Parse a table of ``{source_start, shift, destination_start}`` triples.

    These tables describe how vectors are gathered from the equalizer's
    sample buffers into decision slots; each entry is a nested braced triple,
    so a dedicated regex is used instead of the flat integer parser.

    Args:
        text: C/C++ source text to search.
        name: Identifier of the copy table.

    Returns:
        The triples in definition order.

    Raises:
        ValueError: If no initializer for ``name`` is found.
    """
    body = extract_initializer(text, name)
    return [
        tuple(int(value) for value in match)
        for match in re.findall(
            r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}", body
        )
    ]


def round_half_up(value: float) -> int:
    """Round a float to the nearest integer, ties toward +infinity.

    This mirrors the ``(int)(x + 0.5)`` idiom used throughout the kernels
    for nonnegative magnitudes, so the verifier reproduces the compiled
    constants exactly.

    Args:
        value: Value to round.

    Returns:
        The rounded integer.
    """
    return math.floor(value + 0.5)


def parity(value: int) -> int:
    """Return the parity (XOR of all bits) of a nonnegative integer.

    Args:
        value: Nonnegative integer whose bit parity is wanted.

    Returns:
        ``0`` for an even number of set bits, ``1`` for an odd count.
    """
    return value.bit_count() & 1


def expected_occupied() -> list[int]:
    """Build the 64-bin OFDM occupied-subcarrier mask.

    Legacy 802.11a/g uses bins 6 through 58 inclusive (the guard bands and
    the outer bins are empty); the DC bin 32 is excluded as well, leaving 52
    occupied subcarriers (48 data + 4 pilots).

    Returns:
        A length-64 list of 0/1 flags indexed by FFT bin.
    """
    return [int(6 <= bin_index <= 58 and bin_index != 32) for bin_index in range(FFT_SIZE)]


def expected_data_bins() -> list[int]:
    """List the FFT bins that carry data (not pilots) in an OFDM symbol.

    Starting from :func:`expected_occupied`, the four pilot subcarriers at
    bins 11, 25, 39, and 53 are removed, leaving the 48 data-carrying bins in
    ascending order.

    Returns:
        Data subcarrier indices in ascending order (length 48).
    """
    pilots = {11, 25, 39, 53}
    return [
        bin_index
        for bin_index in range(FFT_SIZE)
        if 6 <= bin_index <= 58 and bin_index != 32 and bin_index not in pilots
    ]


def expected_comb_interpolation_weights() -> list[int]:
    """Derive Q15 comb-pilot interpolation weights for every FFT bin.

    The comb channel estimator interpolates each subcarrier linearly between
    the two surrounding pilot bins (11, 25, 39, 53), with the edge segments
    interpolating from the nearest pilot toward the occupied band edge.  Each
    weight is ``position / span`` in Q15, where ``position`` is the distance
    from the segment's left pilot and ``span`` the full segment length:

        bins   0..11: position = bin,          span = 11
        bins  12..25: position = bin - 11,     span = 14
        bins  26..39: position = bin - 25,     span = 14
        bins  40..53: position = bin - 39,     span = 14
        bins  54..63: position = bin - 53,     span = 11

    Returns:
        Length-64 list of Q15 integer weights indexed by FFT bin.
    """
    weights = []
    for bin_index in range(FFT_SIZE):
        if bin_index <= 11:
            position, span = bin_index, 11
        elif bin_index <= 25:
            position, span = bin_index - 11, 14
        elif bin_index <= 39:
            position, span = bin_index - 25, 14
        elif bin_index <= 53:
            position, span = bin_index - 39, 14
        else:
            position, span = bin_index - 53, 11
        weights.append(round_half_up(position * Q15 / span))
    return weights


def constellation_points(modulation: int) -> list[complex]:
    """Constellation point per decision code, matching the kernel demapper.

    Decision codes index the points in the same order the kernel's demapper
    uses (I component varying fastest, with Gray-coded per-axis levels).
    Points are normalized so the average symbol energy is one.

    Args:
        modulation: Bits per subcarrier: 2 (BPSK), 4 (QPSK), 16 (16-QAM), or
            anything else for 64-QAM.

    Returns:
        Constellation points as complex numbers, indexed by decision code.

    Raises:
        Nothing directly; only the four values above are supported and any
        other value falls through to 64-QAM, matching the kernel's else
        branch.
    """
    if modulation == 2:
        return [-1, 1]
    if modulation == 4:
        q = 1 / math.sqrt(2)
        return [-q - 1j * q, q - 1j * q, -q + 1j * q, q + 1j * q]
    if modulation == 16:
        levels = [-3, 3, -1, 1]
        norm = math.sqrt(10)
    else:
        levels = [-7, 7, -1, 1, -5, 5, -3, 3]
        norm = math.sqrt(42)
    return [(x + 1j * y) / norm for y in levels for x in levels]


def expected_inverse_points(modulation: int) -> list[int]:
    """Derive the Q7.24 inverse-constellation table for a modulation order.

    For every decision code, the kernel stores ``1 / point`` in a Q7.24 pair,
    used to undo the channel rotation/scale when making decisions.  Real and
    imaginary parts are emitted back-to-back per code, so entry ``2*i`` is the
    real part of code ``i`` and entry ``2*i + 1`` its imaginary part.

    Args:
        modulation: Bits per subcarrier, forwarded to
            :func:`constellation_points`.

    Returns:
        Flat list of round-half-up Q7.24 integers (real then imaginary per
        decision code).
    """
    values: list[int] = []
    for point in constellation_points(modulation):
        inverse = 1 / point
        values.append(round_half_up(inverse.real * (1 << 24)))
        values.append(round_half_up(inverse.imag * (1 << 24)))
    return values


def expected_sta_reciprocals() -> list[int]:
    """Derive the Q7.24 reciprocals of STA window sizes 1 through 5.

    The spatial-temporal-averaging estimator averages each bin over its
    neighboring occupied bins; the table stores ``1 / window`` so the kernel
    can scale the window sum by multiplication.  Index 0 is unused and set to
    zero.

    Returns:
        Six Q7.24 integers: ``[0, 1/1, 1/2, 1/3, 1/4, 1/5]``.
    """
    return [0] + [
        round_half_up((1 << 24) / window) for window in range(1, 6)
    ]


def expected_viterbi_parity(polynomial: int) -> list[int]:
    """Derive a 64-entry convolutional-code parity lookup table.

    The 802.11 rate-1/2 convolutional encoder holds a 7-bit register; the
    kernel's Viterbi front end keeps the last 6 bits as state.  Shifting in a
    zero input bit and masking with the generator ``polynomial`` gives the
    parity contribution for a zero-bit transition from each state.  The two
    tables (generators 0o155 and 0o117) are consumed together when advancing
    the trellis.

    Args:
        polynomial: Convolutional code generator polynomial, for example
            ``0o155`` or ``0o117``.

    Returns:
        Length-64 list where entry ``state`` is the parity of
        ``(state << 1) & polynomial``.
    """
    return [parity((state << 1) & polynomial) for state in range(64)]


def expected_overlap_mapping(
    source_length: int, copies: list[tuple[int, int, int]]
) -> list[int | None]:
    """Simulate the kernels' shifted-overlap vector gather into 64 slots.

    Each triple ``(source_start, shift, destination_start)`` models one
    vector copy from the equalizer buffers into the decision array:

        1. A 32-value (2 x VECTOR_SIZE) load begins at ``source_start``; the
           full load must fit inside ``source_length``.
        2. The destination window holds the 16 values from
           ``source_start + shift`` through ``source_start + shift + 15``:
           its first ``16 - shift`` values come from the load at
           ``source_start`` and the remaining ``shift`` continue past
           ``source_start + 16`` from the second vector.  The window is thus
           shifted by ``shift`` positions while remaining 16 values wide.
        3. The packed window is written to ``destination_start``.

    Source slots are labeled with their index (plus zero padding for source
    arrays longer than 64), so the returned mapping says which source slot
    feeds each of the 64 destination slots.

    Args:
        source_length: Number of valid source slots; 80 for the decision
            lanes (64 per-bin decisions plus a 16-slot zero tail) used by
            ``kDataDecisionCopies``, otherwise 64.
        copies: Copy triples to apply, in order.

    Returns:
        Length-64 list of source slot indices, with ``None`` for destination
        slots no copy wrote.

    Raises:
        ValueError: If a copy's 32-value load runs past ``source_length``.
    """
    source = list(range(64)) + [0] * max(0, source_length - 64)
    destination: list[int | None] = [None] * 64
    for source_start, shift, destination_start in copies:
        load_end = source_start + 2 * VECTOR_SIZE
        if load_end > source_length:
            raise ValueError(
                f"overlap load [{source_start}, {load_end}) exceeds source length "
                f"{source_length}"
            )
        packed = (
            source[source_start + shift : source_start + VECTOR_SIZE]
            + source[source_start + VECTOR_SIZE : source_start + VECTOR_SIZE + shift]
        )
        destination[destination_start : destination_start + VECTOR_SIZE] = packed
    return destination


def expected_copy_tables() -> dict[str, list[tuple[int, int, int]]]:
    """Return the hand-derived gather tables used by the equalizer kernels.

    ``kDataDecisionCopies`` gathers the 48 data-carrier decisions out of the
    64 per-bin decision lanes (plus their 16-slot zero tail) into ascending
    bin order.

    Returns:
        Mapping from table name to its list of
        ``(source_start, shift, destination_start)`` triples.
    """
    return {
        "kDataDecisionCopies": [
            (0, 6, 0),
            (0, 12, 5),
            (16, 10, 18),
            (32, 1, 24),
            (32, 8, 30),
            (48, 6, 43),
        ],
    }


def check_array(
    source_text: str,
    name: str,
    expected: list[int],
    failures: list[str],
) -> None:
    """Verify every definition of ``name`` against an expected value list.

    Reports one ``PASS`` line when at least one definition exists and all of
    them match; otherwise reports a ``FAIL`` line per mismatching definition
    with the first differing index (and a value/length description) and
    appends a message to ``failures``.  A missing definition is a failure
    too, since a rename would silently disable the check.

    Args:
        source_text: Concatenated kernel source text.
        name: Table identifier to look up.
        expected: Reference values derived by this script.
        failures: Accumulator for human-readable failure messages; the caller
            uses its growth to set the exit status.
    """
    occurrences = parse_int_arrays(source_text, name)
    if not occurrences:
        message = f"FAIL {name}: definition not found"
        print(message)
        failures.append(message)
        return
    failures_before = len(failures)
    for occurrence, actual in enumerate(occurrences, start=1):
        if actual == expected:
            continue
        first_difference = next(
            (
                index
                for index, (actual_value, expected_value) in enumerate(zip(actual, expected))
                if actual_value != expected_value
            ),
            min(len(actual), len(expected)),
        )
        actual_value = actual[first_difference] if first_difference < len(actual) else "missing"
        expected_value = expected[first_difference] if first_difference < len(expected) else "missing"
        message = (
            f"FAIL {name} (definition {occurrence} of {len(occurrences)}): "
            f"index {first_difference}, got {actual_value}, expected "
            f"{expected_value}; lengths {len(actual)} != {len(expected)}"
        )
        print(message)
        failures.append(message)
    if len(failures) == failures_before:
        print(f"PASS {name}: {len(expected)} values in {len(occurrences)} definition(s)")


def check_copies(
    source_text: str,
    name: str,
    expected: list[tuple[int, int, int]],
    failures: list[str],
) -> None:
    """Verify a gather-copy table and the carrier order it produces.

    Two checks are performed: the parsed triples must equal ``expected``, and
    simulating them with :func:`expected_overlap_mapping` must map the start
    of the destination array onto the expected subcarrier order.  For
    ``kDataDecisionCopies`` the source is the 80-slot decision array (64
    per-bin decisions plus a 16-slot zero tail) and the expected order is the
    data bins; other tables use a 64-slot buffer and the occupied bins.

    Args:
        source_text: Concatenated kernel source text.
        name: Copy-table identifier to look up.
        expected: Reference triples derived by this script.
        failures: Accumulator for human-readable failure messages.
    """
    actual = parse_copy_array(source_text, name)
    if actual != expected:
        message = f"FAIL {name}: got {actual}, expected {expected}"
        print(message)
        failures.append(message)
        return

    source_length = 80 if name == "kDataDecisionCopies" else 64
    mapped = expected_overlap_mapping(source_length, actual)
    if name == "kDataDecisionCopies":
        expected_output = expected_data_bins()
    else:
        expected_output = [index for index, occupied in enumerate(expected_occupied()) if occupied]
    if mapped[: len(expected_output)] != expected_output:
        message = f"FAIL {name}: overlap mapping does not produce the expected carrier order"
        print(message)
        failures.append(message)
        return
    print(f"PASS {name}: table and overlap mapping")


def check_literal(
    source_text: str,
    label: str,
    literal: int,
    failures: list[str],
    expected_count: int = 1,
) -> None:
    """Verify how often an integer literal appears as a standalone token.

    Used for constants that are more robustly checked by occurrence count
    than by parsing (for example the QAM decision thresholds, which should
    appear exactly once each).  Identifiers containing the digits are ignored
    thanks to the surrounding negative lookarounds.

    Args:
        source_text: Concatenated kernel source text.
        label: Human-readable name for the constant, used in messages.
        literal: Integer literal to count.
        failures: Accumulator for human-readable failure messages.
        expected_count: Required number of standalone occurrences.
    """
    count = len(re.findall(rf"(?<![A-Za-z_]){literal}(?![A-Za-z_])", source_text))
    if count == expected_count:
        print(f"PASS {label}: {literal}")
    else:
        message = f"FAIL {label}: literal {literal} occurs {count} times, expected {expected_count}"
        print(message)
        failures.append(message)


def check_definition(
    source_text: str,
    label: str,
    pattern: str,
    expected: str,
    failures: list[str],
) -> None:
    """Verify that a definition matching a regex still exists in the sources.

    This is the fallback check for ``#define`` constants and other textual
    contracts that have no Python derivation.

    Args:
        source_text: Concatenated kernel source text.
        label: Human-readable name for the definition, used in messages.
        pattern: Regular expression the source must match.
        expected: Textual description of the expected definition, used in
            both the PASS and FAIL messages.
        failures: Accumulator for human-readable failure messages.
    """
    if re.search(pattern, source_text, re.DOTALL):
        print(f"PASS {label}: {expected}")
    else:
        message = f"FAIL {label}: expected definition {expected!r} is missing or changed"
        print(message)
        failures.append(message)


def check_rate_cases(source_text: str, failures: list[str]) -> None:
    """Verify the SIGNAL-field rate mapping for all eight legacy rates.

    The equalizer's ``parse_signal`` switch must map each 4-bit rate code to
    its ``(d_encoding, d_modulation, n_dbps)`` triple.  The expected table is
    the 802.11a/g legacy set:

        rate 11 -> 6 Mbps   (BPSK   1/2,  24 bits/symbol)
        rate 15 -> 9 Mbps   (BPSK   3/4,  36)
        rate 10 -> 12 Mbps  (QPSK   1/2,  48)
        rate 14 -> 18 Mbps  (QPSK   3/4,  72)
        rate  9 -> 24 Mbps  (16-QAM 1/2,  96)
        rate 13 -> 36 Mbps  (16-QAM 3/4, 144)
        rate  8 -> 48 Mbps  (64-QAM 2/3, 192)
        rate 12 -> 54 Mbps  (64-QAM 3/4, 216)

    Each case is matched as an exact ``case N: d_encoding = E; d_modulation =
    M; n_dbps = D;`` statement so reordered or retyped assignments are caught.

    Args:
        source_text: Concatenated kernel source text.
        failures: Accumulator for human-readable failure messages.
    """
    expected = {
        11: (0, 2, 24),
        15: (1, 2, 36),
        10: (2, 4, 48),
        14: (3, 4, 72),
        9: (4, 16, 96),
        13: (5, 16, 144),
        8: (6, 64, 192),
        12: (7, 64, 216),
    }
    for rate, (encoding, modulation, n_dbps) in expected.items():
        pattern = (
            rf"case\s+{rate}:\s*d_encoding\s*=\s*{encoding};\s*"
            rf"d_modulation\s*=\s*{modulation};\s*n_dbps\s*=\s*{n_dbps};"
        )
        if re.search(pattern, source_text):
            print(f"PASS RATE {rate}: encoding={encoding}, modulation={modulation}, N_DBPS={n_dbps}")
        else:
            message = f"FAIL RATE {rate}: expected rate mapping is missing or changed"
            print(message)
            failures.append(message)


def format_values(values: list[int]) -> str:
    """Join integers with commas for C-array output.

    Args:
        values: Values to format.

    Returns:
        Comma-and-space separated decimal string.
    """
    return ", ".join(str(value) for value in values)


def print_generated(reference_text: str) -> None:
    """Print regenerated tables in C-array form for manual comparison.

    Emits the reference-derived ``kLong`` and ``kPolarity`` tables plus this
    script's derivations (occupancy, data bins, Viterbi parity/states) with
    16 values per line, then the gather-copy tables.  Intended to be run with
    ``--print`` after verification passes, so the dump cannot silently
    disagree with the checks.

    Args:
        reference_text: Source text containing the reference ``LONG`` and
            ``POLARITY`` definitions.
    """
    generated = {
        "kLong (standard reference)": parse_int_array(reference_text, "LONG"),
        "kPolarity (standard reference)": parse_int_array(reference_text, "POLARITY"),
        "kOccupied": expected_occupied(),
        "kDataCarrierBins": expected_data_bins(),
        "kViterbiParityA": expected_viterbi_parity(0o155),
        "kViterbiParityB": expected_viterbi_parity(0o117),
        "kViterbiStates": list(range(32)),
    }
    for name, values in generated.items():
        print(f"\n{name}[{len(values)}] = {{")
        for start in range(0, len(values), 16):
            print("    " + format_values(values[start : start + 16]) + ",")
        print("};")

    for name, copies in expected_copy_tables().items():
        print(f"\n{name} = {copies}")


def main() -> int:
    """Run the frame-equalizer constant verifier.

    Builds the source set (``--source`` arguments or the default kernel
    files), reads the reference source, runs every table/literal/definition
    check, optionally dumps regenerated tables (``--print``), and summarizes.

    Returns:
        ``0`` when all checks pass; ``1`` when any check failed (the failure
        count is printed to stderr).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        help="kernel source to verify; may be specified more than once",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "stage34/original/equalizer/base.cc",
        help="reference source containing the standard LONG and POLARITY tables",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="print regenerated tables after verification",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sources = args.source or [
        root / "kernel/frame_equalizer_preprocess.cc",
        root / "kernel/frame_equalizer_equalize.cc",
        root / "kernel/demodulator/frame_equalizer_engine_common.h",
        root / "kernel/channel_estimator/common.h",
        root / "kernel/channel_estimator/constellation.h",
        root / "kernel/channel_estimator/ls.h",
        root / "kernel/channel_estimator/lms.h",
        root / "kernel/channel_estimator/comb.h",
        root / "kernel/channel_estimator/sta.h",
    ]
    source_text = "\n".join(source.read_text() for source in sources)
    reference_text = args.reference.read_text()
    failures: list[str] = []

    # Every entry is (table name, expected values), checked in each place the
    # source defines it (some tables are shared across translation units).
    checks = [
        ("kLong", parse_int_array(reference_text, "LONG")),
        ("kPolarity", parse_int_array(reference_text, "POLARITY")),
        ("kOccupied", expected_occupied()),
        ("kCombInterpolationWeightQ15", expected_comb_interpolation_weights()),
        ("kInversePointsBpsk", expected_inverse_points(2)),
        ("kInversePointsQpsk", expected_inverse_points(4)),
        ("kInversePoints16Qam", expected_inverse_points(16)),
        ("kInversePoints64Qam", expected_inverse_points(64)),
        ("kStaReciprocalQ24", expected_sta_reciprocals()),
        ("kPhaseLaneIndices", list(range(16))),
        ("kViterbiParityA", expected_viterbi_parity(0o155)),
        ("kViterbiParityB", expected_viterbi_parity(0o117)),
        ("kViterbiStates", list(range(32))),
    ]
    for name, expected in checks:
        check_array(source_text, name, expected, failures)

    for name, expected in expected_copy_tables().items():
        check_copies(source_text, name, expected, failures)

    # QAM decision boundaries: midpoints between adjacent PAM levels scaled by
    # the constellation normalization (16-QAM: 2/sqrt(10); 64-QAM: n/sqrt(42)
    # for n = 2, 4, 6), all in Q15.
    qam_thresholds = {
        "QAM16 boundary": round_half_up(math.sqrt(2 / 5) * Q15),
        "QAM64 boundary 1": round_half_up(math.sqrt(2 / 21) * Q15),
        "QAM64 boundary 2": round_half_up(2 * math.sqrt(2 / 21) * Q15),
        "QAM64 boundary 3": round_half_up(3 * math.sqrt(2 / 21) * Q15),
    }
    for label, value in qam_thresholds.items():
        check_literal(source_text, label, value, failures)
    check_definition(source_text, "FFT size", r"#define\s+FFT_SIZE\s+64\b", "FFT_SIZE = 64", failures)
    check_definition(
        source_text,
        "OFDM symbol samples",
        r"#define\s+SAMPLES_WITH_CP\s+80\b",
        "SAMPLES_WITH_CP = 80",
        failures,
    )
    check_rate_cases(source_text, failures)

    if args.print:
        print_generated(reference_text)

    if failures:
        print(f"\n{len(failures)} verification failure(s)", file=sys.stderr)
        return 1
    print("\nAll frame_equalizer constant checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
