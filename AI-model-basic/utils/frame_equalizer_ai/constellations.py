"""Load the exact gr-ieee802-11 constellation class-index ordering.

The receiver's demapper assigns each constellation point a class index (the
coded-bit pattern interpreted LSB-first) and the learned demapper must share
that ordering for its exported weights to be valid.  When the installed GNU
Radio ``ieee802_11`` bindings are importable they are the source of truth;
otherwise a source-compatible fallback reproduces the same points so that
checkpoints and exported artifacts remain usable on machines without GNU
Radio.
"""

from __future__ import annotations

import numpy as np

# Fixed order of the modulation models.  The index into this tuple is the
# "model index" encoded in every packed artifact (NPZ/JSON/C++ header), so do
# not reorder it without regenerating and reinstalling the weights header.
MODULATIONS = ("bpsk", "qpsk", "qam16", "qam64")
# Coded bits per constellation symbol, i.e. the number of output logits the
# network produces for each modulation.
BITS = {"bpsk": 1, "qpsk": 2, "qam16": 4, "qam64": 6}


def _fallback_points(name: str) -> np.ndarray:
    """Reproduce constellations_impl.cc if the optional binding is unavailable.

    Point ordering matches the receiver's class indices, which vary the
    in-phase component fastest.  Normalization makes the average symbol
    energy one:

        - bpsk:  ``[-1, 1]``
        - qpsk:  ``(+-1 +-1j) * sqrt(1/2)``
        - qam16: Gray-coded per-axis levels ``[-3, 3, -1, 1]`` scaled by
          ``sqrt(1/10)``
        - qam64: Gray-coded per-axis levels ``[-7, 7, -1, 1, -5, 5, -3, 3]``
          scaled by ``1/sqrt(42)``

    Args:
        name: Modulation key; must be one of :data:`MODULATIONS`.

    Returns:
        ``complex64`` array indexed by class code, one point per code.

    Raises:
        KeyError: If ``name`` is not present in :data:`BITS`.
    """
    bits = BITS[name]
    if bits == 1:
        return np.asarray([-1, 1], np.complex64)
    if bits == 2:
        level, axes = np.sqrt(0.5), [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    elif bits == 4:
        level = np.sqrt(0.1)
        # I varies fastest within each row; rows step through the Gray-coded
        # Q levels -3, 3, -1, 1.
        re = [-3, 3, -1, 1] * 4
        im = [-3] * 4 + [3] * 4 + [-1] * 4 + [1] * 4
        axes = zip(re, im)
    else:
        level = np.sqrt(1.0 / 42.0)
        # Same construction one Gray level deeper for 64-QAM.
        re = [-7, 7, -1, 1, -5, 5, -3, 3] * 8
        im = sum(([v] * 8 for v in (-7, 7, -1, 1, -5, 5, -3, 3)), [])
        axes = zip(re, im)
    return np.asarray([complex(i * level, q * level) for i, q in axes], np.complex64)


def load_constellations() -> tuple[dict[str, np.ndarray], str]:
    """Return points by modulation and identify whether GNU Radio supplied them.

    The installed ``ieee802_11`` bindings are preferred because they define
    the authoritative class ordering.  The fallback is used only when the
    bindings are missing or do not expose the expected constructors.

    Returns:
        A ``(points, source)`` tuple, where ``points`` maps each modulation
        in :data:`MODULATIONS` to a ``complex64`` array indexed by class code
        and ``source`` is a human-readable description of the provenance
        (recorded in training reports for traceability).
    """
    try:
        import ieee802_11

        constructors = {
            "bpsk": ieee802_11.constellation_bpsk,
            "qpsk": ieee802_11.constellation_qpsk,
            "qam16": ieee802_11.constellation_16qam,
            "qam64": ieee802_11.constellation_64qam,
        }
        points = {
            name: np.asarray(list(constructors[name]().points()), np.complex64)
            for name in MODULATIONS
        }
        return points, "installed ieee802_11 Python bindings"
    except (ImportError, AttributeError):
        return {name: _fallback_points(name) for name in MODULATIONS}, "source-compatible fallback"


def indices_to_bits(indices: np.ndarray, bit_count: int) -> np.ndarray:
    """Expand constellation class indices into the receiver's LSB-first bits.

    Bit ``b`` of class code ``i`` becomes column ``b`` of the output, so the
    expanded labels match the order in which the trained network's output
    logits are interpreted (the same convention as
    :func:`quantize.integer_classes`).

    Args:
        indices: Integer class codes, shape ``(N,)``.
        bit_count: Number of coded bits per class code.

    Returns:
        ``float32`` array of shape ``(N, bit_count)`` holding 0.0/1.0 labels.
    """
    return ((indices[:, None] >> np.arange(bit_count)) & 1).astype(np.float32)
