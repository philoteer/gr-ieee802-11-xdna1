#!/usr/bin/env python3
"""Verify bit-exact parity between the CPU CNN reference and the AIE kernel.

The test deliberately exercises the estimator in isolation.  It supplies the
two long-training symbols directly to a small AIE graph, obtains the resulting
64-bin complex channel estimate, and compares that result with the NumPy
integer reference used during model export.

The comparison is exact rather than tolerance-based.  A one-unit difference
usually indicates a changed rounding mode, shift, saturation rule, or memory
layout, all of which can alter downstream hard decisions.
"""

import argparse
import sys
from pathlib import Path

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "utils" / "frame_equalizer_channel_ai"
sys.path.insert(0, str(REFERENCE))

from dataset import ChannelCorpus, LONG, OCCUPIED, _shared_q15, average_lts
from quantize import integer_estimate, load_npz


def make_lts() -> np.ndarray:
    """Create deterministic, sign-coded LTS observations for the test.

    The generated values are already in the corrected FFT-domain representation
    expected by the estimator adapter.  The IEEE 802.11 long-training signs are
    applied afterward so that the adapter receives the same raw LTS convention
    as the production equalizer.

    :return: Integer array with shape ``(2, 64, 2)``.  The final dimension is
        real/imaginary, and the first dimension contains LTS0 and LTS1.
    """
    bins = np.arange(64, dtype=np.int64)
    rng = np.random.default_rng(8021140)
    corrected = np.empty((2, 64, 2), dtype=np.int32)
    corrected[:, :, 0] = (
        11731 + 83 * bins + rng.integers(-1703, 1704, size=(2, 64))
    )
    corrected[:, :, 1] = (
        -4937 + 47 * bins + rng.integers(-1301, 1302, size=(2, 64))
    )
    corrected[:, ~OCCUPIED] = 0
    return corrected * LONG[None, :, None]


def cpu_estimate(lts: np.ndarray) -> np.ndarray:
    """Run the exported CNN using the host-side integer reference.

    This function intentionally follows the production path: remove the LTS
    signs, compute the LS initialization, normalize the two observations into
    Q15, and apply the generated Q7 convolution weights.

    :param lts: Raw sign-coded LTS observations with shape ``(2, 64, 2)``.
    :return: Estimated channel with shape ``(64, 2)`` and signed int32
        real/imaginary components.
    :raises ValueError: Propagated by the reference utilities for malformed
        tensor shapes or unavailable weight artifacts.
    """
    corrected = lts * LONG[None, :, None]
    ls = average_lts(corrected[0], corrected[1])
    normalized, shifts = _shared_q15(corrected[None])
    inputs = np.empty((1, 5, 64), dtype=np.int32)
    inputs[:, :4] = normalized.transpose(0, 1, 3, 2).reshape(1, 4, 64)
    inputs[:, 4] = OCCUPIED.astype(np.int32) * 32767
    corpus = ChannelCorpus(
        inputs,
        lts[None],
        ls[None],
        ls[None].copy(),
        np.zeros((1, 64, 2), dtype=np.int32),
        shifts,
    )
    weights = REFERENCE / "artifacts" / "channel_estimator_weights.npz"
    return integer_estimate(load_npz(weights), corpus)[0]


def assert_exact(actual: np.ndarray, expected: np.ndarray, lts: np.ndarray) -> None:
    """Fail with actionable diagnostics when CPU and AIE results differ.

    The first mismatches include the LS baseline and both residual values.  This
    makes it possible to distinguish an incorrect CNN activation from an
    incorrect residual de-normalization or channel accumulation.

    :param actual: Channel estimate returned by the AIE graph.
    :param expected: Channel estimate returned by :func:`cpu_estimate`.
    :param lts: Input LTS tensor used to produce both estimates.
    :raises AssertionError: If any real or imaginary component differs.
    """
    mismatch = np.argwhere(actual != expected)
    if mismatch.size == 0:
        return
    delta = actual.astype(np.int64) - expected.astype(np.int64)
    max_index = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    complex_bins = np.unique(mismatch[:, 0]).size
    corrected = lts * LONG[None, :, None]
    ls = average_lts(corrected[0], corrected[1])
    details = []
    for bin_index, component in mismatch[:12]:
        name = "real" if component == 0 else "imag"
        got = int(actual[bin_index, component])
        want = int(expected[bin_index, component])
        details.append(
            f"bin {bin_index:2d} {name}: AIE={got} CPU={want} "
            f"delta={got - want} AIE_res={got - int(ls[bin_index, component])} "
            f"CPU_res={want - int(ls[bin_index, component])}"
        )
    raise AssertionError(
        f"CNN CPU/AIE parity failed at {len(mismatch)} components across "
        f"{complex_bins} complex bins; max_abs_delta={abs(delta[max_index])} "
        f"at bin {max_index[0]} component {max_index[1]}\n"
        + "\n".join(details)
    )


def main() -> None:
    """Run the runtime parity check against supplied AIE artifacts.

    :raises AssertionError: If the AIE result is not bit-exact with the CPU
        reference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--xclbin", type=Path, default=Path("build/final.xclbin"))
    parser.add_argument("--insts", type=Path, default=Path("build/insts.bin"))
    args = parser.parse_args()

    # Compute the expected result before touching the device.  This keeps the
    # reference independent of any state retained by the AIE runtime.
    lts = make_lts()
    expected = cpu_estimate(lts)
    output = iron.zeros(64 * 2, dtype=np.int32)
    handle = DefaultNPURuntime.load(
        NPUKernel(str(args.xclbin), str(args.insts), kernel_name="MLIR_AIE")
    )
    # The adapter performs the symbol-0 capture and symbol-1 CNN inference.
    DefaultNPURuntime.run(
        handle, [iron.tensor(lts.reshape(-1), dtype=np.int32), output]
    )
    actual = output.numpy().reshape(64, 2)
    # Compare guards, DC, and occupied bins alike; the production contract
    # requires the complete 64-bin channel buffer to be deterministic.
    assert_exact(actual, expected, lts)
    print("CNN parity PASS: all 64 complex channel estimates match exactly")


if __name__ == "__main__":
    main()
