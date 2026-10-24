"""Integer Q7/Q15 reference implementation for the CNN estimator.

The routines in this module define the numerical contract shared by the model
exporter and the AIE kernel: signed Q7 weights, signed Q15 activations, int64
convolution accumulators, arithmetic right shifts, ReLU activation, and int32
saturation.
"""

from __future__ import annotations

import numpy as np

from ..common.dataset import ChannelCorpus, OCCUPIED
from ..common.quantized import QuantizedEstimator


def _conv1d_q15(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Evaluate one same-padded convolution layer in integer arithmetic.

    Args:
        values: Input activation tensor with shape ``(N, input_channels, W)``.
        weights: Q7 weights with shape ``(output_channels, input_channels,
            kernel_width)``.

    Returns:
        Q15 int32 output tensor with shape
        ``(N, output_channels, W)`` after the Q7 product shift and saturation.
    """
    batch, _, width = values.shape
    output_channels, _, kernel = weights.shape
    padded = np.pad(values.astype(np.int64), ((0, 0), (0, 0),
                                              (kernel // 2, kernel // 2)))
    output = np.zeros((batch, output_channels, width), dtype=np.int64)
    for tap in range(kernel):
        output += np.einsum("bci,oc->boi", padded[:, :, tap:tap + width],
                            weights[:, :, tap].astype(np.int64), optimize=True)
    output >>= 7
    return np.clip(output, -2147483648, 2147483647).astype(np.int32)


def integer_residual(model: QuantizedEstimator, inputs_q15: np.ndarray) -> np.ndarray:
    """Compute CNN residuals using the exported integer weights.

    Args:
        model: Quantized estimator containing three convolution tensors.
        inputs_q15: Q15 input tensor with shape ``(N, 5, 64)``.

    Returns:
        Int32 residual tensor with shape ``(N, 64, 2)``.
    """
    value = _conv1d_q15(inputs_q15, model.weights[0])
    value = np.maximum(value, 0).astype(np.int32)
    value = _conv1d_q15(value, model.weights[1])
    value = np.maximum(value, 0).astype(np.int32)
    return _conv1d_q15(value, model.weights[2]).transpose(0, 2, 1)


def integer_estimate(model: QuantizedEstimator, corpus: ChannelCorpus) -> np.ndarray:
    """Apply integer CNN residuals to the LS estimate.

    Args:
        model: Quantized channel-estimator model.
        corpus: Corpus containing Q15 inputs, LS estimates, and normalization
            shifts.

    Returns:
        Saturated int32 channel estimates with shape ``(N, 64, 2)`` and zero
        values on unoccupied bins.
    """
    residual = integer_residual(model, corpus.inputs_q15)
    estimate = np.empty_like(corpus.ls)
    for index, shift in enumerate(corpus.normalization_shift):
        if shift >= 0:
            raw = residual[index].astype(np.int64) >> int(shift)
        else:
            raw = residual[index].astype(np.int64) << int(-shift)
        estimate[index] = np.clip(corpus.ls[index].astype(np.int64) + raw,
                                  -2147483648, 2147483647).astype(np.int32)
    estimate[:, ~OCCUPIED] = 0
    estimate[corpus.snr_ratio_x16 >= model.gate_snr_ratio_x16] = corpus.ls[
        corpus.snr_ratio_x16 >= model.gate_snr_ratio_x16]
    return estimate


def load_npz(path) -> QuantizedEstimator:
    """Load exported convolution tensors from a NumPy archive.

    Args:
        path: Path to an archive containing ``w1``, ``w2``, and ``w3`` arrays.

    Returns:
        A :class:`QuantizedEstimator` reconstructed from the archive.
    """
    with np.load(path, allow_pickle=False) as data:
        return QuantizedEstimator(
            tuple(data[f"w{index}"] for index in range(1, 4)),
            int(data["gate_snr_ratio_x16"]),
        )  # type: ignore[arg-type]
