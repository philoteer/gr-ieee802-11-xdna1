"""Integer Q7/Q15 reference implementation for the CNN estimator.

The routines in this module define the numerical contract shared by the model
exporter and the AIE kernel: signed Q7 weights, signed Q15 activations, int64
convolution accumulators, arithmetic right shifts, ReLU activation, and int32
saturation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

try:
    from .dataset import ChannelCorpus, OCCUPIED
except ImportError:
    from dataset import ChannelCorpus, OCCUPIED

# Integer inference only needs NumPy, not the training framework.
if TYPE_CHECKING:
    try:
        from .model import ChannelEstimator
    except ImportError:
        from model import ChannelEstimator


@dataclass(frozen=True)
class QuantizedEstimator:
    """Container for the three exported int8 convolution weight tensors.

    Attributes:
        weights: Tuple containing the first hidden, second hidden, and output
            convolution tensors in ``[output, input, tap]`` layout.
    """

    weights: tuple[np.ndarray, np.ndarray, np.ndarray]


def quantize_model(model: ChannelEstimator) -> QuantizedEstimator:
    """Quantize floating-point convolution weights into signed Q7 values.

    Args:
        model: Floating-point :class:`ChannelEstimator` to quantize.

    Returns:
        A :class:`QuantizedEstimator` with int8 weights rounded to the nearest
        representable Q7 value and clipped to the int8 range.
    """
    weights = tuple(np.clip(np.rint(layer.weight.detach().numpy() * 128.0),
                            -128, 127).astype(np.int8) for layer in model.layers)
    return QuantizedEstimator(weights)  # type: ignore[arg-type]


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
    return estimate


def load_header(path: Path) -> QuantizedEstimator:
    """Read the deployed Q7 tensors without requiring private training artifacts.

    The exporter writes nested integer initializer lists. Parse those as data,
    checking the declared tensor shapes and int8 range before narrowing.
    """
    source = path.read_text()
    weights = []
    for index in range(1, 4):
        match = re.search(
            rf"\bkW{index}((?:\[\d+\])+)\s*=\s*(\{{.*?\}});", source, re.S)
        if match is None:
            raise ValueError(f"missing CNN weight kW{index} in {path}")
        shape = tuple(int(size) for size in re.findall(r"\d+", match.group(1)))
        values = np.asarray(ast.literal_eval(
            match.group(2).replace("{", "[").replace("}", "]")))
        if (values.shape != shape or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < -128) or np.any(values > 127)):
            raise ValueError(f"invalid CNN weight kW{index} in {path}")
        weights.append(values.astype(np.int8))
    return QuantizedEstimator(tuple(weights))  # type: ignore[arg-type]


def load_npz(path) -> QuantizedEstimator:
    """Load exported convolution tensors from a NumPy archive.

    Args:
        path: Path to an archive containing ``w1``, ``w2``, and ``w3`` arrays.

    Returns:
        A :class:`QuantizedEstimator` reconstructed from the archive.
    """
    with np.load(path, allow_pickle=False) as data:
        return QuantizedEstimator(tuple(data[f"w{index}"] for index in range(1, 4)))  # type: ignore[arg-type]
