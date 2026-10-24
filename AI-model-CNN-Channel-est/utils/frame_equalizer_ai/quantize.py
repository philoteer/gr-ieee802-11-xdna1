"""Symmetric int8 quantization and bit-exact integer reference inference.

The AIE equalizer engine runs the learned demapper with int8 weights, int32
Q23 inputs, acc64 accumulators, ReLU, and per-layer arithmetic right shifts.
The number of hidden layers depends on the selected model architecture.  This
module quantizes a trained float model to that contract and provides an
integer NumPy reference that ``cli.py validate`` checks for class parity
against the exported arrays.  The reference mirrors the kernel exactly:
floor-based input conversion, truncating shifts, and unshifted final logits
thresholded at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .model import DATA_BINS, FEATURE_COUNT, FFT_SIZE
except ImportError:
    from model import DATA_BINS, FEATURE_COUNT, FFT_SIZE

# Number of fractional bits in the Q23 signed fixed-point input format. One
# normalized unit equals 2**INPUT_FRACTIONAL_BITS.
INPUT_FRACTIONAL_BITS = 23
INPUT_SCALE = 1 << INPUT_FRACTIONAL_BITS


@dataclass
class QuantizedModel:
    """Weights and shifts implementing the AIE-facing integer contract.

    Attributes:
        bit_count: Number of output logits (coded bits) this model emits.
        hidden_weights: One ``(hidden, 3)`` or ``(hidden, hidden)`` int8 matrix
            per hidden layer, in forward order.
        output_weights: ``(bit_count, hidden)`` int8 output-layer weights.
        shifts: One arithmetic right shift per hidden-layer ReLU, in order.
    """

    bit_count: int
    hidden_weights: list[np.ndarray]
    output_weights: np.ndarray
    shifts: list[int]


def quantize_inputs(features: np.ndarray) -> np.ndarray:
    """Normalize matched-filter features and map them to the Q23 AIE contract.

    The per-symbol power-of-two scale is derived from the largest absolute
    feature over all 64 bins, matching the AIE input stage.  The three
    data-subcarrier features are then clipped to [-1, 1] and scaled by
    ``2**23``.  The result is floored rather than rounded because the AIE input
    stage uses truncating shifts; the integer reference must produce identical
    values.

    Args:
        features: Float matched-filter features of shape ``(N, 64, 3)`` or
            flattened ``(N, 192)``.

    Returns:
        ``int32`` array of Q23 values, shape ``(N * 48, 3)``.
    """
    values = np.asarray(features)
    if values.ndim == 2:
        if values.shape[1] != FFT_SIZE * FEATURE_COUNT:
            raise ValueError("flattened features must contain 192 values")
        values = values.reshape(-1, FFT_SIZE, FEATURE_COUNT)
    if values.ndim != 3 or values.shape[1:] != (FFT_SIZE, FEATURE_COUNT):
        raise ValueError("features must have shape (N, 64, 3) or (N, 192)")
    maximum = np.maximum(np.max(np.abs(values), axis=(1, 2), keepdims=True), 1e-12)
    exponent = np.floor(np.log2(maximum)) + 1.0
    normalized = values / np.exp2(exponent)
    selected = normalized[:, DATA_BINS, :].reshape(-1, FEATURE_COUNT)
    clipped = np.clip(selected, -1.0, 1.0)
    # The AIE input stage uses truncating shifts, so the reference floors too.
    return np.floor(clipped * INPUT_SCALE).astype(np.int32)


def _weights(layer) -> np.ndarray:
    """Quantize one dense layer symmetrically to int8.

    The scale is chosen per tensor as ``max|w| / 127`` so the largest
    magnitude uses the full int8 range; values are rounded to nearest with
    ``np.rint`` and clamped to [-127, 127] (``-128`` is avoided to keep the
    range symmetric).

    Args:
        layer: A bias-free ``nn.Linear`` whose weight should be quantized.

    Returns:
        ``int8`` weight array with the layer's original shape.
    """
    values = layer.weight.detach().cpu().numpy()
    scale = max(float(np.max(np.abs(values))) / 127.0, 1e-12)
    return np.clip(np.rint(values / scale), -127, 127).astype(np.int8)


def _safe_shift(maximum: int, target_bits: int = 20) -> int:
    """Retain useful activation precision while bounding the following acc64 MAC.

    Returns the number of low bits to discard so the largest activation keeps
    roughly ``target_bits`` significant bits.  Shifting down by that amount
    keeps the next layer's int8 x int64 dot products comfortably inside int64
    while preserving enough resolution for classification.

    Args:
        maximum: Largest post-ReLU activation magnitude seen during
            calibration.
        target_bits: Desired number of significant bits to retain.

    Returns:
        Nonnegative arithmetic right-shift amount.
    """
    return max(0, int(maximum).bit_length() - target_bits)


def quantize_model(model: JointIQMLP, calibration: np.ndarray,
                   bit_count: int) -> QuantizedModel:
    """Choose activation shifts from calibration maxima; no labels are consulted.

    Quantizes every hidden weight matrix and the output matrix, runs the
    integer pipeline over the calibration split while tracking the largest
    post-ReLU activation after each hidden layer, and derives one shift per
    layer with :func:`_safe_shift`. Using only feature maxima means the shifts
    do not leak label information.

    Args:
        model: Trained float :class:`model.JointIQMLP`.
        calibration: Float features of shape ``(N, 64, 3)`` or flattened
            ``(N, 192)``, used only to size activations.
        bit_count: Number of output logits for this modulation.

    Returns:
        The fully quantized :class:`QuantizedModel`.
    """
    linears = model.linear_layers()
    hidden_weights = [_weights(layer) for layer in linears[:-1]]
    output_weights = _weights(linears[-1])
    x = quantize_inputs(calibration).astype(np.int64)
    shifts = []
    for weights in hidden_weights:
        activations = np.maximum(x @ weights.astype(np.int64).T, 0)
        shift = _safe_shift(int(activations.max(initial=0)))
        shifts.append(shift)
        x = activations >> shift
    return QuantizedModel(bit_count, hidden_weights, output_weights, shifts)


def integer_logits(model: QuantizedModel, features: np.ndarray) -> np.ndarray:
    """Run int32 input, int8 MACs, acc64, ReLU/shifts, then unshifted output.

    The stage order mirrors the AIE kernel exactly:

        1. Q23 int32 inputs (:func:`quantize_inputs`).
        2. For each hidden layer: ``W`` int8 MAC accumulated in int64, ReLU,
           shift.
        3. Output-layer int8 MAC accumulated in int64, returned unscaled.

    Args:
        model: Quantized model providing weights and shifts.
        features: Float features, converted internally via
            :func:`quantize_inputs`.

    Returns:
        ``(N, bit_count)`` int64 logits.
    """
    x = quantize_inputs(features).astype(np.int64)
    for weights, shift in zip(model.hidden_weights, model.shifts):
        x = np.maximum(x @ weights.astype(np.int64).T, 0) >> shift
    return x @ model.output_weights.astype(np.int64).T


def integer_classes(model: QuantizedModel, features: np.ndarray) -> np.ndarray:
    """Threshold integer logits at zero and pack them LSB-first into classes.

    This is the quantized equivalent of :func:`model.predict_classes` and the
    final decision made by the AIE demapper.

    Args:
        model: Quantized model providing weights, shifts, and bit count.
        features: Float features.

    Returns:
        ``(N,)`` int64 class codes.
    """
    logits = integer_logits(model, features)
    return np.sum((logits >= 0).astype(np.int64) << np.arange(model.bit_count), axis=1)
