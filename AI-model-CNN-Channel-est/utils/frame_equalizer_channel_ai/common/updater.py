"""Shared data preparation and fixed-point helpers for the channel updater."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import numpy as np

from .dataset import OCCUPIED, snr_feature_q15, usable_pair_indices
from .quantized import QuantizedEstimator
from ..verification.quantized import integer_estimate


PILOT_BINS = np.asarray((11, 25, 39, 53))
OCCUPIED_BINS = np.flatnonzero(OCCUPIED)
GROUPS = tuple(OCCUPIED_BINS[start:start + 13] for start in range(0, 52, 13))
INPUT_SIZE = 61


def load_cnn_header(path: Path) -> QuantizedEstimator:
    """Loads CNN weights and its SNR gate from a generated C++ header.

    Args:
        path: Path to the exported channel-estimator weights header.

    Returns:
        A quantized estimator containing the three convolution layers and gate.

    Raises:
        ValueError: If a required weight array or SNR gate is missing.
    """
    source = path.read_text()
    weights = []
    for index in range(1, 4):
        match = re.search(rf"\bkW{index}(?:\[\d+\])+\s*=\s*(\{{.*?\}});", source, re.S)
        if match is None:
            raise ValueError(f"missing CNN weight kW{index} in {path}")
        weights.append(np.asarray(ast.literal_eval(match.group(1).replace('{', '[').replace('}', ']')),
                                  dtype=np.int8))
    gate = re.search(r"\bkLsGateSnrRatioX16\s*=\s*(\d+)", source)
    if gate is None:
        raise ValueError(f"missing CNN SNR gate in {path}")
    return QuantizedEstimator(tuple(weights), int(gate.group(1)))


def load_mlp_header(path: Path) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    """Loads the grouped MLP weights and SNR gate from an exported header.

    Args:
        path: Path to the exported MLP updater weights header.

    Returns:
        A pair containing the Q7 weight arrays ``W1`` and ``W2``, followed by
        the SNR gate in units of ``snr_ratio_x16``.

    Raises:
        ValueError: If either weight array or the SNR gate is missing.
    """
    source = path.read_text()
    arrays = []
    for index in (1, 2):
        match = re.search(rf"\bkW{index}(?:\[\d+\])+\s*=\s*(\{{.*?\}});", source, re.S)
        if match is None:
            raise ValueError(f"missing MLP weight kW{index} in {path}")
        arrays.append(np.asarray(ast.literal_eval(match.group(1).replace('{', '[').replace('}', ']')),
                                 dtype=np.int8))
    gate = re.search(r"\bkGateSnrRatioX16\s*=\s*(\d+)", source)
    if gate is None:
        raise ValueError(f"missing MLP SNR gate in {path}")
    return (arrays[0], arrays[1]), int(gate.group(1))


def normalize(value: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Applies per-example power-of-two scaling and clamps values to Q15.

    Args:
        value: Input array with one example along axis 0.
        shifts: Signed shift for each example; nonnegative values shift left,
            while negative values shift right.

    Returns:
        An ``int16`` array with the same shape as ``value``.
    """
    output = np.empty(value.shape, dtype=np.int16)
    for index, shift in enumerate(shifts):
        raw = value[index].astype(np.int64)
        raw = raw << int(shift) if shift >= 0 else raw >> int(-shift)
        output[index] = np.clip(raw, -32768, 32767).astype(np.int16)
    return output


def example_indices(symbol_count: int, reference_radius: int,
                    min_reference_symbols: int, target_mode: str = 'averaged'):
    if target_mode not in ('averaged', 'single_symbol'):
        raise ValueError('unknown updater target mode')
    pair_indices = usable_pair_indices(symbol_count, reference_radius, min_reference_symbols)
    indices = pair_indices[pair_indices + 2 < symbol_count]
    references = []
    if target_mode == 'averaged':
        retained = []
        for index in indices:
            current_index = index + 2
            window = np.arange(max(0, current_index - reference_radius),
                               min(symbol_count, current_index + reference_radius + 1))
            window = window[(window < index) | (window > current_index)]
            if len(window) >= min_reference_symbols:
                retained.append(index)
                references.append(window)
        indices = np.asarray(retained, dtype=np.int32)
    return pair_indices, indices, references


def frame_examples(frame, cnn: QuantizedEstimator, reference_radius: int,
                   min_reference_symbols: int,
                   target_mode: str = 'averaged') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds MLP training examples from one decoded-symbol frame.

    The CNN estimate supplies a baseline channel. For each usable symbol pair
    with a following payload symbol, this function encodes four group-specific
    feature vectors and the corresponding Q15 residual targets.

    Args:
        frame: Frame containing decoded samples and the associated CNN corpus.
        cnn: Quantized CNN estimator used to calculate baseline channels.
        reference_radius: Maximum number of reference symbols used on either
            side of a candidate pair.
        min_reference_symbols: Minimum number of surrounding reference symbols
            required for a pair to be used.

    Returns:
        A tuple of features ``(N, 4, 61)``, labels ``(N, 4, 26)``, baseline
        channels ``(N, 52, 2)``, and SNR ratios ``(N,)``. Empty arrays with
        matching trailing dimensions are returned when the frame has no usable
        examples.
    """
    if target_mode not in ('averaged', 'single_symbol'):
        raise ValueError('unknown updater target mode')
    samples = frame.samples
    if samples is None or len(samples) < 3:
        return (np.empty((0, 4, INPUT_SIZE), dtype=np.int16),
                np.empty((0, 4, 26), dtype=np.int16),
                np.empty((0, 52, 2), dtype=np.int32), np.empty(0, dtype=np.uint64))
    pair_indices, indices, references = example_indices(
        len(samples), reference_radius, min_reference_symbols, target_mode)
    count = len(indices)
    if not count:
        return (np.empty((0, 4, INPUT_SIZE), dtype=np.int16),
                np.empty((0, 4, 26), dtype=np.int16),
                np.empty((0, 52, 2), dtype=np.int32), np.empty(0, dtype=np.uint64))
    positions = np.searchsorted(pair_indices, indices)
    received = samples[:, :, 0]
    transmitted = samples[:, :, -1]
    channel = received / transmitted
    amplitude = float(np.median(np.abs(channel)))
    scale = min(12000.0 / amplitude, 30000.0 / float(np.max(np.abs(channel))))
    baseline = integer_estimate(cnn, frame.corpus)[positions]
    current = channel[np.asarray(indices) + 2]
    pilot = np.rint(np.stack((current[:, np.searchsorted(OCCUPIED_BINS, PILOT_BINS)].real,
                               current[:, np.searchsorted(OCCUPIED_BINS, PILOT_BINS)].imag), axis=-1) * scale).astype(np.int32)
    if target_mode == 'averaged':
        reference = np.asarray([
            np.sum(received[window] * np.conj(transmitted[window]), axis=0,
                   dtype=np.complex128) / np.sum(np.abs(transmitted[window]) ** 2, axis=0)
            for window in references])
        target_complex = reference * scale
    else:
        target_complex = current * scale
    target = np.rint(np.stack((target_complex.real, target_complex.imag), axis=-1)).astype(np.int32)
    base = baseline[:, OCCUPIED_BINS]
    maximum = np.maximum(np.max(np.abs(base.astype(np.int64)), axis=(1, 2)),
                         np.max(np.abs(pilot.astype(np.int64)), axis=(1, 2)))
    shifts = 15 - np.asarray([int(value).bit_length() for value in maximum], dtype=np.int32)
    base_q15 = normalize(base, shifts)
    pilot_q15 = normalize(pilot, shifts)
    residual_q15 = normalize(target - base, shifts)
    snr = frame.corpus.snr_ratio_x16[positions]
    features = np.empty((count, 4, INPUT_SIZE), dtype=np.int16)
    labels = np.empty((count, 4, 26), dtype=np.int16)
    for group in range(4):
        half = group // 2
        features[:, group, :52] = base_q15[:, half * 26:(half + 1) * 26].reshape(count, 52)
        delta = pilot_q15.astype(np.int32) - base_q15[:, np.searchsorted(OCCUPIED_BINS, PILOT_BINS)].astype(np.int32)
        features[:, group, 52:60] = np.clip(delta, -32768, 32767).reshape(count, 8)
        features[:, group, 60] = snr_feature_q15(snr)
        labels[:, group] = residual_q15[:, group * 13:(group + 1) * 13].reshape(count, 26)
    return features, labels, base, snr
