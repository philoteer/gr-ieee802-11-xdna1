"""Channel-estimator corpus schema, input processing, and digest utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


# The estimator operates on one FFT-shifted 802.11 OFDM symbol.
FFT_SIZE = 64
# Frequency-domain BPSK signs of the two repeated long-training symbols.
LONG = np.asarray([
    0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
    1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
    0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
], dtype=np.int32)
# True for the 52 occupied subcarriers, including the four pilot carriers.
OCCUPIED = (LONG != 0)
SNR_LOG2_RANGE_Q8 = 3402


def usable_pair_indices(symbol_count: int, radius: int, minimum: int) -> np.ndarray:
    return np.asarray([index for index in range(symbol_count - 1)
                       if min(index, radius) + min(symbol_count - index - 2, radius) >= minimum],
                      dtype=np.int32)


@dataclass(frozen=True)
class ChannelCorpus:
    """Container for one deterministic channel-estimation corpus.

    Attributes:
        inputs_q15: Model inputs with shape ``(N, 6, 64)`` in signed Q15.
            The first four planes are real and imaginary LTS components; the
            fifth plane is the occupied-subcarrier mask, and the sixth is the
            SNR feature.
        observations: Raw two-LTS observations with shape ``(N, 2, 64, 2)``.
        ls: Rounded two-LTS least-squares estimates with shape ``(N, 64, 2)``.
        truth: Rounded channel responses with shape ``(N, 64, 2)``.
        target_q15: Residual targets in the normalized Q15 domain.
        normalization_shift: Per-example power-of-two input normalization
            shifts used by both the reference and AIE implementations.
    """

    inputs_q15: np.ndarray
    observations: np.ndarray
    snr_ratio_x16: np.ndarray
    ls: np.ndarray
    truth: np.ndarray
    target_q15: np.ndarray
    normalization_shift: np.ndarray

    def save(self, path: Path) -> None:
        """Save all corpus arrays to a compressed NumPy archive.

        Args:
            path: Destination path for the ``.npz`` archive.  Parent
                directories are created when necessary.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, inputs_q15=self.inputs_q15, observations=self.observations,
                 snr_ratio_x16=self.snr_ratio_x16, ls=self.ls, truth=self.truth, target_q15=self.target_q15,
                 normalization_shift=self.normalization_shift)

    @classmethod
    def load(cls, path: Path) -> ChannelCorpus:
        with np.load(path, allow_pickle=False) as archive:
            corpus = cls(*(archive[name] for name in (
                "inputs_q15", "observations", "snr_ratio_x16", "ls", "truth",
                "target_q15", "normalization_shift")))
        samples = corpus.inputs_q15.shape[0]
        expected_shapes = {
            "inputs_q15": (samples, 6, FFT_SIZE),
            "observations": (samples, 2, FFT_SIZE, 2),
            "snr_ratio_x16": (samples,),
            "ls": (samples, FFT_SIZE, 2),
            "truth": (samples, FFT_SIZE, 2),
            "target_q15": (samples, FFT_SIZE, 2),
            "normalization_shift": (samples,),
        }
        for name, shape in expected_shapes.items():
            if getattr(corpus, name).shape != shape:
                raise ValueError(f"invalid {name} shape in dataset {path}")
        return corpus


def average_lts(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Average two LTS observations using the AIE symmetric-floor contract.

    The correction before the arithmetic right shift preserves the production
    estimator's behavior for negative odd sums.

    Args:
        first: First sign-corrected LTS array.
        second: Second sign-corrected LTS array with the same shape.

    Returns:
        Int32 elementwise average with the input shape.
    """
    total = first.astype(np.int64) + second.astype(np.int64)
    return ((total + (total < 0)) >> 1).astype(np.int32)


def _shared_q15(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each example into the signed Q15 input range.

    A single power-of-two shift is selected for all channels, bins, and LTS
    components in each example.  Sharing the shift preserves the relative
    real/imaginary amplitudes consumed by the CNN.

    Args:
        values: Integer tensor with shape ``(N, 2, 64, 2)``.

    Returns:
        A tuple containing the normalized int32 tensor and one signed shift per
        example.
    """
    maximum = np.max(np.abs(values.astype(np.int64)), axis=(1, 2, 3))
    bits = np.asarray([int(value).bit_length() for value in maximum], dtype=np.int32)
    shifts = 15 - bits
    output = np.empty_like(values)
    for index, shift in enumerate(shifts):
        wide = values[index].astype(np.int64)
        shifted = wide << int(shift) if shift >= 0 else wide >> int(-shift)
        output[index] = np.clip(shifted, -32768, 32767).astype(np.int32)
    return output, shifts


def estimate_snr_ratio_x16(observations: np.ndarray) -> np.ndarray:
    signal = observations[:, 0].astype(np.int64) + observations[:, 1].astype(np.int64)
    noise = observations[:, 0].astype(np.int64) - observations[:, 1].astype(np.int64)
    signal_power = np.sum(signal[:, OCCUPIED] ** 2, axis=(1, 2), dtype=np.int64)
    noise_power = np.sum(noise[:, OCCUPIED] ** 2, axis=(1, 2), dtype=np.int64)
    return ((signal_power << 4) // (noise_power + 1)).astype(np.uint64)


def snr_feature_q15(snr_ratio_x16: np.ndarray) -> np.ndarray:
    """Map the scaled ratio to a bounded logarithmic Q15 feature.

    The leading-bit and mantissa approximation implements log2(ratio / 32)
    without floating-point arithmetic. Values at or below the low-SNR anchor
    map to zero, while the 40 dB anchor saturates at the maximum Q15 value.
    """
    # Allocate one signed Q15 feature value for every input SNR value.
    output = np.empty(np.asarray(snr_ratio_x16).shape, dtype=np.int32)
    # Process each scalar independently so the integer operations match AIE.
    for index, value in np.ndenumerate(np.asarray(snr_ratio_x16, dtype=np.uint64)):
        # Convert NumPy's unsigned scalar into an unbounded Python integer.
        raw = int(value)
        # 32 / 32 = 1, so the logarithmic feature is zero at this anchor.
        if raw <= 32:
            # Clamp low or invalid measured ratios to the bottom of Q15.
            output[index] = 0
            # No logarithm is needed after the lower clamp.
            continue
        # 320000 / 32 = 10000, which is the 40 dB upper training anchor.
        if raw >= 320000:
            # Clamp high ratios to the top of signed Q15.
            output[index] = 32767
            # No logarithm is needed after the upper clamp.
            continue
        # Find floor(log2(raw)); raw is positive because the low clamp passed.
        exponent = raw.bit_length() - 1
        # Keep an eight-bit fractional mantissa with 256 representing 1.0.
        mantissa_q8 = raw >> (exponent - 8) if exponent >= 8 else raw << (8 - exponent)
        # Approximate log2(raw / 32) in Q8 and prevent negative features.
        log2_snr_q8 = max(0, exponent * 256 + mantissa_q8 - 256 - 5 * 256)
        # Scale the 0-40 dB logarithmic range into signed Q15.
        output[index] = min(32767, log2_snr_q8 * 32767 // SNR_LOG2_RANGE_Q8)
    # Return the feature tensor with the same shape as the input ratios.
    return output


def snr_gate_candidates() -> list[int]:
    return [0] + [int(round(32.0 * 10.0 ** (db / 10.0)))
                  for db in np.arange(0.0, 40.5, 0.5)] + [2**64 - 1]


def corpus_digest(corpus: ChannelCorpus) -> str:
    """Compute a stable SHA-256 digest over every corpus array.

    Args:
        corpus: Corpus whose arrays should be fingerprinted.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for value in (corpus.inputs_q15, corpus.observations, corpus.snr_ratio_x16, corpus.ls, corpus.truth,
                   corpus.target_q15, corpus.normalization_shift):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def concatenate_corpora(*corpora: ChannelCorpus) -> ChannelCorpus:
    if not corpora:
        raise ValueError("at least one corpus is required")
    return ChannelCorpus(*(np.concatenate(values, axis=0) for values in zip(*(
        (corpus.inputs_q15, corpus.observations, corpus.snr_ratio_x16, corpus.ls,
         corpus.truth, corpus.target_q15, corpus.normalization_shift)
        for corpus in corpora))))
