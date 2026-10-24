"""Generate deterministic channel-estimation training and test corpora.

The corpus models two repeated 802.11 long-training symbols transmitted
through a random causal tapped-delay-line channel.  It stores both the noisy
observations and the rounded ground-truth frequency response so that channel
NMSE, equalization EVM, floating-point inference, and integer inference can be
evaluated from the same samples.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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
# Independent deterministic seed bases for the three corpus partitions.
SPLIT_SEEDS = {"train": 101003, "validation": 202009, "test": 303007}
_DIRECTORY = Path(__file__).resolve().parent
_GNU_RADIO_FLOWGRAPH = _DIRECTORY / "data.grc"
_GNU_RADIO_BUILD = _DIRECTORY / "build"
_GNU_RADIO_GENERATED = _GNU_RADIO_BUILD / "data.py"
_GNU_RADIO_STAMP = _GNU_RADIO_BUILD / ".data-grc.sha256"
_GNU_RADIO_LOCK = _GNU_RADIO_BUILD / ".data-grc.lock"
_GNU_RADIO_CHUNK = 512


@dataclass(frozen=True)
class ChannelCorpus:
    """Container for one deterministic channel-estimation corpus.

    Attributes:
        inputs_q15: Model inputs with shape ``(N, 5, 64)`` in signed Q15.
            The first four planes are real and imaginary LTS components; the
            fifth plane is the occupied-subcarrier mask.
        observations: Raw two-LTS observations with shape ``(N, 2, 64, 2)``.
        ls: Rounded two-LTS least-squares estimates with shape ``(N, 64, 2)``.
        truth: Rounded channel responses with shape ``(N, 64, 2)``.
        target_q15: Residual targets in the normalized Q15 domain.
        normalization_shift: Per-example power-of-two input normalization
            shifts used by both the reference and AIE implementations.
    """

    inputs_q15: np.ndarray
    observations: np.ndarray
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
                 ls=self.ls, truth=self.truth, target_q15=self.target_q15,
                 normalization_shift=self.normalization_shift)


def _round_complex(values: np.ndarray) -> np.ndarray:
    """Convert complex floating-point samples to integer real/imaginary pairs.

    Args:
        values: Complex-valued samples of arbitrary leading shape.

    Returns:
        An int32 array with one trailing dimension of length two containing
        rounded real and imaginary components.
    """
    output = np.empty(values.shape + (2,), dtype=np.int32)
    output[..., 0] = np.rint(values.real).astype(np.int32)
    output[..., 1] = np.rint(values.imag).astype(np.int32)
    return output


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


def _build_corpus(observations: np.ndarray, truth: np.ndarray) -> ChannelCorpus:
    samples = len(observations)
    corrected = observations * LONG[None, None, :, None]
    ls = average_lts(corrected[:, 0], corrected[:, 1])
    truth[:, ~OCCUPIED] = 0
    inputs_four, shifts = _shared_q15(corrected)
    inputs = np.empty((samples, 5, FFT_SIZE), dtype=np.int32)
    inputs[:, 0] = inputs_four[:, 0, :, 0]
    inputs[:, 1] = inputs_four[:, 0, :, 1]
    inputs[:, 2] = inputs_four[:, 1, :, 0]
    inputs[:, 3] = inputs_four[:, 1, :, 1]
    inputs[:, 4] = OCCUPIED.astype(np.int32)[None, :] * 32767
    residual = truth.astype(np.int64) - ls.astype(np.int64)
    target = np.empty_like(ls)
    for index, shift in enumerate(shifts):
        value = residual[index] << int(shift) if shift >= 0 else residual[index] >> int(-shift)
        target[index] = np.clip(value, -2147483648, 2147483647).astype(np.int32)
    target[:, ~OCCUPIED] = 0
    return ChannelCorpus(inputs, observations, ls, truth, target, shifts)


def generate_corpus(samples: int, seed: int, snr_min_db: float = 8.0,
                    snr_max_db: float = 32.0) -> ChannelCorpus:
    """Generate independent noisy LTS observations and channel labels.

    Each sample uses a random causal channel of one to sixteen taps, a random
    channel gain, and a uniformly sampled SNR.  The returned arrays use the
    same FFT ordering, long-training signs, masking, rounding, and
    normalization conventions as the estimator kernel.

    Args:
        samples: Number of independent channel realizations to generate.
        seed: NumPy random seed for reproducible generation.
        snr_min_db: Inclusive lower bound of the sampled SNR range in dB.
        snr_max_db: Inclusive upper bound of the sampled SNR range in dB.

    Returns:
        A :class:`ChannelCorpus` containing observations, LS estimates, truth,
        and normalized CNN residual targets.

    Raises:
        ValueError: If ``samples`` is not positive or the SNR bounds are
            reversed.
    """
    if samples <= 0 or snr_min_db > snr_max_db:
        raise ValueError("invalid corpus size or SNR range")
    rng = np.random.default_rng(seed)
    truth_complex = np.empty((samples, FFT_SIZE), dtype=np.complex128)
    for index in range(samples):
        length = int(rng.integers(1, 17))
        delay = np.arange(length)
        decay = np.exp(-delay / rng.uniform(2.0, 6.0))
        taps = (rng.normal(size=length) + 1j * rng.normal(size=length)) * decay
        taps /= np.sqrt(np.sum(np.abs(taps) ** 2))
        gain = 10.0 ** rng.uniform(-0.15, 0.15)
        truth_complex[index] = np.fft.fftshift(np.fft.fft(taps * gain, FFT_SIZE))

    amplitude = rng.integers(7000, 24001, size=(samples, 1))
    truth_scaled = truth_complex * amplitude
    snr_db = rng.uniform(snr_min_db, snr_max_db, size=(samples, 1))
    noise_sigma = amplitude / np.sqrt(2.0 * 10.0 ** (snr_db / 10.0))
    observations = np.empty((samples, 2, FFT_SIZE, 2), dtype=np.int32)
    for lts in range(2):
        noise = (rng.normal(size=(samples, FFT_SIZE)) +
                 1j * rng.normal(size=(samples, FFT_SIZE))) * noise_sigma
        received = truth_scaled * LONG[None, :] + noise
        received[:, ~OCCUPIED] = 0.0
        observations[:, lts] = _round_complex(received)

    truth = _round_complex(truth_scaled)
    return _build_corpus(observations, truth)


def _compile_gnuradio_flowgraph() -> None:
    _GNU_RADIO_BUILD.mkdir(exist_ok=True)
    source_digest = hashlib.sha256(_GNU_RADIO_FLOWGRAPH.read_bytes()).hexdigest()
    with _GNU_RADIO_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (_GNU_RADIO_GENERATED.is_file() and _GNU_RADIO_STAMP.is_file() and
                _GNU_RADIO_STAMP.read_text().strip() == source_digest):
            return
        subprocess.run(
            ["grcc", "-o", str(_GNU_RADIO_BUILD), str(_GNU_RADIO_FLOWGRAPH)],
            cwd=_DIRECTORY,
            check=True,
        )
        if not _GNU_RADIO_GENERATED.is_file():
            raise RuntimeError("grcc did not generate the channel dataset flowgraph")
        _GNU_RADIO_STAMP.write_text(source_digest + "\n")


def generate_gnuradio_corpus(samples: int, seed: int, snr_min_db: float = 8.0,
                              snr_max_db: float = 32.0) -> ChannelCorpus:
    if samples <= 0 or snr_min_db > snr_max_db:
        raise ValueError("invalid corpus size or SNR range")
    _compile_gnuradio_flowgraph()
    rng = np.random.default_rng(seed)
    arrays: dict[str, list[np.ndarray]] = {
        "first": [], "second": [], "truth": [],
    }
    with tempfile.TemporaryDirectory(prefix="channel-estimator-gr-") as directory:
        for chunk_index, offset in enumerate(range(0, samples, _GNU_RADIO_CHUNK)):
            count = min(_GNU_RADIO_CHUNK, samples - offset)
            output = Path(directory) / f"dataset-{chunk_index}.npz"
            environment = os.environ.copy()
            environment.update({
                "DATASET_SAMPLES": str(count),
                "DATASET_SEED": str(seed + chunk_index * 1_000_003),
                "DATASET_SNR_DB": str(rng.uniform(snr_min_db, snr_max_db)),
                "DATASET_AMPLITUDE": str(int(rng.integers(7000, 24001))),
                "DATASET_OUTPUT": str(output),
            })
            subprocess.run(
                [sys.executable, str(_GNU_RADIO_GENERATED)],
                cwd=_DIRECTORY,
                env=environment,
                check=True,
            )
            with np.load(output, allow_pickle=False) as archive:
                for name in arrays:
                    value = archive[name]
                    if value.shape != (count, FFT_SIZE) or value.dtype != np.complex64:
                        raise RuntimeError(
                            f"generated {name} has shape/dtype {value.shape}/{value.dtype}; "
                            f"expected {(count, FFT_SIZE)}/complex64")
                    arrays[name].append(value)
    generated = {name: np.concatenate(parts) for name, parts in arrays.items()}
    observations = np.stack((_round_complex(generated["first"]),
                             _round_complex(generated["second"])), axis=1)
    truth = _round_complex(generated["truth"] * LONG[None, :])
    observations[:, :, ~OCCUPIED] = 0
    return _build_corpus(observations, truth)


def concatenate_corpora(*corpora: ChannelCorpus) -> ChannelCorpus:
    if not corpora:
        raise ValueError("at least one corpus is required")
    return ChannelCorpus(*(np.concatenate(values, axis=0) for values in zip(*(
        (corpus.inputs_q15, corpus.observations, corpus.ls, corpus.truth,
         corpus.target_q15, corpus.normalization_shift) for corpus in corpora))))


def corpus_digest(corpus: ChannelCorpus) -> str:
    """Compute a stable SHA-256 digest over every corpus array.

    Args:
        corpus: Corpus whose arrays should be fingerprinted.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for value in (corpus.inputs_q15, corpus.observations, corpus.ls, corpus.truth,
                  corpus.target_q15, corpus.normalization_shift):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def split_corpus(split: str, samples: int, run_seed: int, source: str = "original",
                 **kwargs) -> ChannelCorpus:
    """Generate a corpus partition using its independent seed base.

    Args:
        split: Partition name, one of ``train``, ``validation``, or ``test``.
        samples: Number of samples to generate.
        run_seed: User-selected seed mixed with the partition seed.
        **kwargs: Additional keyword arguments forwarded to
            :func:`generate_corpus`.

    Returns:
        The requested deterministic corpus partition.

    Raises:
        ValueError: If ``split`` is not a known partition name.
    """
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split {split!r}")
    seed = SPLIT_SEEDS[split] + run_seed
    if source == "original":
        return generate_corpus(samples, seed, **kwargs)
    if source == "gnuradio":
        return generate_gnuradio_corpus(samples, seed, **kwargs)
    if source == "both":
        if samples < 2:
            raise ValueError("combined datasets require at least two samples")
        original_samples = (samples + 1) // 2
        return concatenate_corpora(
            generate_corpus(original_samples, seed, **kwargs),
            generate_gnuradio_corpus(samples - original_samples, seed, **kwargs),
        )
    raise ValueError(f"unknown dataset source {source!r}")
