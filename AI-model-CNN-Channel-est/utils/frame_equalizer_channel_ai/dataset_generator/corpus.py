"""Offline simulated and GNU Radio channel corpus generation."""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.frame_equalizer_channel_ai.common.dataset import (
    ChannelCorpus,
    FFT_SIZE,
    LONG,
    OCCUPIED,
    _shared_q15,
    average_lts,
    estimate_snr_ratio_x16,
    snr_feature_q15,
)


SPLIT_SEEDS = {"train": 101003, "validation": 202009, "test": 303007}
_DIRECTORY = Path(__file__).resolve().parent
_GNU_RADIO_FLOWGRAPH = _DIRECTORY / "data.grc"
_GNU_RADIO_BUILD = _DIRECTORY / "build"
_GNU_RADIO_GENERATED = _GNU_RADIO_BUILD / "data.py"
_GNU_RADIO_STAMP = _GNU_RADIO_BUILD / ".data-grc.sha256"
_GNU_RADIO_LOCK = _GNU_RADIO_BUILD / ".data-grc.lock"
_GNU_RADIO_CHUNK = 512


def _round_complex(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape + (2,), dtype=np.int32)
    output[..., 0] = np.rint(values.real).astype(np.int32)
    output[..., 1] = np.rint(values.imag).astype(np.int32)
    return output


def _build_corpus(observations: np.ndarray, truth: np.ndarray) -> ChannelCorpus:
    samples = len(observations)
    corrected = observations * LONG[None, None, :, None]
    ls = average_lts(corrected[:, 0], corrected[:, 1])
    truth[:, ~OCCUPIED] = 0
    inputs_four, shifts = _shared_q15(corrected)
    snr_ratio_x16 = estimate_snr_ratio_x16(observations)
    inputs = np.empty((samples, 6, FFT_SIZE), dtype=np.int32)
    inputs[:, 0] = inputs_four[:, 0, :, 0]
    inputs[:, 1] = inputs_four[:, 0, :, 1]
    inputs[:, 2] = inputs_four[:, 1, :, 0]
    inputs[:, 3] = inputs_four[:, 1, :, 1]
    inputs[:, 4] = OCCUPIED.astype(np.int32)[None, :] * 32767
    inputs[:, 5] = snr_feature_q15(snr_ratio_x16)[:, None]
    residual = truth.astype(np.int64) - ls.astype(np.int64)
    target = np.empty_like(ls)
    for index, shift in enumerate(shifts):
        value = residual[index] << int(shift) if shift >= 0 else residual[index] >> int(-shift)
        target[index] = np.clip(value, -2147483648, 2147483647).astype(np.int32)
    target[:, ~OCCUPIED] = 0
    return ChannelCorpus(inputs, observations, snr_ratio_x16, ls, truth, target, shifts)


def generate_corpus(samples: int, seed: int, snr_min_db: float = 8.0,
                    snr_max_db: float = 32.0) -> ChannelCorpus:
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


def generate_gnuradio_raw_archive(samples: int, seed: int, snr_min_db: float,
                                  snr_max_db: float, output_path: Path) -> Path:
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        np.savez(output, **generated)
    return output_path


def convert_gnuradio_archive(path: Path) -> ChannelCorpus:
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in ("first", "second", "truth"):
            if name not in archive:
                raise ValueError(f"raw GNU Radio archive is missing {name!r}: {path}")
            arrays[name] = archive[name]
    samples = arrays["first"].shape[0] if arrays["first"].ndim else 0
    for name, value in arrays.items():
        if (value.shape != (samples, FFT_SIZE) or
                not np.issubdtype(value.dtype, np.complexfloating)):
            raise ValueError(
                f"raw GNU Radio array {name!r} must be complex with shape "
                f"(N, {FFT_SIZE}); got {value.shape}/{value.dtype}")
    if samples <= 0:
        raise ValueError("raw GNU Radio archive contains no samples")
    observations = np.stack((_round_complex(arrays["first"]),
                             _round_complex(arrays["second"])), axis=1)
    truth = _round_complex(arrays["truth"] * LONG[None, :])
    observations[:, :, ~OCCUPIED] = 0
    return _build_corpus(observations, truth)


def generate_gnuradio_corpus(samples: int, seed: int, snr_min_db: float = 8.0,
                              snr_max_db: float = 32.0) -> ChannelCorpus:
    with tempfile.TemporaryDirectory(prefix="channel-estimator-gr-") as directory:
        raw_path = Path(directory) / "gnuradio-raw.npz"
        generate_gnuradio_raw_archive(samples, seed, snr_min_db, snr_max_db, raw_path)
        return convert_gnuradio_archive(raw_path)


def split_corpus(split: str, samples: int, run_seed: int, source: str = "original",
                 **kwargs) -> ChannelCorpus:
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split {split!r}")
    seed = SPLIT_SEEDS[split] + run_seed
    if source == "original":
        return generate_corpus(samples, seed, **kwargs)
    if source == "gnuradio":
        return generate_gnuradio_corpus(samples, seed, **kwargs)
    raise ValueError(f"unknown dataset source {source!r}")
