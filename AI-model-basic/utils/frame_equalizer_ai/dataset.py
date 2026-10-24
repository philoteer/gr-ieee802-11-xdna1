"""Run the GNU Radio Companion dataset generator and load its output."""

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

try:
    from .constellations import BITS
except ImportError:
    from constellations import BITS


_DIRECTORY = Path(__file__).resolve().parent
_FLOWGRAPH = _DIRECTORY / "data.grc"
_BUILD_DIRECTORY = _DIRECTORY / "build"
_GENERATED = _BUILD_DIRECTORY / "data.py"
_GENERATED_HELPER = _BUILD_DIRECTORY / "data_feature_extractor.py"
_BUILD_STAMP = _BUILD_DIRECTORY / ".data-grc.sha256"
_BUILD_LOCK = _BUILD_DIRECTORY / ".data-grc.lock"
_FRAMES_PER_FLOWGRAPH = 128


@dataclass
class Dataset:
    """Flattened data lanes and complete normalized OFDM-symbol context."""

    features: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    full_features: np.ndarray


def _compile_flowgraph() -> None:
    _BUILD_DIRECTORY.mkdir(exist_ok=True)
    source_digest = hashlib.sha256(_FLOWGRAPH.read_bytes()).hexdigest()
    with _BUILD_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        generated = (_GENERATED, _GENERATED_HELPER)
        if (all(path.is_file() for path in generated) and
                _BUILD_STAMP.is_file() and
                _BUILD_STAMP.read_text().strip() == source_digest):
            return
        subprocess.run(
            ["grcc", "-o", str(_BUILD_DIRECTORY), str(_FLOWGRAPH)],
            cwd=_DIRECTORY,
            check=True,
        )
        if not all(path.is_file() for path in generated):
            raise RuntimeError("grcc did not generate all data flowgraph modules")
        _BUILD_STAMP.write_text(source_digest + "\n")


def generate_dataset(modulation: str, frames: int, symbols_per_frame: int,
                     seed: int) -> Dataset:
    """Generate a frame-disjoint corpus by running ``data.grc`` via GNU Radio."""
    if modulation not in BITS:
        raise ValueError(f"unsupported modulation: {modulation}")
    if frames <= 0 or symbols_per_frame <= 0:
        raise ValueError("frames and symbols_per_frame must be positive")

    _compile_flowgraph()
    with tempfile.TemporaryDirectory(prefix="frame-equalizer-ai-") as directory:
        chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in
            ("features", "labels", "classes", "full_features")
        }
        for chunk_index, frame_offset in enumerate(
                range(0, frames, _FRAMES_PER_FLOWGRAPH)):
            chunk_frames = min(_FRAMES_PER_FLOWGRAPH, frames - frame_offset)
            output = Path(directory) / f"dataset-{chunk_index}.npz"
            environment = os.environ.copy()
            environment.update({
                "DATASET_MODULATION": modulation,
                "DATASET_FRAMES": str(chunk_frames),
                "DATASET_SYMBOLS_PER_FRAME": str(symbols_per_frame),
                "DATASET_SEED": str(seed + chunk_index * 1_000_003),
                "DATASET_OUTPUT": str(output),
            })
            subprocess.run(
                [sys.executable, str(_GENERATED)],
                cwd=_DIRECTORY,
                env=environment,
                check=True,
            )
            with np.load(output, allow_pickle=False) as archive:
                for name in chunks:
                    chunks[name].append(archive[name])
        arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}

    sample_count = frames * symbols_per_frame * 48
    symbol_count = frames * symbols_per_frame
    expected = {
        "features": ((sample_count, 3), np.dtype(np.float32)),
        "labels": ((sample_count, BITS[modulation]), np.dtype(np.float32)),
        "classes": ((sample_count,), np.dtype(np.int64)),
        "full_features": ((symbol_count, 64, 3), np.dtype(np.float32)),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise RuntimeError(
                f"generated {name} has shape/dtype {array.shape}/{array.dtype}; "
                f"expected {shape}/{dtype}")
    return Dataset(**arrays)


def dataset_digest(dataset: Dataset) -> str:
    """Hash generated tensors for deterministic-generation validation."""
    digest = hashlib.sha256()
    for array in (dataset.features, dataset.labels, dataset.classes,
                  dataset.full_features):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
