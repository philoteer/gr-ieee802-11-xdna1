"""Export quantized CNN weights for Python and the AIE C++ kernel."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from ..common.quantized import QuantizedEstimator

DEFAULT_CPU_DIRECTORY = Path(__file__).resolve().parents[4] / "gr-ieee802-11-modified/lib/equalizer/model"


def export_float(weights, gate: int, header: Path, namespace: str,
                 source: str = "trained_float32") -> Path:
    arrays = [np.asarray(value, dtype=np.float32) for value in weights]
    digest = hashlib.sha256()
    for value in arrays:
        if not np.all(np.isfinite(value)):
            raise ValueError("cannot export nonfinite weights")
        digest.update(value.astype('<f4').tobytes())
    digest.update(str(gate).encode())
    lines = ["#pragma once", "#include <cstdint>",
             f"namespace {namespace} {{",
             f'constexpr const char* kSourceSha256 = "{digest.hexdigest()}";',
             f'constexpr const char* kWeightSource = "{source}";',
             f"constexpr std::uint64_t kGateSnrRatioX16 = {gate}ull;"]
    for index, value in enumerate(arrays, 1):
        values = [f"{float(v):.9e}f" for v in value.flat]
        rows = [", ".join(values[start:start + 8]) for start in range(0, len(values), 8)]
        lines.append(f"alignas(32) static constexpr float kW{index}[{value.size}] = {{\n"
                     + ",\n".join(rows) + "\n};")
    lines.extend(["}", ""])
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("\n".join(lines))
    return header


def export_float_cnn(model: QuantizedEstimator, header: Path) -> Path:
    return export_float(tuple(value.astype(np.float32) / 128 for value in model.weights),
                        model.gate_snr_ratio_x16, header, "channel_cnn_weights", "dequantized_q7")


def export_trained_cnn(model, gate: int, header: Path) -> Path:
    return export_float(tuple(layer.weight.detach().cpu().numpy() for layer in model.layers),
                        gate, header, "channel_cnn_weights")


def export_trained_updater(models, gate: int, header: Path) -> Path:
    weights = tuple(np.stack([model.layers[index].weight.detach().cpu().numpy()
                             for model in models]) for index in (0, 2))
    return export_float(weights, gate, header, "channel_updater_weights")


def export_float_updater(weights, gate: int, header: Path) -> Path:
    first, second = weights
    return export_float((first[:, :61, :].transpose(0, 2, 1).astype(np.float32) / 128,
                         second[:, :, :26].transpose(0, 2, 1).astype(np.float32) / 128),
                        gate, header, "channel_updater_weights", "dequantized_q7")


def _initializer(values: np.ndarray) -> str:
    """Render an integer NumPy array as nested C++ initializer braces.

    Args:
        values: One- or multi-dimensional integer array.

    Returns:
        C++ initializer-list text preserving the input array nesting.
    """
    if values.ndim == 1:
        return "{" + ", ".join(str(int(value)) for value in values) + "}"
    return "{" + ",\n".join(_initializer(value) for value in values) + "}"


def export_model(model: QuantizedEstimator, header: Path,
                 artifact_prefix: Path, cpu_header: Path | None = None) -> tuple[Path, ...]:
    """Write the quantized model as C++, NPZ, and JSON artifacts.

    The C++ header is the deployed AIE representation.  The NPZ and JSON files
    support host-side validation, inspection, and exact export-reload parity
    tests.

    Args:
        model: Quantized estimator containing the three int8 weight tensors.
        header: Destination C++ header consumed by ``kernel/channel_estimator/cnn.h``.
        artifact_prefix: Prefix used for the companion ``.npz`` and ``.json``
            files.

    Returns:
        Tuple containing the header, NPZ, and JSON paths in that order.
    """
    arrays = {f"w{index}": value for index, value in enumerate(model.weights, 1)}
    arrays["gate_snr_ratio_x16"] = np.asarray(model.gate_snr_ratio_x16, dtype=np.uint64)
    header.parent.mkdir(parents=True, exist_ok=True)
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz = artifact_prefix.with_suffix(".npz")
    json_path = artifact_prefix.with_suffix(".json")
    np.savez(npz, **arrays)
    json_path.write_text(json.dumps({key: value.tolist() for key, value in arrays.items()},
                                    indent=2) + "\n")
    lines = [
        "#ifndef FRAME_EQUALIZER_CHANNEL_AI_WEIGHTS_H",
        "#define FRAME_EQUALIZER_CHANNEL_AI_WEIGHTS_H",
        "#include <stdint.h>",
        "namespace frame_equalizer_channel_ai {",
        "constexpr unsigned kWeightFractionalBits = 7;",
        "constexpr unsigned kActivationFractionalBits = 15;",
        f"constexpr uint64_t kLsGateSnrRatioX16 = {model.gate_snr_ratio_x16}u;",
    ]
    for index, weights in enumerate(model.weights, 1):
        shape = "".join(f"[{size}]" for size in weights.shape)
        lines.append(f"alignas(32) static constexpr int8_t kW{index}{shape} = "
                     f"{_initializer(weights)};")
    lines += ["} // namespace frame_equalizer_channel_ai", "#endif", ""]
    header.write_text("\n".join(lines))
    paths = (header, npz, json_path)
    if cpu_header is not None:
        paths += (export_float_cnn(model, cpu_header),)
    return paths
