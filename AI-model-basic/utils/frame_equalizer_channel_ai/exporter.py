"""Export quantized CNN weights for Python and the AIE C++ kernel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .quantize import QuantizedEstimator
except ImportError:
    from quantize import QuantizedEstimator


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
                 artifact_prefix: Path) -> tuple[Path, Path, Path]:
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
    ]
    for index, weights in enumerate(model.weights, 1):
        shape = "".join(f"[{size}]" for size in weights.shape)
        lines.append(f"alignas(32) static constexpr int8_t kW{index}{shape} = "
                     f"{_initializer(weights)};")
    lines += ["} // namespace frame_equalizer_channel_ai", "#endif", ""]
    header.write_text("\n".join(lines))
    return header, npz, json_path
