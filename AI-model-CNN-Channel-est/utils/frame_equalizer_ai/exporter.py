"""Pack four quantized demappers into NPZ/JSON and a standalone C++ header.

The AIE equalizer engine selects a learned model by modulation index, so the
four quantized models are packed into shared dense arrays with a fixed model
axis.  The number of hidden layers comes from the models themselves; hidden
layers keep their full shape and only the output rows are padded, because each
modulation has a different number of output bits (``bit_counts`` records the
valid row count).  The same payload is written three ways:

    - ``*.npz``: arrays for the Python validation path.
    - ``*.json``: human-readable mirror (nested lists).
    - ``*.h``: ``constexpr`` C++ arrays for the AIE build, named ``kW1`` ...
      ``kW<hidden+1>`` so the selected model header under
      ``kernel/demodulator/model/`` can consume them unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .constellations import MODULATIONS
    from .quantize import INPUT_FRACTIONAL_BITS, INPUT_SCALE, QuantizedModel
except ImportError:
    from constellations import MODULATIONS
    from quantize import INPUT_FRACTIONAL_BITS, INPUT_SCALE, QuantizedModel


def _validate_architecture(models: dict[str, QuantizedModel]) -> int:
    """Return the shared hidden-layer count or reject mixed architectures.

    Args:
        models: Quantized models keyed by modulation name.

    Returns:
        Number of hidden layers shared by every model.

    Raises:
        ValueError: If the models have different depths, hidden sizes, or
            output widths.
    """
    reference = models[MODULATIONS[0]]
    depth = len(reference.hidden_weights)
    hidden_size = reference.hidden_weights[0].shape[0]
    expected_hidden = [(hidden_size, 3)] + [(hidden_size, hidden_size)] * (depth - 1)
    for model in models.values():
        if ([weights.shape for weights in model.hidden_weights] != expected_hidden or
                model.output_weights.shape != (model.bit_count, hidden_size)):
            raise ValueError("all exported models must share one architecture")
    return depth


def pack_models(models: dict[str, QuantizedModel]) -> dict[str, np.ndarray]:
    """Pad output rows only; hidden matrices always use their complete shape.

    All models must share one architecture.  The returned dictionary uses a
    four-entry model axis ordered like :data:`constellations.MODULATIONS`:

        - ``w1``: ``(4, hidden, 3)`` int8 first-layer weights.
        - ``w2`` ... ``w<depth>``: ``(4, hidden, hidden)`` int8 hidden-layer
          weights, present only for multi-hidden architectures.
        - ``w<depth+1>``: ``(4, 6, hidden)`` int8 output weights, with rows at
          and beyond ``bit_count`` left zero.
        - ``shifts``: ``(4, depth)`` int32 activation right shifts.
        - ``bit_counts``: ``(4,)`` int32 output bit counts.
        - ``hidden_layers``: scalar int32 hidden-layer count.
        - ``input_fractional_bits`` / ``input_scale``: the Q-format input
          contract as scalar int32 values.

    Args:
        models: Quantized models keyed by modulation name.

    Returns:
        The packed array dictionary.

    Raises:
        ValueError: If the models do not all share one architecture.
    """
    depth = _validate_architecture(models)
    hidden_size = models[MODULATIONS[0]].hidden_weights[0].shape[0]
    output_size = max(model.bit_count for model in models.values())
    packed = {
        "w1": np.zeros((4, hidden_size, 3), np.int8),
        "shifts": np.zeros((4, depth), np.int32),
        "bit_counts": np.zeros(4, np.int32),
        "hidden_layers": np.asarray(depth, np.int32),
        "input_fractional_bits": np.asarray(INPUT_FRACTIONAL_BITS, np.int32),
        "input_scale": np.asarray(INPUT_SCALE, np.int32),
    }
    for index in range(2, depth + 1):
        packed[f"w{index}"] = np.zeros((4, hidden_size, hidden_size), np.int8)
    packed[f"w{depth + 1}"] = np.zeros((4, output_size, hidden_size), np.int8)
    for i, name in enumerate(MODULATIONS):
        model = models[name]
        for index, weights in enumerate(model.hidden_weights):
            packed[f"w{index + 1}"][i] = weights
        # Only the first bit_count output rows are meaningful; the rest stay zero.
        packed[f"w{depth + 1}"][i, :model.bit_count] = model.output_weights
        packed["shifts"][i] = model.shifts
        packed["bit_counts"][i] = model.bit_count
    return packed


def _cpp_array(values: np.ndarray) -> str:
    """Render a NumPy array as a nested C brace initializer.

    Elements become decimal text; 1-D arrays become ``{a, b, ...}``; higher
    dimensions recurse with one brace level per axis and a newline between
    inner entries, matching the checked-in header style.

    Args:
        values: Array-like to render (1-D or deeper).

    Returns:
        The initializer text without a trailing semicolon.
    """
    if values.ndim == 1:
        return "{" + ", ".join(str(int(v)) for v in values) + "}"
    return "{" + ",\n".join(_cpp_array(v) for v in values) + "}"


def export_models(models: dict[str, QuantizedModel], header_path: Path,
                  artifact_prefix: Path) -> tuple[Path, Path, Path]:
    """Write simple row-major C arrays and equivalent machine-readable files.

    Creates the header's and artifact prefix's parent directories as needed,
    writes ``<prefix>.npz`` and ``<prefix>.json`` beside each other, and
    writes the guarded C++ header declaring the model constants and weight
    arrays in ``namespace frame_equalizer_ai``.  The weight arrays are
    row-major and aligned to ``aie::vector_decl_align`` so the AIE build can
    load them directly.  ``kHiddenLayers`` identifies the architecture so the
    selected model header can reject a mismatched weights file at compile
    time.

    Args:
        models: Quantized models keyed by modulation name.
        header_path: Destination ``.h`` file for the C++ arrays.
        artifact_prefix: Destination path whose suffix selects the NPZ and
            JSON names (for example
            ``artifacts/frame_equalizer_ai_mlp_16_weights``).

    Returns:
        ``(header_path, npz_path, json_path)`` in that order.
    """
    packed = pack_models(models)
    depth = int(packed["hidden_layers"])
    hidden_size = packed["w1"].shape[1]
    output_size = packed[f"w{depth + 1}"].shape[1]
    header_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = artifact_prefix.with_suffix(".npz")
    json_path = artifact_prefix.with_suffix(".json")
    np.savez(npz_path, **packed)
    json_path.write_text(json.dumps({key: value.tolist() for key, value in packed.items()},
                                    indent=2) + "\n")
    text = """// Generated by frame_equalizer_ai; row-major signed int8 weights.
// Inputs are Q(kInputFractionalBits) signed fixed-point values.
#ifndef FRAME_EQUALIZER_AI_WEIGHTS_H
#define FRAME_EQUALIZER_AI_WEIGHTS_H
#include <stdint.h>
#include <aie_api/aie.hpp>
namespace frame_equalizer_ai {
"""
    text += "constexpr unsigned kModelCount = 4;\n"
    text += f"constexpr unsigned kInputFeatures = {packed['w1'].shape[2]};\n"
    text += f"constexpr unsigned kHiddenLayers = {depth};\n"
    text += f"constexpr unsigned kHiddenSize = {hidden_size};\n"
    text += "constexpr unsigned kMaxOutputBits = 6;\n"
    text += (f"constexpr unsigned kInputFractionalBits = "
             f"{int(packed['input_fractional_bits'])};\n")
    # AIE vector loads require the natural vector alignment.
    align = "alignas(aie::vector_decl_align) "
    for index in range(1, depth + 1):
        weights = packed[f"w{index}"]
        text += (f"{align}static constexpr int8_t kW{index}[4]"
                 f"[{weights.shape[1]}][{weights.shape[2]}] = "
                 f"{_cpp_array(weights)};\n")
    output = packed[f"w{depth + 1}"]
    text += (f"{align}static constexpr int8_t kW{depth + 1}[4]"
             f"[{output.shape[1]}][{output.shape[2]}] = "
             f"{_cpp_array(output)};\n")
    text += f"static constexpr int32_t kShifts[4][{depth}] = {_cpp_array(packed['shifts'])};\n"
    text += f"static constexpr int32_t kBitCounts[4] = {_cpp_array(packed['bit_counts'])};\n"
    text += "} // namespace frame_equalizer_ai\n#endif\n"
    header_path.write_text(text)
    return header_path, npz_path, json_path
