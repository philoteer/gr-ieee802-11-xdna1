#!/usr/bin/env python3
"""Command-line training, export, and deterministic validation entry point.

Subcommands:
    train: seed the RNGs, generate per-modulation train/validation/test
        datasets, fit one network per modulation for the selected model
        architecture, quantize it, evaluate the float and quantized
        demappers, and write the checkpoint, JSON report, and exported weights
        artifacts.
    export: load a checkpoint and re-write the header/NPZ/JSON artifacts
        without retraining.
    validate: reload a checkpoint, regenerate its held-out corpus, and prove
        that data generation is reproducible, recorded metrics still match,
        and the exported integer arrays reproduce the quantized reference.

Data splits use fixed seed offsets combined with the run seed, so a
checkpoint's embedded ``metadata`` fully determines its evaluation corpus.
This docstring is also the ``--help`` description.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

try:
    from .constellations import BITS, MODULATIONS, load_constellations
    from .dataset import dataset_digest, generate_dataset
    from .exporter import export_models
    from .model import (ARCHITECTURES, DEFAULT_ARCHITECTURE, JointIQMLP,
                        predict_classes, train_model)
    from .quantize import (INPUT_FRACTIONAL_BITS, INPUT_SCALE, QuantizedModel,
                           integer_classes, integer_logits, quantize_model)
except ImportError:
    from constellations import BITS, MODULATIONS, load_constellations
    from dataset import dataset_digest, generate_dataset
    from exporter import export_models
    from model import (ARCHITECTURES, DEFAULT_ARCHITECTURE, JointIQMLP,
                       predict_classes, train_model)
    from quantize import (INPUT_FRACTIONAL_BITS, INPUT_SCALE, QuantizedModel,
                          integer_classes, integer_logits, quantize_model)

# Fixed per-split seed bases.  Each modulation adds its model index and the
# run's user seed, so the three splits never overlap for any seed.
SPLIT_SEEDS = {"train": 11003, "validation": 22003, "test": 33013}
# Generated architecture headers are installed directly into the kernel tree.
MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "kernel" / "demodulator" / "model"


def _rates(prediction: np.ndarray, truth: np.ndarray, bits: int) -> dict[str, float]:
    """Compute BER and SER for class-code predictions.

    BER counts mismatched bits over the LSB-first class codes (total bit
    count = samples x ``bits``); SER counts whole-symbol mismatches.

    Args:
        prediction: Predicted class codes, shape ``(N,)``.
        truth: True class codes, shape ``(N,)``.
        bits: Coded bits per symbol.

    Returns:
        ``{"ber": float, "ser": float}``.
    """
    xor = prediction ^ truth
    errors = sum(int(((xor >> bit) & 1).sum()) for bit in range(bits))
    return {"ber": errors / (len(truth) * bits), "ser": float(np.mean(prediction != truth))}


def _save_checkpoint(path: Path, float_models, quant_models, metadata) -> None:
    """Serialize float state dicts, quantized arrays, and metadata to a file.

    The payload schema is ``{"metadata", "float_models", "quant_models"}``
    with one entry per modulation in :data:`constellations.MODULATIONS`.  The
    quantized entries store only their small array/scalar fields, so a
    checkpoint can be reloaded without pickling the dataclass definition.

    Args:
        path: Destination ``.pt`` file; parent directories are created.
        float_models: Float models keyed by modulation name.
        quant_models: Quantized models keyed by modulation name.
        metadata: Training report embedded in the checkpoint so ``validate``
            can replay the exact corpus and settings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "float_models": {}, "quant_models": {}}
    for name in MODULATIONS:
        payload["float_models"][name] = float_models[name].state_dict()
        q = quant_models[name]
        payload["quant_models"][name] = {
            "bit_count": q.bit_count,
            "hidden_weights": [np.asarray(weights) for weights in q.hidden_weights],
            "output_weights": q.output_weights,
            "shifts": list(q.shifts),
        }
    torch.save(payload, path)


def _load_checkpoint(path: Path):
    """Restore models, metadata, and architecture from a checkpoint.

    Rebuilds each float model with the hidden sizes recorded in the metadata
    and reconstructs the quantized models from their arrays.

    Args:
        path: Checkpoint path.

    Returns:
        ``(float_models, quant_models, metadata, architecture)`` where
        ``architecture`` is the :class:`model.Architecture` that produced the
        checkpoint.

    Raises:
        SystemExit: If the checkpoint has no architecture metadata or names an
            unknown architecture.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    recorded = metadata.get("architecture")
    if recorded is None:
        raise SystemExit(
            f"{path} has no architecture metadata; retrain with the current cli")
    name = recorded["name"]
    if name not in ARCHITECTURES:
        raise SystemExit(f"checkpoint references unknown architecture {name!r}")
    hidden_sizes = tuple(recorded["hidden_sizes"])
    floats, quants = {}, {}
    for modulation in MODULATIONS:
        floats[modulation] = JointIQMLP(BITS[modulation], hidden_sizes)
        floats[modulation].load_state_dict(payload["float_models"][modulation])
        floats[modulation].eval()
        entry = payload["quant_models"][modulation]
        quants[modulation] = QuantizedModel(
            entry["bit_count"], list(entry["hidden_weights"]),
            entry["output_weights"], list(entry["shifts"]))
    return floats, quants, metadata, ARCHITECTURES[name]


def _resolve_paths(args: argparse.Namespace, architecture) -> None:
    """Fill unset output paths with architecture-specific defaults.

    The header default points at the kernel model directory so a fresh export
    installs the weights where the compile-time model selection expects them.
    Attributes the subcommand does not define (for example ``report`` for
    ``export``) are left alone.

    Args:
        args: Parsed subcommand namespace; ``None`` paths are replaced.
        architecture: :class:`model.Architecture` selected by the run.
    """
    stem = f"frame_equalizer_ai_{architecture.name}"
    defaults = {
        "checkpoint": Path("artifacts") / f"{stem}.pt",
        "report": Path("artifacts") / f"{stem}_report.json",
        "header": MODEL_DIRECTORY / architecture.header_basename,
        "artifact_prefix": Path("artifacts") / f"{stem}_weights",
    }
    for name, default in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, str(default))


def _train(args: argparse.Namespace) -> None:
    """Run the full train -> quantize -> evaluate -> export pipeline.

    Seeds Python, NumPy, and torch from ``args.seed`` so the run is
    reproducible, loads the authoritative constellations, and for each
    modulation:

        1. Generates independent train/validation/test datasets (split seed =
           base seed + run seed + model index).
        2. Trains the selected architecture's float MLP on the train split.
        3. Quantizes using the validation split and evaluates the float and
           quantized demappers on the test split.
        4. Records the per-model report entry (training metrics, sample
           counts, BER/SER, float/quant class parity, shifts, dataset digest).

    Afterwards it writes the JSON report, the checkpoint, and the exported
    header/NPZ/JSON artifacts, and prints the wall-clock runtime.

    Args:
        args: Parsed ``train`` subcommand namespace.
    """
    start = time.perf_counter()
    architecture = ARCHITECTURES[args.model]
    _resolve_paths(args, architecture)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    points, source = load_constellations()
    float_models, quant_models, report = {}, {}, {
        "seed": args.seed, "split_seeds": SPLIT_SEEDS, "constellation_source": source,
        "constellation_points": {k: [[float(z.real), float(z.imag)] for z in v]
                                 for k, v in points.items()},
        "architecture": {"name": architecture.name,
                         "hidden_sizes": list(architecture.hidden_sizes)},
        "settings": {key: value for key, value in vars(args).items() if key != "func"},
        "integer_contract": {
            "input_fractional_bits": INPUT_FRACTIONAL_BITS,
            "input_scale": INPUT_SCALE},
        "models": {},
    }
    for index, name in enumerate(MODULATIONS):
        # Distinct seeds keep the three splits disjoint across modulations.
        train = generate_dataset(name, args.train_frames, args.symbols_per_frame,
                                 SPLIT_SEEDS["train"] + args.seed + index)
        validation = generate_dataset(name, args.validation_frames, args.symbols_per_frame,
                                      SPLIT_SEEDS["validation"] + args.seed + index)
        test = generate_dataset(name, args.test_frames, args.symbols_per_frame,
                                SPLIT_SEEDS["test"] + args.seed + index)
        model, training = train_model(train.full_features, train.labels, BITS[name], args.epochs,
                                      args.batch_size, args.learning_rate,
                                      args.seed + 100 * index, architecture.hidden_sizes)
        quant = quantize_model(model, validation.full_features, BITS[name])
        float_models[name], quant_models[name] = model, quant
        float_prediction = predict_classes(model, test.full_features)
        quant_prediction = integer_classes(quant, test.full_features)
        report["models"][name] = {
            "training": training,
            "samples": {"train": len(train.features), "validation": len(validation.features),
                        "test": len(test.features)},
            "test": {"float": _rates(float_prediction, test.classes, BITS[name]),
                     "quantized": _rates(quant_prediction, test.classes, BITS[name])},
            "float_quant_class_parity": float(np.mean(float_prediction == quant_prediction)),
            "shifts": list(quant.shifts),
            "test_digest": dataset_digest(test),
        }
        print(name, json.dumps(report["models"][name]["test"]))
    runtime_seconds = time.perf_counter() - start
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    _save_checkpoint(Path(args.checkpoint), float_models, quant_models, report)
    export_models(quant_models, Path(args.header), Path(args.artifact_prefix))
    print(f"runtime_seconds={runtime_seconds:.3f}")


def _export(args: argparse.Namespace) -> None:
    """Re-export the header/NPZ/JSON artifacts from a checkpoint.

    Args:
        args: Parsed ``export`` subcommand namespace.
    """
    _, models, _, architecture = _load_checkpoint(Path(args.checkpoint))
    _resolve_paths(args, architecture)
    paths = export_models(models, Path(args.header), Path(args.artifact_prefix))
    print("\n".join(str(path) for path in paths))


def _validate(args: argparse.Namespace) -> None:
    """Validate a checkpoint end to end and fail on any mismatch.

    For each modulation the validation re-derives or reloads everything the
    training report claims and checks:

        - ``integer_contract``: exported NPZ and checkpoint metadata agree on
          the Q-format constants.
        - determinism: two same-seed dataset generations hash identically.
        - ``checkpoint_digest``: the regenerated held-out corpus equals the
          one recorded during training.
        - ``checkpoint_metrics``: float and quantized rates reproduce the
          recorded values to machine precision.
        - ``integer_parity``: the reloaded exported arrays (including packing
          and shift handling) produce the same class codes as the quantized
          reference.

    Prints a JSON summary and raises ``SystemExit`` listing every failed
    modulation (plus ``integer_contract``) when any check fails.

    Args:
        args: Parsed ``validate`` subcommand namespace.

    Raises:
        SystemExit: If any validation check fails.
    """
    floats, quants, metadata, architecture = _load_checkpoint(Path(args.checkpoint))
    _resolve_paths(args, architecture)
    paths = export_models(quants, Path(args.header), Path(args.artifact_prefix))
    packed = np.load(paths[1])
    depth = int(packed["hidden_layers"])
    failures, results = [], {}
    contract_valid = (
        int(packed["input_fractional_bits"]) == INPUT_FRACTIONAL_BITS and
        int(packed["input_scale"]) == INPUT_SCALE and
        metadata.get("integer_contract") == {
            "input_fractional_bits": INPUT_FRACTIONAL_BITS,
            "input_scale": INPUT_SCALE})
    if not contract_valid:
        failures.append("integer_contract")
    for index, name in enumerate(MODULATIONS):
        # Rebuild the exact held-out split recorded at training time.
        seed = SPLIT_SEEDS["test"] + int(metadata["seed"]) + index
        first = generate_dataset(name, args.frames, args.symbols_per_frame, seed)
        second = generate_dataset(name, args.frames, args.symbols_per_frame, seed)
        deterministic = dataset_digest(first) == dataset_digest(second)
        checkpoint_test = generate_dataset(
            name, int(metadata["settings"]["test_frames"]),
            int(metadata["settings"]["symbols_per_frame"]), seed)
        checkpoint_digest = (
            dataset_digest(checkpoint_test) == metadata["models"][name]["test_digest"])
        checkpoint_rates = {
            "float": _rates(predict_classes(floats[name], checkpoint_test.full_features),
                             checkpoint_test.classes, BITS[name]),
            "quantized": _rates(integer_classes(quants[name], checkpoint_test.full_features),
                                 checkpoint_test.classes, BITS[name]),
        }
        recorded_rates = metadata["models"][name]["test"]
        checkpoint_metrics = all(
            np.isclose(value, recorded_rates[kind][metric], rtol=0, atol=1e-15)
            for kind, rates in checkpoint_rates.items()
            for metric, value in rates.items())
        float_prediction = predict_classes(floats[name], first.full_features)
        quant_prediction = integer_classes(quants[name], first.full_features)
        # Reload exported arrays so parity covers packing as well as threshold/LSB logic.
        exported = QuantizedModel(
            BITS[name],
            [packed[f"w{layer}"][index] for layer in range(1, depth + 1)],
            packed[f"w{depth + 1}"][index, :BITS[name]],
            [int(shift) for shift in packed["shifts"][index]],
        )
        logits = integer_logits(exported, first.full_features)
        exported_reference = np.sum((logits >= 0).astype(np.int64)
                                    << np.arange(BITS[name]), axis=1)
        parity = bool(np.array_equal(quant_prediction, exported_reference))
        results[name] = {
            "deterministic": deterministic, "checkpoint_digest": checkpoint_digest,
            "checkpoint_metrics": checkpoint_metrics, "integer_parity": parity,
            "float": _rates(float_prediction, first.classes, BITS[name]),
            "quantized": _rates(quant_prediction, first.classes, BITS[name]),
        }
        if not deterministic or not checkpoint_digest or not checkpoint_metrics or not parity:
            failures.append(name)
    print(json.dumps({"integer_contract": contract_valid, "results": results,
                      "generated": [str(p) for p in paths]}, indent=2))
    if failures:
        raise SystemExit("validation failed: " + ", ".join(failures))


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser with the train/export/validate subcommands.

    Each subcommand stores its handler in ``args.func``; running this module
    as a script parses the arguments and dispatches to that handler.  Output
    paths default to ``None`` and are resolved per architecture in
    :func:`_resolve_paths`.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="generate data, train, evaluate, quantize, and export")
    train.add_argument("--model", choices=sorted(ARCHITECTURES),
                       default=DEFAULT_ARCHITECTURE,
                       help="MLP architecture to train and export")
    train.add_argument("--train-frames", type=int, default=1500)
    train.add_argument("--validation-frames", type=int, default=320)
    train.add_argument("--test-frames", type=int, default=600)
    train.add_argument("--symbols-per-frame", type=int, default=5)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--learning-rate", type=float, default=0.002)
    train.add_argument("--seed", type=int, default=39)
    train.add_argument("--checkpoint", default=None)
    train.add_argument("--report", default=None)
    train.add_argument("--header", default=None)
    train.add_argument("--artifact-prefix", default=None)
    train.set_defaults(func=_train)
    export = sub.add_parser("export", help="re-export a saved checkpoint")
    export.add_argument("--checkpoint", required=True,
                        help="checkpoint written by train; selects the architecture")
    export.add_argument("--header", default=None)
    export.add_argument("--artifact-prefix", default=None)
    export.set_defaults(func=_export)
    validate = sub.add_parser("validate", help="check generation, metrics, parity, and export")
    validate.add_argument("--checkpoint", required=True,
                          help="checkpoint written by train; selects the architecture")
    validate.add_argument("--frames", type=int, default=64)
    validate.add_argument("--symbols-per-frame", type=int, default=4)
    validate.add_argument("--header", default=None)
    validate.add_argument("--artifact-prefix", default=None)
    validate.set_defaults(func=_validate)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    arguments.func(arguments)
