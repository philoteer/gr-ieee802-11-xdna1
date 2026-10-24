#!/usr/bin/env python3
"""Command-line training, export, validation, and evaluation entry point.

The command-line workflow keeps corpus generation, floating-point training,
integer quantization, artifact export, and reproducibility checks under one
interface.  The default paths match the deployed stage39 kernel layout while
still allowing every output to be overridden for experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

try:
    from .dataset import OCCUPIED, corpus_digest, split_corpus
    from .exporter import export_model
    from .model import ChannelEstimator, float_estimate, train_model
    from .quantize import integer_estimate, load_npz, quantize_model
except ImportError:
    from dataset import OCCUPIED, corpus_digest, split_corpus
    from exporter import export_model
    from model import ChannelEstimator, float_estimate, train_model
    from quantize import integer_estimate, load_npz, quantize_model


# All default utility artifacts are local to this package, except the deployed
# C++ header, which is written into the stage39 kernel tree.
DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = DIRECTORY / "artifacts" / "channel_estimator.pt"
DEFAULT_PREFIX = DIRECTORY / "artifacts" / "channel_estimator_weights"
DEFAULT_REPORT = DIRECTORY / "artifacts" / "channel_estimator_report.json"
DEFAULT_HEADER = DIRECTORY.parents[1] / "kernel" / "channel_estimator" / "model" / "frame_equalizer_channel_ai_weights.h"


def _metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Calculate occupied-bin channel NMSE and equalization EVM.

    Args:
        estimate: Estimated channel tensor with shape ``(N, 64, 2)``.
        truth: Ground-truth channel tensor with the same shape.

    Returns:
        Dictionary containing normalized mean-squared channel error and the
        relative equalization EVM.
    """
    estimated = estimate[:, OCCUPIED, 0] + 1j * estimate[:, OCCUPIED, 1]
    actual = truth[:, OCCUPIED, 0] + 1j * truth[:, OCCUPIED, 1]
    error = estimated - actual
    nmse = np.mean(np.abs(error) ** 2) / np.mean(np.abs(actual) ** 2)
    safe = np.where(np.abs(estimated) > 1e-9, estimated, 1e-9 + 0j)
    evm = np.sqrt(np.mean(np.abs(actual / safe - 1.0) ** 2))
    return {"nmse": float(nmse), "equalization_evm": float(evm)}


def _models_from_checkpoint(path: Path):
    """Restore a floating-point estimator and its training metadata.

    Args:
        path: PyTorch checkpoint produced by :func:`train`.

    Returns:
        Tuple containing the evaluated model and the metadata dictionary.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ChannelEstimator()
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload["metadata"]


def _evaluate(model: ChannelEstimator, corpus) -> dict:
    """Evaluate LS, floating-point CNN, and quantized CNN estimates.

    Args:
        model: Floating-point channel estimator to evaluate and quantize.
        corpus: Corpus supplying observations and channel truth.

    Returns:
        Nested metric dictionary keyed by ``ls``, ``float``, and ``quantized``.
    """
    quantized = quantize_model(model)
    return {
        "ls": _metrics(corpus.ls, corpus.truth),
        "float": _metrics(float_estimate(model, corpus), corpus.truth),
        "quantized": _metrics(integer_estimate(quantized, corpus), corpus.truth),
    }


def train(args) -> None:
    """Train, evaluate, checkpoint, and export a channel estimator.

    Args:
        args: Parsed command-line namespace containing corpus, optimizer, seed,
            and output-path settings.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = split_corpus("train", args.train_samples, args.seed, source=args.dataset,
                              snr_min_db=args.snr_min, snr_max_db=args.snr_max)
    validation = split_corpus("validation", args.validation_samples, args.seed,
                              source=args.dataset,
                              snr_min_db=args.snr_min, snr_max_db=args.snr_max)
    test = split_corpus("test", args.test_samples, args.seed, source=args.dataset,
                        snr_min_db=args.snr_min, snr_max_db=args.snr_max)
    model, losses = train_model(train_data, args.epochs, args.batch_size,
                                args.learning_rate, args.seed)
    metadata = {"seed": args.seed, "train_samples": args.train_samples,
                "validation_samples": args.validation_samples,
                "test_samples": args.test_samples, "snr_min": args.snr_min,
                "snr_max": args.snr_max, "epochs": args.epochs,
                "dataset": args.dataset,
                "test_digest": corpus_digest(test)}
    report = {"settings": metadata, "loss": losses,
              "validation": _evaluate(model, validation),
              "test": _evaluate(model, test)}
    checkpoint, report_path = Path(args.checkpoint), Path(args.report)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    export_model(quantize_model(model), Path(args.header), Path(args.artifact_prefix))
    print(json.dumps(report["test"], indent=2))


def export(args) -> None:
    """Re-export quantized artifacts from an existing checkpoint.

    Args:
        args: Parsed namespace containing checkpoint and destination paths.
    """
    model, _ = _models_from_checkpoint(Path(args.checkpoint))
    print("\n".join(str(path) for path in export_model(
        quantize_model(model), Path(args.header), Path(args.artifact_prefix))))


def validate(args) -> None:
    """Verify deterministic data generation and export-reload parity.

    Args:
        args: Parsed namespace containing checkpoint and destination paths.

    Raises:
        SystemExit: If the regenerated corpus or exported integer weights do
            not reproduce the checkpoint's recorded results.
    """
    model, metadata = _models_from_checkpoint(Path(args.checkpoint))
    corpus = split_corpus("test", int(metadata["test_samples"]), int(metadata["seed"]),
                          source=str(metadata.get("dataset", "original")),
                          snr_min_db=float(metadata["snr_min"]),
                          snr_max_db=float(metadata["snr_max"]))
    regenerated = corpus_digest(corpus) == metadata["test_digest"]
    paths = export_model(quantize_model(model), Path(args.header), Path(args.artifact_prefix))
    reloaded = load_npz(paths[1])
    parity = np.array_equal(integer_estimate(quantize_model(model), corpus),
                            integer_estimate(reloaded, corpus))
    result = {"dataset_deterministic": regenerated, "export_parity": parity,
              "metrics": _evaluate(model, corpus)}
    print(json.dumps(result, indent=2))
    if not regenerated or not parity:
        raise SystemExit("validation failed")


def evaluate(args) -> None:
    """Evaluate a checkpoint on a selected deterministic corpus split.

    Args:
        args: Parsed namespace containing checkpoint, split, sample count, and
            optional corpus/report output paths.
    """
    model, metadata = _models_from_checkpoint(Path(args.checkpoint))
    source = args.dataset or str(metadata.get("dataset", "original"))
    corpus = split_corpus(args.split, args.samples, int(metadata["seed"]), source=source,
                          snr_min_db=float(metadata["snr_min"]),
                          snr_max_db=float(metadata["snr_max"]))
    result = {"dataset": source, "split": args.split, "samples": args.samples,
              "metrics": _evaluate(model, corpus)}
    if args.corpus:
        corpus.save(Path(args.corpus))
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser for all estimator workflows.

    Returns:
        An argument parser with ``train``, ``export``, ``validate``, and
        ``evaluate`` subcommands.
    """
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--train-samples", type=int, default=12000)
    training.add_argument("--validation-samples", type=int, default=2000)
    training.add_argument("--test-samples", type=int, default=3000)
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--learning-rate", type=float, default=0.002)
    training.add_argument("--snr-min", type=float, default=8.0)
    training.add_argument("--snr-max", type=float, default=32.0)
    training.add_argument("--seed", type=int, default=40)
    training.add_argument("--dataset", choices=("original", "gnuradio", "both"),
                          default="original")
    training.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    training.add_argument("--report", default=str(DEFAULT_REPORT))
    training.add_argument("--header", default=str(DEFAULT_HEADER))
    training.add_argument("--artifact-prefix", default=str(DEFAULT_PREFIX))
    training.set_defaults(func=train)
    exporting = commands.add_parser("export")
    exporting.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    exporting.add_argument("--header", default=str(DEFAULT_HEADER))
    exporting.add_argument("--artifact-prefix", default=str(DEFAULT_PREFIX))
    exporting.set_defaults(func=export)
    validating = commands.add_parser("validate")
    validating.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    validating.add_argument("--header", default=str(DEFAULT_HEADER))
    validating.add_argument("--artifact-prefix", default=str(DEFAULT_PREFIX))
    validating.set_defaults(func=validate)
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    evaluating.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluating.add_argument("--samples", type=int, default=3000)
    evaluating.add_argument("--dataset", choices=("original", "gnuradio", "both"))
    evaluating.add_argument("--corpus")
    evaluating.add_argument("--report")
    evaluating.set_defaults(func=evaluate)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
