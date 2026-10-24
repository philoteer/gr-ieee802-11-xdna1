#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.frame_equalizer_channel_ai.common.corpora import DEFAULT_DATASET_DIRECTORY, load_corpus
from utils.frame_equalizer_channel_ai.common.dataset import corpus_digest
from utils.frame_equalizer_channel_ai.common.model import ChannelEstimator, float_estimate
from utils.frame_equalizer_channel_ai.common.updater import load_cnn_header
from utils.frame_equalizer_channel_ai.training.exporter import DEFAULT_CPU_DIRECTORY, export_model, export_trained_cnn
from utils.frame_equalizer_channel_ai.training.model import train_model
from utils.frame_equalizer_channel_ai.training.quantize import quantize_model
from utils.frame_equalizer_channel_ai.verification.cli import metrics, select_gate, select_estimate_gate
from utils.frame_equalizer_channel_ai.verification.quantized import integer_estimate


DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = DIRECTORY / "artifacts" / "channel_estimator.pt"
DEFAULT_PREFIX = DIRECTORY / "artifacts" / "channel_estimator_weights"
DEFAULT_REPORT = DIRECTORY / "artifacts" / "channel_estimator_report.json"
DEFAULT_HEADER = PROJECT_ROOT / "kernel" / "channel_estimator" / "model" / "frame_equalizer_channel_ai_weights_by_frame.h"


def train(args) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    directory = Path(args.dataset_dir)
    training = load_corpus(directory, args.dataset, "train")
    validation = load_corpus(directory, args.dataset, "validation")
    model, losses, validation_losses, best_epoch = train_model(
        training, validation, args.epochs, args.batch_size, args.learning_rate,
        args.seed, args.patience, args.min_delta, args.noise_std)
    quantized = quantize_model(model)
    gate, _ = select_gate(quantized, validation)
    cpu_gate, _ = select_estimate_gate(float_estimate(model, validation), validation)
    quantized = quantize_model(model, gate)
    # Test observations/labels are not loaded until weights and both gates are fixed.
    test = load_corpus(directory, args.dataset, "test")
    metadata = {"seed": args.seed, "train_samples": len(training.inputs_q15),
                "validation_samples": len(validation.inputs_q15),
                "test_samples": len(test.inputs_q15), "epochs": args.epochs,
                "epochs_trained": len(losses), "best_epoch": best_epoch,
                "patience": args.patience, "min_delta": args.min_delta,
                "noise_std": args.noise_std,
                "dataset": args.dataset, "dataset_dir": str(directory.resolve()),
                "threads": args.threads, "test_digest": corpus_digest(test),
                "gate_snr_ratio_x16": gate, "cpu_gate_snr_ratio_x16": cpu_gate}
    if args.domain_manifest is not None:
        manifest = json.loads(args.domain_manifest.read_text())
        metadata['domain_policy'] = manifest['domain_policy']
        metadata['split_digests'] = {name: split['digest'] for name, split in manifest['splits'].items()}
    results = {}
    for name, corpus in (("validation", validation), ("test", test)):
        floating = float_estimate(model, corpus)
        floating[corpus.snr_ratio_x16 >= cpu_gate] = corpus.ls[corpus.snr_ratio_x16 >= cpu_gate]
        results[name] = {"ls": metrics(corpus.ls, corpus.truth),
                         "quantized": metrics(integer_estimate(quantized, corpus), corpus.truth),
                         "float": metrics(floating, corpus.truth)}
    if args.domain_manifest is None and Path(args.header).is_file():
        previous = load_cnn_header(Path(args.header))
        results["previous_quantized_test"] = metrics(integer_estimate(previous, test), test.truth)
    checkpoint, report = Path(args.checkpoint), Path(args.report)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
    payload = {"settings": metadata, "loss": losses,
               "validation_loss": validation_losses, "metrics": results}
    if args.domain_manifest is not None:
        payload['domain_policy'] = metadata['domain_policy']
    report.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n".join(str(path) for path in export_model(
        quantized, Path(args.header), Path(args.artifact_prefix))))
    print(export_trained_cnn(model, cpu_gate, Path(args.cpu_header)))
    print(json.dumps({"settings": metadata, "metrics": results}, indent=2))


def export(args) -> None:
    torch.set_num_threads(args.threads)
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    model = ChannelEstimator()
    model.load_state_dict(payload["state_dict"])
    metadata = payload["metadata"]
    gate = args.gate_snr_ratio_x16
    if gate is None:
        gate = metadata["gate_snr_ratio_x16"]
    cpu_gate = args.cpu_gate_snr_ratio_x16
    if cpu_gate is None:
        cpu_gate = metadata.get("cpu_gate_snr_ratio_x16", gate)
    print("\n".join(str(path) for path in export_model(
        quantize_model(model, gate), Path(args.header), Path(args.artifact_prefix))))
    print(export_trained_cnn(model, cpu_gate, Path(args.cpu_header)))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--learning-rate", type=float, default=0.002)
    training.add_argument("--patience", type=int, default=8)
    training.add_argument("--min-delta", type=float, default=0.0)
    training.add_argument("--noise-std", type=float, default=0.02)
    training.add_argument("--seed", type=int, default=40)
    training.add_argument("--threads", type=int, default=2)
    training.add_argument("--dataset", default="original")
    training.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIRECTORY))
    training.add_argument("--domain-manifest", type=Path,
                          help="cross-domain manifest recording source-only selection and final test provenance")
    training.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    training.add_argument("--report", default=str(DEFAULT_REPORT))
    training.add_argument("--header", default=str(DEFAULT_HEADER))
    training.add_argument("--cpu-header", default=str(DEFAULT_CPU_DIRECTORY / "channel_cnn_weights_by_frame.h"))
    training.add_argument("--artifact-prefix", default=str(DEFAULT_PREFIX))
    training.set_defaults(func=train)
    exporting = commands.add_parser("export")
    exporting.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    exporting.add_argument("--header", default=str(DEFAULT_HEADER))
    exporting.add_argument("--cpu-header", default=str(DEFAULT_CPU_DIRECTORY / "channel_cnn_weights_by_frame.h"))
    exporting.add_argument("--artifact-prefix", default=str(DEFAULT_PREFIX))
    exporting.add_argument("--gate-snr-ratio-x16", type=int)
    exporting.add_argument("--cpu-gate-snr-ratio-x16", type=int)
    exporting.add_argument("--threads", type=int, default=2)
    exporting.set_defaults(func=export)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
