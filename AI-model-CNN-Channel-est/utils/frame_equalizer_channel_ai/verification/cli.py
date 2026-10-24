#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.frame_equalizer_channel_ai.common.corpora import DEFAULT_DATASET_DIRECTORY, load_corpus
from utils.frame_equalizer_channel_ai.common.dataset import ChannelCorpus, OCCUPIED, corpus_digest, snr_gate_candidates
from utils.frame_equalizer_channel_ai.common.quantized import QuantizedEstimator
from utils.frame_equalizer_channel_ai.verification.quantized import integer_estimate, load_npz


DEFAULT_WEIGHTS = PROJECT_ROOT / "utils" / "frame_equalizer_channel_ai" / "training" / "artifacts" / "channel_estimator_weights.npz"


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimated = estimate[:, OCCUPIED, 0] + 1j * estimate[:, OCCUPIED, 1]
    actual = truth[:, OCCUPIED, 0] + 1j * truth[:, OCCUPIED, 1]
    error = estimated - actual
    nmse = np.mean(np.abs(error) ** 2) / np.mean(np.abs(actual) ** 2)
    safe = np.where(np.abs(estimated) > 1e-9, estimated, 1e-9 + 0j)
    evm = np.sqrt(np.mean(np.abs(actual / safe - 1.0) ** 2))
    return {"nmse": float(nmse), "equalization_evm": float(evm)}


def select_gate(model: QuantizedEstimator, corpus: ChannelCorpus) -> tuple[int, dict[str, float]]:
    ungated = QuantizedEstimator(model.weights)
    cnn = integer_estimate(ungated, corpus)
    return select_estimate_gate(cnn, corpus)


def select_estimate_gate(cnn: np.ndarray, corpus: ChannelCorpus) -> tuple[int, dict[str, float]]:
    actual = corpus.truth[:, OCCUPIED].astype(np.float64)
    cnn_mse = np.mean((cnn[:, OCCUPIED].astype(np.float64) - actual) ** 2, axis=(1, 2))
    ls_mse = np.mean((corpus.ls[:, OCCUPIED].astype(np.float64) - actual) ** 2, axis=(1, 2))
    candidates = snr_gate_candidates()
    best_threshold = candidates[-1]
    best_metrics = metrics(cnn, corpus.truth)
    best_score = float(np.mean(cnn_mse))
    for threshold in candidates:
        use_ls = corpus.snr_ratio_x16 >= threshold
        score = float(np.mean(np.where(use_ls, ls_mse, cnn_mse)))
        if score < best_score:
            estimate = cnn.copy()
            estimate[use_ls] = corpus.ls[use_ls]
            best_threshold = threshold
            best_metrics = metrics(estimate, corpus.truth)
            best_score = score
    best_metrics["mean_frame_mse"] = best_score
    return int(best_threshold), best_metrics


def by_snr(model: QuantizedEstimator, corpus: ChannelCorpus) -> dict:
    estimate = integer_estimate(model, corpus)
    snr_db = 10.0 * np.log10(np.maximum(corpus.snr_ratio_x16.astype(np.float64), 1.0) / 32.0)
    result = {}
    for label, lower, upper in (("below_6_db", -np.inf, 6),
                                ("6-12_db", 6, 12), ("12-18_db", 12, 18),
                                ("18-24_db", 18, 24), ("24-30_db", 24, 30),
                                ("30-36_db", 30, 36), ("36_db_and_above", 36, np.inf)):
        selected = (snr_db >= lower) & (snr_db < upper)
        if np.any(selected):
            result[label] = {"samples": int(np.sum(selected)),
                             "ls": metrics(corpus.ls[selected], corpus.truth[selected]),
                             "gated": metrics(estimate[selected], corpus.truth[selected])}
    return result


def run(args) -> None:
    model = load_npz(Path(args.weights))
    corpus = load_corpus(Path(args.dataset_dir), args.dataset, args.split)
    if args.command == "evaluate" and args.checkpoint and (args.split != "test" or args.samples is not None):
        raise ValueError("checkpoint digest verification requires the full test split")
    if args.samples is not None:
        if args.samples <= 0 or args.samples > len(corpus.inputs_q15):
            raise ValueError("requested sample count is outside the saved split")
        corpus = ChannelCorpus(*(getattr(corpus, name)[:args.samples] for name in (
            "inputs_q15", "observations", "snr_ratio_x16", "ls", "truth",
            "target_q15", "normalization_shift")))
    if args.command == "gate":
        threshold, validation = select_gate(model, corpus)
        result = {"gate_snr_ratio_x16": threshold, "validation": validation}
    else:
        result = {"dataset": args.dataset, "split": args.split,
                  "samples": len(corpus.inputs_q15),
                  "metrics": {"ls": metrics(corpus.ls, corpus.truth),
                              "quantized": metrics(integer_estimate(
                                  QuantizedEstimator(model.weights), corpus), corpus.truth),
                              "gated": metrics(integer_estimate(model, corpus), corpus.truth)},
                  "by_snr": by_snr(model, corpus)}
        if args.checkpoint:
            import torch
            payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
            result["dataset_matches_checkpoint"] = (
                corpus_digest(corpus) == payload["metadata"]["test_digest"])
            if not result["dataset_matches_checkpoint"]:
                raise SystemExit("validation failed: dataset does not match checkpoint")
    if args.corpus:
        corpus.save(Path(args.corpus))
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("gate", "evaluate"):
        option = commands.add_parser(command)
        option.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
        option.add_argument("--dataset", default="original")
        option.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIRECTORY))
        option.add_argument("--split", choices=("train", "validation", "test"),
                            default="validation" if command == "gate" else "test")
        option.add_argument("--samples", type=int)
        option.add_argument("--corpus")
        option.add_argument("--report")
        if command == "evaluate":
            option.add_argument("--checkpoint")
        option.set_defaults(func=run)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
