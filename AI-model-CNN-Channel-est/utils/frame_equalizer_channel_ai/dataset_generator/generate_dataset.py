#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.frame_equalizer_channel_ai.common.dataset import corpus_digest
from utils.frame_equalizer_channel_ai.dataset_generator.corpus import (
    convert_gnuradio_archive,
    generate_gnuradio_raw_archive,
    split_corpus,
)


DEFAULT_DATASET_DIRECTORY = PROJECT_ROOT / "data" / "datasets" / "channel_estimator"


def dataset_path(directory: Path, dataset: str, split: str) -> Path:
    return directory / dataset / f"{split}.npz"


def generate_dataset(args) -> None:
    root_directory = Path(args.dataset_dir)
    split_sizes = {
        "train": args.train_samples,
        "validation": args.validation_samples,
        "test": args.test_samples,
    }
    sources = ("original", "gnuradio") if args.dataset == "both" else (args.dataset,)
    if args.dataset == "both" and any(samples < 2 for samples in split_sizes.values()):
        raise ValueError("combined datasets require at least two samples per split")
    generated = {}
    for source in sources:
        dataset_name = f"synthetic_{source}"
        directory = root_directory / dataset_name
        directory.mkdir(parents=True, exist_ok=True)
        source_splits = {}
        manifest = {
            "dataset": dataset_name,
            "source": source,
            "request_dataset": args.dataset,
            "seed": args.seed,
            "snr_min_db": args.snr_min,
            "snr_max_db": args.snr_max,
            "splits": source_splits,
        }
        for split, total_samples in split_sizes.items():
            if args.dataset == "both":
                samples = ((total_samples + 1) // 2 if source == "original"
                           else total_samples // 2)
            else:
                samples = total_samples
            corpus = split_corpus(split, samples, args.seed, source=source,
                                  snr_min_db=args.snr_min, snr_max_db=args.snr_max)
            corpus.save(dataset_path(root_directory, dataset_name, split))
            source_splits[split] = {
                "samples": samples,
                "digest": corpus_digest(corpus),
            }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        generated[dataset_name] = {"directory": str(directory), "splits": source_splits}
    print(json.dumps({"directory": str(root_directory), "dataset": args.dataset,
                      "sources": generated}, indent=2))


def run(args) -> None:
    if args.convert is not None:
        if args.raw_samples is not None:
            raise ValueError("--raw-samples cannot be used with --convert")
        output = args.output or args.convert.with_name(
            f"{args.convert.stem}-converted.npz")
        if output.suffix.lower() != ".npz":
            output = Path(f"{output}.npz")
        corpus = convert_gnuradio_archive(args.convert)
        corpus.save(output)
        print(json.dumps({
            "converted_from": str(args.convert),
            "output": str(output),
            "samples": len(corpus.inputs_q15),
            "digest": corpus_digest(corpus),
        }, indent=2))
        return
    if args.raw_output is not None:
        if args.output is not None:
            raise ValueError("--output cannot be used with --raw-output")
        samples = args.raw_samples if args.raw_samples is not None else 12000
        path = generate_gnuradio_raw_archive(
            samples, args.seed, args.snr_min, args.snr_max, args.raw_output)
        print(json.dumps({"raw_archive": str(path), "samples": samples}, indent=2))
        return
    if args.output is not None or args.raw_samples is not None:
        raise ValueError("--output and --raw-samples require --convert or --raw-output")
    generate_dataset(args)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    operations = result.add_mutually_exclusive_group()
    operations.add_argument("--convert", type=Path, metavar="RAW_NPZ")
    operations.add_argument("--raw-output", type=Path, metavar="RAW_NPZ")
    result.add_argument("--output", type=Path)
    result.add_argument("--raw-samples", type=int)
    result.add_argument("--train-samples", type=int, default=12000)
    result.add_argument("--validation-samples", type=int, default=2000)
    result.add_argument("--test-samples", type=int, default=3000)
    result.add_argument("--snr-min", type=float, default=2.0)
    result.add_argument("--snr-max", type=float, default=40.0)
    result.add_argument("--seed", type=int, default=40)
    result.add_argument("--dataset", choices=("original", "gnuradio", "both"),
                        default="original")
    result.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIRECTORY))
    result.set_defaults(func=run)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
