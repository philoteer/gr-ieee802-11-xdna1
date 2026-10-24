#!/usr/bin/env python3
"""Dump FFT output, then benchmark standalone frame equalization through EOF.

Usage: python3 base/test/frame_equalizer/run_speedtests.py
Preparation output goes to the console; only benchmark logs are saved.
One complex64 dump and its detached metadata header are kept at a time
and deleted after use, failure, or interruption.
Requires GNU Radio/MLIR-AIE and built kernels in this directory and
base/test/until_fft/kernel/build.
"""

import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import yaml


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[2] / "AI-model-CNN-Channel-est" / "data"


def dataset_paths(flowgraph):
    """Read the dataset choices without evaluating Python expressions."""
    with flowgraph.open() as stream:
        graph = yaml.safe_load(stream)
    paths = []
    for block in graph["blocks"]:
        if block["name"] != "file_path":
            continue
        expression = ast.parse(block["parameters"]["value"], mode="eval").body
        if not (
            isinstance(expression, ast.BinOp)
            and isinstance(expression.op, ast.Add)
            and isinstance(expression.left, ast.Name)
            and expression.left.id == "data_dir"
        ):
            raise ValueError("Expected file_path choices of the form data_dir + '/...'")
        suffix = ast.literal_eval(expression.right)
        if not isinstance(suffix, str) or not suffix.startswith("/"):
            raise ValueError("Expected a dataset path relative to data_dir")
        path = DATA_DIR / suffix.lstrip("/")
        if path not in paths:
            paths.append(path)
    if not paths:
        raise ValueError("No file_path dataset choices found in dump.grc")
    return paths


def nominal_frequency(dataset):
    """Use the capture frequency when configuring the equalizer's frequency RTP."""
    with Path(str(dataset) + "-meta").open() as stream:
        metadata = json.load(stream)
    for capture in metadata.get("captures", []):
        if "core:frequency" in capture:
            return float(capture["core:frequency"])
    raise ValueError(f"No capture frequency found in {dataset}-meta")


def stop_process(process):
    """Stop and reap a child before deleting its temporary dump, without a timeout."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    process.wait()


def run_stage(command, log_path):
    """Wait for EOF without a timeout, logging to a file or the console."""
    with log_path.open("w") if log_path is not None else nullcontext(sys.stdout) as log:
        print("Command: " + " ".join(map(str, command)), file=log, flush=True)
        process = subprocess.Popen(
            command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            returncode = process.wait()
        finally:
            stop_process(process)
        print(f"Exit code: {returncode}", file=log)
    if returncode:
        destination = log_path if log_path is not None else "console output"
        raise RuntimeError(f"Exit code {returncode}; see {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "speedtest_logs",
        help="Directory for benchmark .log files; preparation output goes to the console.",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=HERE / "tmp_data",
        help="Disk-backed directory for temporary dumps; avoid RAM-backed /tmp.",
    )
    args = parser.parse_args()
    datasets = dataset_paths(HERE / "dump.grc")
    frequencies = {}
    for dataset in datasets:
        for suffix in ("-data", "-meta"):
            if not Path(str(dataset) + suffix).is_file():
                parser.error(f"Missing capture file: {dataset}{suffix}")
        frequencies[dataset] = nominal_frequency(dataset)
    for directory in (HERE / "build", HERE.parent / "until_fft/kernel/build"):
        for filename in ("final.xclbin", "insts.bin"):
            if not (directory / filename).is_file():
                parser.error(f"Missing built kernel file: {directory / filename}")

    for graph in ("dump.grc", "speedtest.grc"):
        subprocess.run(["grcc", "-o", str(HERE), str(HERE / graph)], cwd=HERE, check=True)
    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for dataset in datasets:
        try:
            with tempfile.TemporaryDirectory(prefix="frame-equalizer-", dir=work_dir) as temporary:
                dump = Path(temporary) / "fft.iqdata"
                print(f"Preparing {dataset.name} -> temporary complex64 FFT dump", flush=True)
                run_stage(
                    [sys.executable, "-u", str(HERE / "fft_dump.py"),
                     "--dataset", str(dataset), "--dump-path", str(dump)],
                    None,
                )
                if not dump.is_file() or dump.stat().st_size == 0 or not Path(str(dump) + ".hdr").is_file():
                    raise RuntimeError(f"Empty or incomplete FFT dump for {dataset.name}")
                log = output_dir / (dataset.name + ".log")
                print(
                    f"Benchmarking {dataset.name} through EOF "
                    f"({dump.stat().st_size / 1024**3:.2f} GiB dump) -> {log}", flush=True,
                )
                run_stage(
                    [sys.executable, "-u", str(HERE / "frame_equalizer_test.py"),
                     "--dump-path", str(dump), "--freq", str(frequencies[dataset])],
                    log,
                )
            print(f"Completed {dataset.name}; temporary dump deleted", flush=True)
        except RuntimeError as error:
            failed = True
            print(f"{dataset.name}: {error}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
