#!/usr/bin/env python3
"""Generate the speedtest and log each dataset's processing rate through EOF.

Usage: python3 base/test/threshold_cf/run_speedtests.py
Requires the GNU Radio/MLIR-AIE runtime environment and the local built kernel.
"""

import argparse
import ast
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[2] / "AI-model-CNN-Channel-est" / "data"


def dataset_paths(flowgraph):
    """Read all file_path choices from GRC without evaluating Python code."""
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
        raise ValueError("No file_path dataset choices found in the flowgraph")
    return paths


def stop_process(process):
    """Allow GNU Radio to stop and flush its output, then enforce cleanup."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError("Flowgraph did not stop within 10 seconds of SIGINT")


def run_dataset(dataset, log_path, environment):
    """Run one capture in a fresh process and capture native stdout and stderr."""
    command = [sys.executable, "-u", str(HERE / "wifi_rx.py"), "--dataset", str(dataset)]
    with log_path.open("w") as log:
        print(f"Dataset: {dataset}\nRun until EOF", file=log, flush=True)
        process = subprocess.Popen(
            command, cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            process.wait()
        finally:
            stop_process(process)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "speedtest_logs",
        help="Directory for per-dataset .log files (default: alongside this script).",
    )
    args = parser.parse_args()

    flowgraph = HERE / "speedtest.grc"
    datasets = dataset_paths(flowgraph)
    for dataset in datasets:
        for suffix in ("-data", "-meta"):
            path = Path(str(dataset) + suffix)
            if not path.is_file():
                parser.error(f"Missing capture file: {path}")

    subprocess.run(["grcc", "-o", str(HERE), str(flowgraph)], cwd=HERE, check=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    library_dir = HERE.parents[3] / "gr-foo-fixed" / "build" / "lib"
    previous = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(library_dir) + (":" + previous if previous else "")

    failed = False
    for dataset in datasets:
        log_path = output_dir / (dataset.name + ".log")
        print(f"Running {dataset.name} through EOF -> {log_path}", flush=True)
        returncode = run_dataset(dataset, log_path, environment)
        if returncode:
            failed = True
            print(f"  Failed with exit code {returncode}; see {log_path}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
