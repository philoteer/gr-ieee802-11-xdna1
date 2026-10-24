#!/usr/bin/env python3
"""Validate or benchmark preprocess, channel estimation, and equalize through EOF.

Usage: python3 run_speedtests.py --validate --estimator cnn --estimator mlp
Preparation output goes to the console; only benchmark logs are saved.
FFT/prepared/estimated dumps and validation PCAPs are temporary and removed after use.
Requires the globally installed GNU Radio/MLIR-AIE module and built kernels.
"""

import argparse
import ast
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import signal
import struct
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
        raise ValueError("No file_path dataset choices found in until_fft.grc")
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


def flowgraph(name, *arguments):
    return [sys.executable, "-u", str(HERE / (name + ".py")), *map(str, arguments)]


def pcap_frames(path):
    """Check PCAP/radiotap framing and count MAC payloads, ignoring timestamps."""
    frames = Counter()
    with path.open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24:
            raise RuntimeError(f"Missing PCAP header: {path}")
        magic, major, minor, _, _, snaplen, linktype = struct.unpack("<IHHiIII", header)
        if magic != 0xa1b2c3d4 or (major, minor) != (2, 4) or linktype != 127:
            raise RuntimeError(f"Invalid radiotap PCAP header: {path}")
        while record := stream.read(16):
            if len(record) != 16:
                raise RuntimeError(f"Truncated PCAP record: {path}")
            _, usec, length, original = struct.unpack("<IIII", record)
            if usec >= 1000000 or length != original or not 8 <= length <= snaplen:
                raise RuntimeError(f"Invalid PCAP record: {path}")
            packet = stream.read(length)
            if len(packet) != length or packet[0] != 0:
                raise RuntimeError(f"Truncated or invalid radiotap packet: {path}")
            radiotap_length = int.from_bytes(packet[2:4], "little")
            if not 8 <= radiotap_length < length:
                raise RuntimeError(f"Invalid radiotap length: {path}")
            digest = hashlib.sha256(packet[radiotap_length:]).digest()
            frames[digest] += 1
    if not frames:
        raise RuntimeError(f"No CRC-accepted received packets: {path}")
    return frames


def validate(fft, prepared, frequency, temporary, estimators, log_dir):
    """Run every graph and compare live/file-separated decoding for each estimator."""
    log_dir.mkdir(parents=True, exist_ok=True)

    def stage(name, *arguments, label=None):
        run_stage(flowgraph(name, *arguments), log_dir / ((label or name) + '.log'))

    reference = temporary / "native.pcap"
    stage("until_fft_test", "--dump-path", fft, "--freq", frequency, "--pcap-path", reference)
    native_frames = pcap_frames(reference)
    print(f"Native reference: {sum(native_frames.values())} CRC-accepted packets", flush=True)
    stage("preprocess_speedtest", "--dump-path", fft, "--freq", frequency)
    stage("preprocess_dump", "--dump-path", fft, "--freq", frequency, "--output-path", prepared)
    results = {"native_packets": sum(native_frames.values()), "estimators": {}}
    for estimator in estimators:
        print(f"Validating {estimator}: live chain, prepared replay, estimated replay, and rate graphs", flush=True)
        live = temporary / f"{estimator}-live.pcap"
        post = temporary / f"{estimator}-post.pcap"
        replay = temporary / f"{estimator}-replay.pcap"
        estimated = temporary / "estimated.iqdata"
        stage("channel_est_speedtest", "--dump-path", prepared, "--estimator", estimator,
              label=f"{estimator}-channel_est_speedtest")
        stage("channel_est_dump", "--dump-path", prepared, "--estimator", estimator,
              "--output-path", estimated, label=f"{estimator}-channel_est_dump")
        stage("equalize_speedtest", "--dump-path", estimated, "--freq", frequency,
              label=f"{estimator}-equalize_speedtest")
        stage("channel_est_test", "--dump-path", fft, "--freq", frequency,
              "--estimator", estimator, "--pcap-path", live, label=f"{estimator}-channel_est_test")
        stage("until_preproc_test", "--dump-path", prepared, "--freq", frequency,
              "--estimator", estimator, "--pcap-path", post, label=f"{estimator}-until_preproc_test")
        stage("equalize_test", "--dump-path", estimated, "--freq", frequency,
              "--pcap-path", replay, label=f"{estimator}-equalize_test")
        counters = {"live": pcap_frames(live), "post_preproc": pcap_frames(post),
                    "estimated_replay": pcap_frames(replay)}
        result = {name: sum(frames.values()) for name, frames in counters.items()}
        print(f"{estimator} decoded counts: {result}", flush=True)
        result["unmatched_occurrences"] = {}
        for label, first, second in (
            ("live vs post-preprocess", counters["live"], counters["post_preproc"]),
            ("live vs estimated replay", counters["live"], counters["estimated_replay"]),
            ("post-preprocess vs estimated replay", counters["post_preproc"], counters["estimated_replay"]),
        ):
            difference = sum((first - second).values()) + sum((second - first).values())
            print(f"{estimator} {label}: {sum((first & second).values())} matching MAC frames; "
                  f"{difference} unmatched occurrences", flush=True)
            result["unmatched_occurrences"][label] = difference
            # Kernel-level small nondeterminism is reported, not debugged or
            # corrected here. Reject a broken handoff with no common MAC frames.
            if not (first & second):
                raise RuntimeError(f"{estimator} {label}: no matching decoded MAC frames")
        # Learned estimators intentionally differ from the LS reference. Report
        # recovery/overlap rather than requiring them to reproduce LS decisions.
        result["matching_native"] = sum((counters["live"] & native_frames).values())
        print(f"{estimator}: {result}; native LS comparison is informational", flush=True)
        results["estimators"][estimator] = result
        remove_dump(estimated)
    (log_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


def remove_dump(path):
    path.unlink()
    Path(str(path) + '.hdr').unlink()


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
    parser.add_argument("--validate", action="store_true",
                        help="Validate stress_test only by default; do not run speed measurements.")
    parser.add_argument("--dataset", action="append",
                        help="Dataset basename from until_fft.grc; may be repeated.")
    parser.add_argument("--estimator", choices=("cnn", "mlp"), action="append",
                        help="Estimator variant; default is both. mlp means CNN + MLP payload updater.")
    args = parser.parse_args()
    estimators = list(dict.fromkeys(args.estimator or ["cnn", "mlp"]))
    datasets = dataset_paths(HERE / "until_fft.grc")
    selected = args.dataset or (["stress_test.sigmf"] if args.validate else None)
    if selected:
        names = {dataset.name: dataset for dataset in datasets}
        unknown = set(selected) - names.keys()
        if unknown:
            parser.error("Unknown datasets: " + ", ".join(sorted(unknown)))
        datasets = [names[name] for name in dict.fromkeys(selected)]
    frequencies = {}
    for dataset in datasets:
        for suffix in ("-data", "-meta"):
            if not Path(str(dataset) + suffix).is_file():
                parser.error(f"Missing capture file: {dataset}{suffix}")
        frequencies[dataset] = nominal_frequency(dataset)
    for directory in (HERE / "preprocess/build", HERE / "equalize/build",
                      HERE.parent / "until_fft/kernel/build",
                      *[HERE / "channel_est/build" / estimator for estimator in estimators]):
        for filename in ("final.xclbin", "insts.bin"):
            if not (directory / filename).is_file():
                parser.error(f"Missing built kernel file: {directory / filename}")

    graphs = ("until_fft", "preprocess_speedtest", "preprocess_dump",
              "channel_est_speedtest", "channel_est_dump", "channel_est_test",
              "until_preproc_test", "equalize_speedtest", "equalize_test", "until_fft_test")
    subprocess.run(["grcc", "-o", str(HERE),
                    *[str(HERE / (name + ".grc")) for name in graphs]], cwd=HERE, check=True)
    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve()
    if not args.validate:
        (output_dir / "preprocess").mkdir(parents=True, exist_ok=True)
        for estimator in estimators:
            for stage in ("channel_est", "equalize"):
                (output_dir / stage / estimator).mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for dataset in datasets:
        try:
            with tempfile.TemporaryDirectory(prefix="split-equalizer-", dir=work_dir) as temporary:
                temporary = Path(temporary)
                dump = temporary / "fft.iqdata"
                prepared = temporary / "preproc.iqdata"
                print(f"Preparing {dataset.name} -> temporary complex64 FFT dump", flush=True)
                run_stage(flowgraph("fft_dump", "--dataset", dataset, "--dump-path", dump), None)
                if not dump.is_file() or dump.stat().st_size == 0 or not Path(str(dump) + ".hdr").is_file():
                    raise RuntimeError(f"Empty or incomplete FFT dump for {dataset.name}")
                frequency = frequencies[dataset]
                if args.validate:
                    validate(dump, prepared, frequency, temporary, estimators,
                             HERE / "validation_logs" / dataset.name)
                else:
                    run_stage(flowgraph("preprocess_speedtest", "--dump-path", dump,
                                        "--freq", frequency),
                              output_dir / "preprocess" / (dataset.name + ".log"))
                    run_stage(flowgraph("preprocess_dump", "--dump-path", dump,
                                        "--freq", frequency, "--output-path", prepared), None)
                    remove_dump(dump)
                    for estimator in estimators:
                        estimated = temporary / "estimated.iqdata"
                        run_stage(flowgraph("channel_est_speedtest", "--dump-path", prepared,
                                            "--estimator", estimator),
                                  output_dir / "channel_est" / estimator / (dataset.name + ".log"))
                        run_stage(flowgraph("channel_est_dump", "--dump-path", prepared,
                                            "--estimator", estimator, "--output-path", estimated), None)
                        run_stage(flowgraph("equalize_speedtest", "--dump-path", estimated,
                                            "--freq", frequency),
                                  output_dir / "equalize" / estimator / (dataset.name + ".log"))
                        remove_dump(estimated)
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
