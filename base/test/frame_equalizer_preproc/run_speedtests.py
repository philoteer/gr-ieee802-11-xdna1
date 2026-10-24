#!/usr/bin/env python3
"""Validate or benchmark the native split frame-equalizer stages through EOF.

Usage: python3 base/test/frame_equalizer_preproc/run_speedtests.py --validate
Preparation output goes to the console; only benchmark logs are saved.
FFT/prepared dumps and validation PCAPs are temporary and removed after use.
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


def validate(fft, prepared, frequency, temporary):
    """Compare native decoding and both prepared-file receive-chain flowgraphs."""
    reference = temporary / "native.pcap"
    post_preproc = temporary / "post_preproc.pcap"
    replay = temporary / "replay.pcap"
    run_stage(flowgraph("until_fft_test", "--dump-path", fft,
                        "--freq", frequency, "--pcap-path", reference), None)
    run_stage(flowgraph("preprocess_dump", "--dump-path", fft,
                        "--freq", frequency, "--output-path", prepared), None)
    run_stage(flowgraph("until_preproc_test", "--dump-path", prepared,
                        "--freq", frequency, "--pcap-path", post_preproc), None)
    run_stage(flowgraph("equalize_test", "--dump-path", prepared,
                        "--freq", frequency, "--pcap-path", replay), None)
    compare_pcaps(reference, post_preproc, replay)


def compare_pcaps(reference, post_preproc, replay):
    """Validate framing and MAC contents, allowing the observed one-packet variation."""
    native_frames = pcap_frames(reference)
    post_preproc_frames = pcap_frames(post_preproc)
    replay_frames = pcap_frames(replay)
    print(f"Native reference: {sum(native_frames.values())} CRC-accepted packets", flush=True)
    print(f"Post-preprocess receive chain: {sum(post_preproc_frames.values())} packets", flush=True)
    print(f"Prepared-file replay: {sum(replay_frames.values())} packets", flush=True)
    for label, first, second in (
        ("Post-preprocess vs native", post_preproc_frames, native_frames),
        ("Equalize replay vs native", replay_frames, native_frames),
        ("Post-preprocess vs equalize replay", post_preproc_frames, replay_frames),
    ):
        matches = sum((first & second).values())
        difference = sum((first - second).values()) + sum((second - first).values())
        print(f"{label}: {matches} matching MAC frames; {difference} unmatched occurrences", flush=True)
        # A missing packet counts once; a replaced packet or one missing packet
        # in each run counts twice. Ignore this known small variation for now.
        if difference > 2:
            raise RuntimeError(f"{label}: mismatch exceeds the allowed one-packet variation")


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
    args = parser.parse_args()
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
                      HERE.parent / "until_fft/kernel/build"):
        for filename in ("final.xclbin", "insts.bin"):
            if not (directory / filename).is_file():
                parser.error(f"Missing built kernel file: {directory / filename}")

    graphs = ("until_fft", "preprocess_speedtest", "preprocess_dump",
              "until_preproc_test", "equalize_speedtest", "equalize_test", "until_fft_test")
    subprocess.run(["grcc", "-o", str(HERE),
                    *[str(HERE / (name + ".grc")) for name in graphs]], cwd=HERE, check=True)
    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve()
    if not args.validate:
        (output_dir / "preprocess").mkdir(parents=True, exist_ok=True)
        (output_dir / "equalize").mkdir(parents=True, exist_ok=True)
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
                    validate(dump, prepared, frequency, temporary)
                else:
                    run_stage(flowgraph("preprocess_speedtest", "--dump-path", dump,
                                        "--freq", frequency),
                              output_dir / "preprocess" / (dataset.name + ".log"))
                    run_stage(flowgraph("preprocess_dump", "--dump-path", dump,
                                        "--freq", frequency, "--output-path", prepared), None)
                    dump.unlink()
                    Path(str(dump) + ".hdr").unlink()
                    run_stage(flowgraph("equalize_speedtest", "--dump-path", prepared,
                                        "--freq", frequency),
                              output_dir / "equalize" / (dataset.name + ".log"))
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
