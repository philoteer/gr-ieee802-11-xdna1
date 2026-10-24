#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
GRAPHS = {
    "evm": "replaced-diagnostic-evm-cui.grc",
    "csi": "replaced-diagnostic-cui.grc",
}
CSI_METRICS = (
    "snr_db",
    "deepest_fade_db",
    "spectral_flatness_db",
    "noise_enhancement_db",
    "phase_residual_rms_rad",
    "rms_delay_spread_ns",
    "previous_csi_correlation",
)


def run(command, log_path, timeout, env=None):
    with log_path.open("w", encoding="utf-8") as log:
        try:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"{command[0]} failed; see {log_path}") from error


def build(algorithm, output_symbols, cnn_data_updater, jobs, timeout, directory):
    args = [f"JOBS={jobs}", f"DFRAME_EQUALIZER_OUTPUT_SYMBOLS={output_symbols}",
            f"EQUALIZER_ALGORITHM={algorithm}", "ENGINE=CLASSIC", "MODEL=MLP_16X16",
            f"CNN_DATA_UPDATER={cnn_data_updater}"]
    for target in ("check-build-mode", "all"):
        run(["make", "-C", str(KERNEL), *args, target],
            directory / f"build_{target}.log", timeout)
    marker = (KERNEL / "build/frame_equalizer_build_mode.txt").read_text(encoding="ascii")
    if (f"EQUALIZER_ALGORITHM={algorithm}\n" not in marker or
            f"DFRAME_EQUALIZER_OUTPUT_SYMBOLS={output_symbols}\n" not in marker or
            f"CNN_DATA_UPDATER={cnn_data_updater}\n" not in marker):
        raise RuntimeError(f"Kernel build mode mismatch: {marker}")


def configure_graph(kind, algorithm, capture, directory):
    graph = yaml.safe_load((ROOT / GRAPHS[kind]).read_text(encoding="utf-8"))
    output = {
        "diagnostic_estimator": algorithm,
        "diagnostic_output": str(directory / "frames.jsonl"),
        "blocks_file_source_0": str(capture),
        "blocks_file_sink_0": str(directory / "decoded.pcap"),
    }
    if kind == "evm":
        output["diagnostic_evm_output"] = str(directory / "evm.jsonl")
    for block in graph["blocks"]:
        name = block["name"]
        if name in output:
            key = "file" if name.startswith("blocks_file_") else "value"
            block["parameters"][key] = repr(output.pop(name))
    if output:
        raise RuntimeError(f"Missing GRC blocks: {', '.join(output)}")
    graph_path = directory / "comparison.grc"
    graph_path.write_text(yaml.safe_dump(graph, sort_keys=False), encoding="utf-8")
    run(["grcc", "-o", str(directory), str(graph_path)],
        directory / "grcc.log", 120)
    script = directory / "wifi_rx.py"
    if not script.is_file():
        raise RuntimeError(f"grcc did not generate {script}")
    return script


def records(path):
    if not path.is_file():
        raise RuntimeError(f"Missing diagnostic output: {path}")
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def summarize(directory, kind, algorithm):
    frames = records(directory / "frames.jsonl")
    if any(row.get("estimator") != algorithm for row in frames):
        raise RuntimeError(f"Unexpected estimator label in {directory / 'frames.jsonl'}")
    ber = [row for row in frames if row["event"] == "pre_fec_coded_ber"]
    bit_errors = sum(row["bit_errors"] for row in ber)
    bits = sum(row["bits_compared"] for row in ber)
    if not ber or not bits:
        raise RuntimeError(f"No coded BER samples in {directory / 'frames.jsonl'}")
    result = {
        "crc_ok": sum(row["event"] == "crc_ok" for row in frames),
        "ber_samples": len(ber),
        "bit_errors": bit_errors,
        "bits_compared": bits,
        "coded_ber": bit_errors / bits,
    }
    if kind == "evm":
        evm = [row for row in records(directory / "evm.jsonl") if row["event"] == "evm"]
        if not evm or any(row.get("estimator") != algorithm for row in evm):
            raise RuntimeError(f"Missing or mislabeled EVM samples in {directory / 'evm.jsonl'}")
        result.update({
            "evm_samples": len(evm),
            "complex_symbols_compared": sum(row["complex_symbols_compared"] for row in evm),
            "mean_raw_evm_percent": statistics.fmean(row["evm_rms_percent"] for row in evm),
            "median_raw_evm_percent": statistics.median(row["evm_rms_percent"] for row in evm),
            "cumulative_raw_evm_percent": evm[-1]["cumulative_evm_rms_percent"],
            "mean_gain_aligned_evm_percent": statistics.fmean(
                row["gain_aligned_evm_rms_percent"] for row in evm),
            "median_gain_aligned_evm_percent": statistics.median(
                row["gain_aligned_evm_rms_percent"] for row in evm),
            "encoding_counts": {str(encoding): sum(row["encoding"] == encoding for row in evm)
                                for encoding in sorted({row["encoding"] for row in evm})},
        })
    else:
        csi = [row for row in frames if row["event"] == "frame"]
        if not csi:
            raise RuntimeError(f"No CSI frames in {directory / 'frames.jsonl'}")
        result["csi_frames"] = len(csi)
        result["csi_mean"] = {
            metric: statistics.fmean(row["metrics"][metric] for row in csi
                                     if row["metrics"].get(metric) is not None)
            for metric in CSI_METRICS
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare CNN and LS on the diagnostic GRC capture")
    parser.add_argument("--capture", type=Path, default=ROOT / "data/raw_capture/wifi_iq_hackrf.bin")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/channel_estimator_comparison")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--build-timeout", type=int, default=1800)
    parser.add_argument("--run-timeout", type=int, default=1200)
    parser.add_argument("--kinds", nargs="+", choices=tuple(GRAPHS), default=("evm", "csi"))
    parser.add_argument("--algorithms", nargs="+", choices=("LS", "CNN"), default=("LS", "CNN"))
    parser.add_argument("--cnn-data-updaters", nargs="+", choices=("NULL", "MLP"), default=("NULL",))
    args = parser.parse_args()
    capture = args.capture.resolve()
    if not capture.is_file() or capture.stat().st_size % 2:
        parser.error(f"Capture must be an existing interleaved-byte IQ file: {capture}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"capture": str(capture), "capture_bytes": capture.stat().st_size, "runs": {}}
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM="offscreen", WIFI_DIAGNOSTIC_HEADLESS="1")
    for kind, output_symbols in (("evm", 1), ("csi", 0)):
        if kind not in args.kinds:
            continue
        result["runs"][kind] = {}
        configurations = [
            (algorithm, updater if algorithm == "CNN" else "NULL")
            for algorithm in args.algorithms
            for updater in (args.cnn_data_updaters if algorithm == "CNN" else ("NULL",))
        ]
        for algorithm, updater in configurations:
            name = algorithm.lower() if algorithm != "CNN" or updater == "NULL" else f"cnn_{updater.lower()}"
            directory = output_dir / kind / name
            directory.mkdir(parents=True, exist_ok=True)
            print(f"{kind.upper()} {algorithm}/{updater}: building kernel", flush=True)
            build(algorithm, output_symbols, updater, args.jobs, args.build_timeout, directory)
            script = configure_graph(kind, algorithm, capture, directory)
            print(f"{kind.upper()} {algorithm}/{updater}: replaying capture", flush=True)
            run(["python3", str(script)], directory / "run.log", args.run_timeout, env)
            result["runs"][kind][name] = summarize(directory, kind, algorithm)
            (output_dir / "summary.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result["runs"][kind][name], indent=2), flush=True)
    print(f"Comparison saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
