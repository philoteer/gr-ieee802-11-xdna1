#!/usr/bin/env python3
"""Benchmark the three receiver flowgraphs on matching raw SigMF I/Q windows."""

import argparse
import collections
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
# These names match the source .grc files and their generated GNU Radio classes.
RECEIVERS = ("speedtest_802_11", "speedtest_80211", "speedtest")
LABELS = dict(zip(RECEIVERS, ("gr-ieee802-11", "gr-ieee80211", "NPU")))
# The room and bus files sweep frequencies; workload() selects a fixed-frequency
# excerpt from each rather than benchmarking across a retune boundary.
CASES = (
    ("stress_test", 1),
    ("room_20260927", 1),
    ("room_20260927", 6),
    ("room_20260927", 13),
    ("206", 1),
    ("206", 6),
    ("206", 13),
)
# The extra profiling experiments use a representative subset to keep runtime
# reasonable. The baseline suite still covers every CASES entry.
PROFILE_CASES = (CASES[0], CASES[1], CASES[5])
# 'baseline' is the comparable rate measurement; the remaining modes are
# diagnostic because counters, tags, decoders, or larger buffers alter execution.
EXPERIMENTS = ("baseline", "blocks", "frames", "buffers", "decode", "perf")
RESULT_FILES = dict(zip(EXPERIMENTS, (
    "receiver_benchmarks.json", "receiver_block_profiles.json",
    "receiver_frame_counts.json", "receiver_buffer_experiment.json",
    "receiver_decode_diagnostics.json", "receiver_native_profiles.json",
)))
DEMOD_NAMES = dict(zip(RECEIVERS, (
    "ieee802_11_frame_equalizer_0", "ieee80211_demod_0",
    "mlir_aie_mlir_aie_80211_phy_0",
)))


def case_arg(value):
    try:
        dataset, channel = value.rsplit(":", 1)
        channel = int(channel)
        if not dataset or Path(dataset).name != dataset or not 1 <= channel <= 14:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use DATASET:CHANNEL, e.g. room_20260927:6")
    return dataset, channel


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def workload(dataset, channel, count, skip_seconds):
    """Resolve a SigMF channel window and validate its raw complex-I/Q bounds."""
    path = ROOT / "data/raw_capture" / f"{dataset}.sigmf-data"
    meta_path = path.with_suffix(".sigmf-meta")
    meta = json.loads(meta_path.read_text())
    global_meta = meta["global"]
    rate = global_meta["core:sample_rate"]
    if (global_meta["core:datatype"] != "ci16_le" or rate != 20000000
            or global_meta.get("core:num_channels", 1) != 1):
        raise ValueError(f"{meta_path}: receivers require single-channel ci16_le at 20 MS/s")
    frequency = (2484 if channel == 14 else 2407 + channel * 5) * 1e6
    captures = meta["captures"]
    index = next((i for i, capture in enumerate(captures)
                  if abs(capture["core:frequency"] - frequency) < 1), None)
    if index is None:
        raise ValueError(f"{meta_path}: no capture for channel {channel}")
    file_size = path.stat().st_size
    if file_size % 4:
        raise ValueError(f"{path}: incomplete ci16_le sample")
    # SigMF offsets are complex samples. Each ci16_le complex sample is four bytes.
    start = int(captures[index]["core:sample_start"]) + int(skip_seconds * rate)
    end = (int(captures[index + 1]["core:sample_start"])
           if index + 1 < len(captures) else file_size // 4)
    if start + count > min(end, file_size // 4):
        raise ValueError(f"{dataset}:{channel}: requested window crosses capture boundary or EOF")
    return {
        "path": str(path.relative_to(ROOT)), "dataset": dataset, "channel": channel,
        "sample_start": start, "samples": count, "frequency": frequency,
        "sample_rate": int(rate), "file_bytes": file_size,
        "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
    }


def warm_file(window):
    """Read the selected bytes before timing so results aren't cold-disk tests."""
    with (ROOT / window["path"]).open("rb") as stream:
        stream.seek(4 * window["sample_start"])
        remaining = window["samples"] * 4
        while remaining:
            data = stream.read(min(8 * 1024 * 1024, remaining))
            if not data:
                raise RuntimeError(f"Unexpected EOF warming {window['path']}")
            remaining -= len(data)


def leaf_blocks(obj, prefix=""):
    """Include blocks inside hierarchies when applying/reporting block counters."""
    from gnuradio import gr
    for name, block in vars(obj).items():
        if isinstance(block, gr.hier_block2):
            yield from leaf_blocks(block, prefix + name + ".")
        elif isinstance(block, gr.block):
            yield prefix + name, block


def counters(blocks):
    """Snapshot GNU Radio's per-block thread-CPU and buffer counters."""
    result = {}
    for name, block in blocks:
        result[name] = {
            "work_ns": block.pc_work_time_total(),
            "avg_work_ns": block.pc_work_time_avg(),
            "avg_noutput": block.pc_noutput_items_avg(),
            "avg_nproduced": block.pc_nproduced_avg(),
            "input_fullness": list(block.pc_input_buffers_full_avg()),
            "output_fullness": list(block.pc_output_buffers_full_avg()),
            "items_read": block.nitems_read(0) if block.input_signature().max_streams() else 0,
            "items_written": block.nitems_written(0) if block.output_signature().max_streams() else 0,
        }
    return result


def make_tag_sink(receiver):
    import numpy as np
    import pmt
    from gnuradio import gr

    class TagSink(gr.sync_block):
        def __init__(self):
            # Match each demodulator's native stream item type and vector width.
            dtype = np.float32 if receiver == "speedtest_80211" else np.uint8
            signature = (dtype, 48) if receiver == "speedtest_802_11" else dtype
            gr.sync_block.__init__(self, "tag_counter", [signature], [])
            self.histogram = collections.Counter()
            self.frame_count = 0
            self.declared_output = 0

        def work(self, inputs, outputs):
            # Group metadata tags at the same stream offset into one candidate.
            size = len(inputs[0])
            grouped = collections.defaultdict(dict)
            for tag in self.get_tags_in_window(0, 0, size):
                key = pmt.symbol_to_string(tag.key)
                if key in ("format", "mcs", "len", "total", "encoding", "frame bytes"):
                    grouped[tag.offset][key] = pmt.to_python(tag.value)
            for fields in grouped.values():
                if "format" in fields:
                    self.frame_count += 1
                    self.histogram[f"format={fields['format']},mcs={fields.get('mcs')}"] += 1
                    self.declared_output += fields.get("total", 0)
                elif "frame bytes" in fields:
                    self.frame_count += 1
                    self.histogram[f"encoding={fields.get('encoding')}"] += 1
            return size

    return TagSink()


def worker(args):
    """Run one graph in a fresh subprocess and report its timed input-rate span."""
    os.chdir(ROOT)
    os.environ["GR_CONF_PERFCOUNTERS_ON"] = "True" if args.counters else "False"
    os.environ["GR_CONF_PERFCOUNTERS_CLOCK"] = "thread"
    from gnuradio import blocks, gr
    import pmt

    gr.prefs().set_bool("PerfCounters", "on", args.counters)
    gr.prefs().set_string("PerfCounters", "clock", "thread")
    dataset, channel = args.worker_case
    window = workload(dataset, channel, args.samples, args.skip_seconds)
    warm_file(window)
    sys.path.insert(0, str(args.graph_dir))
    module = importlib.import_module(args.worker_receiver)
    construction_start = time.monotonic()
    tb = getattr(module, args.worker_receiver)()
    construction_s = time.monotonic() - construction_start
    if args.worker_receiver == "speedtest":
        batch_size = getattr(tb, DEMOD_NAMES["speedtest"]).output_multiple()
        if args.samples % batch_size:
            raise ValueError(f"--samples must be divisible by NPU batch size {batch_size}")
    if hasattr(tb, "set_freq"):
        tb.set_freq(window["frequency"])
    converter = tb.blocks_interleaved_short_to_complex_0
    # Replace only the graph's source, preserving the source-to-demodulation
    # connections while selecting an exact window from the raw SigMF payload.
    old_source = getattr(tb, "blocks_file_source_0",
                         getattr(tb, "blocks_file_source_0_0", None))
    source = blocks.file_source(gr.sizeof_short, window["path"], False,
                               2 * window["sample_start"], 2 * args.samples)
    source.set_begin_tag(pmt.PMT_NIL)
    tb.disconnect(old_source, converter)
    old_source.close()
    tb.connect(source, converter)
    sink = tb.blocks_null_sink_0
    demod = getattr(tb, DEMOD_NAMES[args.worker_receiver])
    extras = [("benchmark_source", source)]
    if args.tags:
        # Replacing the null sink lets this diagnostic consume frame metadata.
        tb.disconnect(demod, sink)
        sink = make_tag_sink(args.worker_receiver)
        tb.connect(demod, sink)
        extras.append(("tag_counter", sink))
    decoder = messages = None
    if args.decode:
        # These message decoders accept different stream shapes: 48-byte carrier
        # vectors for gr-ieee802-11/NPU, and scalar float LLRs for gr-ieee80211.
        messages = blocks.message_debug(False)
        if args.worker_receiver == "speedtest_80211":
            from gnuradio import ieee80211
            decoder = ieee80211.decode(False)
            tb.connect(demod, decoder)
        else:
            import ieee802_11
            decoder = ieee802_11.decode_mac(False, False)
            if args.worker_receiver == "speedtest":
                vectors = blocks.stream_to_vector(gr.sizeof_char, 48)
                tb.connect(demod, vectors, decoder)
                extras.append(("decoder_vectors", vectors))
            else:
                tb.connect(demod, decoder)
        tb.msg_connect((decoder, "out"), (messages, "store"))
        extras.extend((("decoder", decoder), ("decoded_messages", messages)))
    leaves = [(name, block) for name, block in leaf_blocks(tb)
              if block is not old_source and (not args.tags or block is not tb.blocks_null_sink_0)]
    leaves.extend(extras)
    if args.buffer_bytes:
        # Apply a controlled minimum to each stream output; message-only blocks
        # have no stream ports and are skipped naturally.
        for _, block in leaves:
            signature = block.output_signature()
            for port in range(signature.max_streams()):
                items = max(1, args.buffer_bytes // signature.sizeof_stream_item(port))
                block.set_min_output_buffer(port, items)
    # Measure the slope between 1/8 and 7/8 of the converter's written count.
    # Counting here uses one common complex-I/Q boundary despite output formats
    # that differ between hard-decision and soft-LLR receivers.
    start_count = args.samples // 8
    end_count = args.samples * 7 // 8
    first = last = pc_start = pc_end = None
    started = time.monotonic()
    tb.start()
    try:
        while True:
            count = converter.nitems_written(0)
            now = time.monotonic()
            if first is None and count >= start_count:
                first = (count, now, time.process_time(), resource.getrusage(resource.RUSAGE_SELF))
                if args.counters:
                    pc_start = counters(leaves)
            if last is None and count >= end_count:
                last = (count, now, time.process_time(), resource.getrusage(resource.RUSAGE_SELF))
                if args.counters:
                    pc_end = counters(leaves)
            if count >= args.samples:
                break
            if now - started > args.timeout:
                raise RuntimeError(f"Timed out at {count}/{args.samples} complex input samples")
            time.sleep(0.002)
        input_done_s = time.monotonic() - started
        # Let queued demodulator/decoder output settle before stopping the graph.
        # Some GNU Radio receiver state machines don't terminate cleanly at EOF.
        previous = None
        quiet_since = time.monotonic()
        drain_deadline = quiet_since + 1
        while time.monotonic() < drain_deadline:
            progress = (sink.nitems_read(0), demod.nitems_written(0))
            if decoder is not None:
                progress += (decoder.nitems_read(0), messages.num_messages())
            now = time.monotonic()
            if progress != previous:
                previous = progress
                quiet_since = now
            elif now - quiet_since >= 0.05:
                break
            time.sleep(0.002)
    finally:
        tb.stop()
        tb.wait()
    if first is None or last is None or last[0] <= first[0] or last[1] <= first[1]:
        raise RuntimeError("Too few samples to resolve the timed interval; increase --samples")
    if converter.nitems_written(0) != args.samples:
        raise RuntimeError("The receiver did not consume the complete selected input window")
    measured_s = last[1] - first[1]
    measured_count = last[0] - first[0]
    result = {
        "receiver": args.worker_receiver, "dataset": dataset, "channel": channel,
        "frequency": window["frequency"], "sample_start": window["sample_start"],
        "samples": args.samples, "input_samples": converter.nitems_written(0),
        "recording_seconds": args.samples / window["sample_rate"],
        "construction_s": construction_s, "input_done_s": input_done_s,
        "measured_samples": measured_count, "measured_s": measured_s,
        "msps": measured_count / measured_s / 1e6,
        "cpu_s": last[2] - first[2], "cpu_cores": (last[2] - first[2]) / measured_s,
        "major_faults": last[3].ru_majflt - first[3].ru_majflt,
        "minor_faults": last[3].ru_minflt - first[3].ru_minflt,
        "voluntary_switches": last[3].ru_nvcsw - first[3].ru_nvcsw,
        "involuntary_switches": last[3].ru_nivcsw - first[3].ru_nivcsw,
        "output_items": sink.nitems_read(0), "counters_enabled": args.counters,
        "tags_enabled": args.tags, "decode_enabled": args.decode,
        "buffer_bytes": args.buffer_bytes, "gnuradio_version": gr.version(),
    }
    if args.worker_receiver == "speedtest":
        result["npu_batch_size"] = batch_size
    if args.counters:
        # Counter snapshots are taken at the throughput interval boundaries;
        # summed block work omits scheduler costs and is not wall time.
        result["blocks"] = {}
        for name, end in pc_end.items():
            values = dict(end)
            for key in ("work_ns", "items_read", "items_written"):
                values[key] -= pc_start[name][key]
            result["blocks"][name] = values
        if not any(block["work_ns"] > 0 for block in result["blocks"].values()):
            raise RuntimeError("GNU Radio performance counters are unavailable or disabled")
    if args.tags:
        result.update(frames=sink.frame_count, frame_histogram=dict(sink.histogram),
                      declared_output=sink.declared_output)
    if decoder is not None:
        result.update(crc_pass_frames=messages.num_messages(),
                      decoder_items_read=decoder.nitems_read(0))
    result["shared_libraries_sha256"] = loaded_libraries()
    print("RESULT " + json.dumps(result), flush=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def loaded_libraries():
    """Fingerprint the receiver shared objects actually mapped into this worker."""
    names = ("libgnuradio-ieee802_11", "libgnuradio-ieee80211", "libgnuradio-mlir_aie")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if path.name.startswith(names):
                paths.add(path)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def run_worker(command, log, timeout):
    """Run a worker in its own process group so timeout/interrupt also stops perf."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def prepare_graphs(receivers, directory):
    """Generate isolated flowgraph copies without changing the source .grc files."""
    import yaml
    directory.mkdir()
    for receiver in receivers:
        graph = yaml.safe_load((ROOT / f"{receiver}.grc").read_text())
        if graph["options"]["parameters"]["generate_options"] != "no_gui":
            raise ValueError(f"{receiver}.grc must use no_gui generation")
        graph["options"]["parameters"]["id"] = receiver
        for block in graph["blocks"]:
            if block["id"] == "blocks_file_source":
                # The worker replaces this placeholder with the selected capture window.
                block["parameters"]["file"] = "/dev/null"
        graph_path = directory / f"{receiver}.grc"
        graph_path.write_text(yaml.safe_dump(graph, sort_keys=False), encoding="utf-8")
        with (directory / f"{receiver}-grcc.log").open("w") as log:
            subprocess.run(["grcc", "-o", str(directory), str(graph_path)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
        if not (directory / f"{receiver}.py").is_file():
            raise RuntimeError(f"grcc did not generate {receiver}.py")


def command_output(command):
    try:
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    except OSError as error:
        return {"error": str(error)}


def manifest(args, cases, receivers, experiments):
    """Capture enough input, code, image, and host details to interpret a run later."""
    files = {f"{receiver}.grc": ROOT / f"{receiver}.grc" for receiver in receivers}
    files["utils/benchmark_receivers.py"] = Path(__file__)
    marker = ROOT / "kernel/build/frame_equalizer_build_mode.txt"
    build_mode = marker.read_text() if "speedtest" in receivers else None
    if build_mode is not None and "DFRAME_EQUALIZER_OUTPUT_SYMBOLS=0" not in build_mode.splitlines():
        raise ValueError("The NPU benchmark requires a production CSI image (DFRAME_EQUALIZER_OUTPUT_SYMBOLS=0)")
    if "speedtest" in receivers:
        for name in ("final.xclbin", "insts.bin"):
            path = ROOT / "kernel/build" / name
            files[str(path.relative_to(ROOT))] = path
    return {
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "root": str(ROOT), "python": sys.version, "platform": platform.platform(),
        "command": sys.argv, "receivers": receivers, "experiments": experiments,
        "settings": {
            "samples": args.samples, "decode_samples": args.decode_samples,
            "skip_seconds": args.skip_seconds, "repeats": args.repeats,
            "diagnostic_repeats": args.diagnostic_repeats,
            "buffer_bytes": args.large_buffer_bytes, "poll_seconds": 0.002,
            "timed_input_fraction": [0.125, 0.875], "timeout": args.timeout,
        },
        "windows": [workload(*case, args.samples, args.skip_seconds) for case in cases],
        "decode_windows": [workload(*case, args.decode_samples, args.skip_seconds)
                           for case in cases] if "decode" in experiments or "perf" in experiments else [],
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in files.items()},
        "npu_build_mode": build_mode,
        "lscpu": command_output(["lscpu"]),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "perf_version": command_output(["perf", "--version"]) if "perf" in experiments else None,
    }


def run_once(args, experiment, case, receiver, repeat, samples, decode=False):
    """Launch one measurement, save its log, and post-process optional perf data."""
    dataset, channel = case
    tag = f"{experiment}-{dataset}-ch{channel}-{receiver}-{repeat}"
    if decode and experiment == "perf":
        tag += "-decode"
    command = [sys.executable, str(Path(__file__).resolve()),
               "--worker-receiver", receiver, "--worker-case", f"{dataset}:{channel}",
               "--graph-dir", str(args.output_dir / "grc"),
               "--samples", str(samples), "--skip-seconds", str(args.skip_seconds),
               "--timeout", str(args.timeout)]
    if experiment == "blocks":
        command.append("--counters")
    if experiment in ("frames", "decode"):
        command.append("--tags")
    if decode:
        command.append("--decode")
    if experiment == "buffers":
        command.extend(("--buffer-bytes", str(args.large_buffer_bytes)))
    if experiment == "perf":
        # perf records user-mode cycles with DWARF stacks; its run is diagnostic,
        # not a substitute for the uninstrumented baseline throughput measurement.
        command = ["perf", "record", "-q", "-e", "cycles:u", "-F", "199",
                   "--call-graph", "dwarf,8192", "-o", str(args.output_dir / f"{tag}.data"),
                   "--", *command]
    log_path = args.output_dir / f"{tag}.log"
    with log_path.open("w") as log:
        try:
            returncode = run_worker(command, log, args.timeout + 60)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Worker timed out; see {log_path}") from error
    if returncode:
        raise RuntimeError(f"Worker exited with status {returncode}; see {log_path}")
    lines = [line[7:] for line in log_path.read_text().splitlines() if line.startswith("RESULT ")]
    if len(lines) != 1:
        raise RuntimeError(f"Missing/ambiguous benchmark result; see {log_path}")
    result = json.loads(lines[0])
    result.update(experiment=experiment, repeat=repeat, log=log_path.name)
    if experiment == "perf":
        result["perf_data"] = f"{tag}.data"
        report_path = args.output_dir / f"{tag}-perf.txt"
        with report_path.open("w") as report:
            subprocess.run([
                "perf", "report", "-i", str(args.output_dir / result["perf_data"]),
                "--stdio", "--no-children", "--sort", "dso,symbol", "--percent-limit", "1",
                "--call-graph", "none", "--field-separator", "|",
            ], stdout=report, stderr=subprocess.STDOUT, check=True, timeout=120)
        result["perf_report"] = report_path.name
    print(f"{experiment:8} {dataset:15} ch{channel:<2} {LABELS[receiver]:15} "
          f"repeat={repeat + 1} {result['msps']:8.2f} MS/s "
          f"cpu={result['cpu_cores']:.2f} output={result['output_items']}", flush=True)
    return result


def write_summary(directory, results):
    """Regenerate the human-readable report as each JSON experiment completes."""
    lines = ["# Receiver benchmark results", "",
             "Input rates are million complex samples/second. Runs are serialized,",
             "cache-warmed, and timed over the middle 75% of each input window.",
             "Construction, startup, and EOF handling are outside the timed interval.",
             "See `manifest.json` for input windows, settings, and build fingerprints.", "",
             "Only baseline rows use native null sinks without profiling or packet decoding.",
             "Frame-start tags are SIGNAL candidates, not CRC-valid packets.", ""]
    for experiment in EXPERIMENTS:
        runs = results.get(experiment, [])
        if not runs:
            continue
        lines.extend((f"## {experiment.capitalize()}", ""))
        if experiment == "perf":
            for run in runs:
                suffix = " + packet decoder" if run["decode_enabled"] else ""
                lines.append(f"- {run['dataset']} ch{run['channel']}, {LABELS[run['receiver']]}{suffix}: "
                             f"[{run['perf_report']}]({run['perf_report']})")
            lines.append("")
            continue
        groups = collections.defaultdict(list)
        for run in runs:
            groups[(run["dataset"], run["channel"], run["receiver"])].append(run)
        lines.extend(("| Recording | Channel | Receiver | Median MS/s [min, max] | CPU cores | Output items | Candidates | CRC pass |",
                      "|---|---:|---|---:|---:|---:|---:|---:|"))
        for (dataset, channel, receiver), group in groups.items():
            rates = [run["msps"] for run in group]
            def middle(key):
                return statistics.median(run[key] for run in group)
            frames = f"{middle('frames'):,.0f}" if "frames" in group[0] else "—"
            crc = f"{middle('crc_pass_frames'):,.0f}" if "crc_pass_frames" in group[0] else "—"
            lines.append(f"| {dataset} | {channel} | {LABELS[receiver]} | "
                         f"{statistics.median(rates):.2f} [{min(rates):.2f}, {max(rates):.2f}] | "
                         f"{middle('cpu_cores'):.2f} | {middle('output_items'):,.0f} | {frames} | {crc} |")
        lines.append("")
        if experiment == "buffers" and results.get("baseline"):
            lines.extend(("### Buffer changes relative to baseline", ""))
            for (dataset, channel, receiver), group in groups.items():
                baseline = [run["msps"] for run in results["baseline"]
                            if (run["dataset"], run["channel"], run["receiver"]) == (dataset, channel, receiver)
                            and run["samples"] == group[0]["samples"]]
                if baseline:
                    change = statistics.median(run["msps"] for run in group) / statistics.median(baseline) - 1
                    lines.append(f"- {dataset} ch{channel}, {LABELS[receiver]}: {100 * change:+.1f}%")
            lines.append("")
        if experiment == "blocks":
            lines.extend(("### Leading block work times", "",
                          "Thread-CPU work counters exclude scheduler work and perturb throughput.", ""))
            for run in runs:
                total = sum(block["work_ns"] for block in run["blocks"].values())
                leaders = sorted(run["blocks"].items(), key=lambda pair: -pair[1]["work_ns"])[:5]
                lines.append(f"- {run['dataset']} ch{run['channel']}, {LABELS[run['receiver']]}:")
                for name, block in leaders:
                    share = 100 * block["work_ns"] / total if total else 0
                    lines.append(f"  - `{name}`: {block['work_ns'] / 1e9:.3f} CPU seconds, "
                                 f"{share:.1f}% of summed block work")
            lines.append("")
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def suite(args):
    """Run selected experiments serially and persist partial results after each run."""
    cases = tuple(dict.fromkeys(args.cases or CASES))
    receivers = tuple(dict.fromkeys(args.receivers or RECEIVERS))
    if args.cpu_only:
        receivers = tuple(receiver for receiver in receivers if receiver != "speedtest")
    if not receivers:
        raise ValueError("No receivers selected")
    experiments = EXPERIMENTS if args.all else tuple(dict.fromkeys(args.experiments))
    profiles = tuple(case for case in cases if case in PROFILE_CASES) or cases
    cpu_receivers = tuple(receiver for receiver in receivers if receiver != "speedtest")
    info = manifest(args, cases, receivers, experiments)
    if args.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        args.output_dir = ROOT / "build/receiver_benchmarks" / stamp
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "manifest.json", info)
    print(f"Results: {args.output_dir}", flush=True)
    results = {}
    write_summary(args.output_dir, results)
    try:
        prepare_graphs(receivers, args.output_dir / "grc")
        for experiment in experiments:
            results[experiment] = []
            write_json(args.output_dir / RESULT_FILES[experiment], results[experiment])
            selected_cases = profiles if experiment in ("buffers", "decode", "perf") else cases
            selected_receivers = cpu_receivers if experiment in ("blocks", "buffers", "perf") else receivers
            repeats = args.repeats if experiment in ("baseline", "buffers") else args.diagnostic_repeats
            samples = args.decode_samples if experiment == "decode" else args.samples
            for case in selected_cases:
                case_receivers = selected_receivers
                # The short decode suite is centered on the stress capture; other
                # explicitly requested cases still compare the two CPU decoders.
                if experiment == "decode" and case != ("stress_test", 1):
                    case_receivers = cpu_receivers
                for repeat in range(repeats):
                    offset = repeat % len(case_receivers) if case_receivers else 0
                    order = case_receivers[offset:] + case_receivers[:offset]
                    for receiver in order:
                        result = run_once(args, experiment, case, receiver, repeat, samples,
                                          decode=experiment == "decode")
                        results[experiment].append(result)
                        write_json(args.output_dir / RESULT_FILES[experiment], results[experiment])
                        write_summary(args.output_dir, results)
            if experiment == "perf" and ("stress_test", 1) in cases and "speedtest_80211" in receivers:
                # Capture the known soft-Viterbi hotspot separately from front-end perf.
                result = run_once(args, experiment, ("stress_test", 1), "speedtest_80211", 0,
                                  args.decode_samples, decode=True)
                results[experiment].append(result)
                write_json(args.output_dir / RESULT_FILES[experiment], results[experiment])
                write_summary(args.output_dir, results)
        info["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        info["status"] = "completed"
    except Exception as error:
        info["status"] = "failed"
        info["error"] = str(error)
        raise
    except KeyboardInterrupt:
        info["status"] = "interrupted"
        raise
    finally:
        write_json(args.output_dir / "manifest.json", info)
    print(f"Summary: {args.output_dir / 'summary.md'}")


def main():
    """Parse the user-facing suite options or dispatch an internal worker process."""
    parser = argparse.ArgumentParser(
        description="Reproduce the raw-I/Q receiver comparison using grcc-generated speed-test graphs.",
        epilog="Default: seven SigMF windows, all three receivers, three baseline repetitions. "
               "--all adds block profiles, frame tags, buffer experiments, CRC/decode diagnostics, "
               "and native perf profiles. All runs are serialized; the existing NPU image is used.")
    parser.add_argument("--all", action="store_true", help="Run all six experiment types")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=["baseline"])
    parser.add_argument("--cases", nargs="+", type=case_arg, metavar="DATASET:CHANNEL")
    parser.add_argument("--receivers", nargs="+", choices=RECEIVERS)
    parser.add_argument("--cpu-only", action="store_true", help="Exclude the NPU receiver")
    parser.add_argument("--samples", type=positive_int, default=134217728,
                        help="Complex samples per main window (default: 134217728)")
    parser.add_argument("--decode-samples", type=positive_int, default=16777216,
                        help="Complex samples per short decoder diagnostic (default: 16777216)")
    parser.add_argument("--skip-seconds", type=float, default=10,
                        help="Recorded seconds after the first matching capture entry (default: 10)")
    parser.add_argument("--repeats", type=positive_int, default=3,
                        help="Repetitions for baseline and buffer experiments (default: 3)")
    parser.add_argument("--diagnostic-repeats", type=positive_int, default=1)
    parser.add_argument("--large-buffer-bytes", type=positive_int, default=524288)
    parser.add_argument("--timeout", type=positive_int, default=120,
                        help="Processing timeout per worker, seconds (default: 120)")
    parser.add_argument("--output-dir", type=Path, help="New results directory; default: build/receiver_benchmarks/TIMESTAMP")
    parser.add_argument("--list-cases", action="store_true", help="Print the seven default dataset/channel windows")
    parser.add_argument("--worker-receiver", choices=RECEIVERS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", type=case_arg, help=argparse.SUPPRESS)
    parser.add_argument("--graph-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--counters", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tags", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--decode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--buffer-bytes", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.skip_seconds < 0:
        parser.error("--skip-seconds must be nonnegative")
    if args.list_cases:
        for dataset, channel in CASES:
            print(f"{dataset}:{channel}")
        return
    try:
        if args.worker_receiver:
            if args.worker_case is None or args.graph_dir is None:
                parser.error("Internal worker requires --worker-case and --graph-dir")
            worker(args)
        else:
            if not shutil.which("grcc"):
                raise RuntimeError("grcc is not on PATH")
            if (args.all or "perf" in args.experiments) and not shutil.which("perf"):
                raise RuntimeError("perf is not on PATH; choose experiments without perf")
            suite(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed measurements remain in the results directory.\n")


if __name__ == "__main__":
    main()
