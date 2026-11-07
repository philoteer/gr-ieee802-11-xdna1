#!/usr/bin/env python3
"""Decode a bounded real ci16_le SigMF recording through the complete receiver.

Run with the Python interpreter used by the installed GNU Radio bindings,
normally /usr/bin/python3 rather than the MLIR-AIE compiler virtualenv.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import time

from gnuradio import blocks, gr, mlir_aie
import ieee802_11
import numpy as np
import pmt


ROOT = Path(__file__).resolve().parents[3]
BATCH_SIZE = 16384


class PacketSink(gr.basic_block):
    """Only decode_mac's CRC-checked output PDUs reach this sink."""

    def __init__(self):
        gr.basic_block.__init__(self, "CRC-valid packet collector", None, None)
        self.packets = []
        self.encodings = Counter()
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.receive)

    def receive(self, message):
        metadata, payload = pmt.car(message), pmt.cdr(message)
        self.packets.append(bytes(pmt.u8vector_elements(payload)))
        encoding = pmt.dict_ref(metadata, pmt.intern("encoding"), pmt.from_long(-1))
        self.encodings[str(int(pmt.to_python(encoding)))] += 1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_window(path, start, count):
    result = hashlib.sha256()
    with path.open("rb") as capture:
        capture.seek(4 * start)
        remaining = 4 * count
        while remaining:
            chunk = capture.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError("capture ended while hashing the selected window")
            result.update(chunk)
            remaining -= len(chunk)
    return result.hexdigest()


def write_pcap(path, packets):
    # decode_mac strips FCS and emits native IEEE 802.11 MAC frames (DLT 105).
    with path.open("xb") as output:
        output.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 105))
        for index, packet in enumerate(packets):
            # Ordinal timestamps only; these are not SDR arrival-time estimates.
            seconds, microseconds = divmod(index, 1_000_000)
            output.write(struct.pack("<IIII", seconds, microseconds, len(packet), len(packet)))
            output.write(packet)


def run(args):
    capture = args.capture.resolve()
    metadata_path = args.metadata or capture.with_suffix(".sigmf-meta")
    metadata = json.loads(metadata_path.read_text())
    if metadata["global"]["core:datatype"] != "ci16_le":
        raise ValueError("capture must contain interleaved little-endian int16 IQ")
    if metadata["global"]["core:sample_rate"] != 20_000_000:
        raise ValueError("this receiver requires a 20 MHz recording")
    if args.start < 0 or args.samples <= 0 or args.samples % BATCH_SIZE:
        raise ValueError("samples must be a positive multiple of 16384; start must be nonnegative")
    if 4 * (args.start + args.samples) > capture.stat().st_size:
        raise ValueError("requested sample window exceeds the capture")
    frequency = None
    for entry in sorted(metadata["captures"], key=lambda item: item["core:sample_start"]):
        if entry["core:sample_start"] <= args.start and "core:frequency" in entry:
            frequency = entry["core:frequency"]
    if frequency is None:
        raise ValueError("no capture frequency at the selected starting sample")
    args.output.mkdir(parents=True, exist_ok=False)
    input_hash = digest_window(capture, args.start, args.samples)

    top = gr.top_block("Recorded-IQ packet decoding regression", catch_exceptions=False)
    source = blocks.file_source(gr.sizeof_short, str(capture), False,
                                2 * args.start, 2 * args.samples)
    convert = blocks.interleaved_short_to_complex(False, False, 32768)
    tags = mlir_aie.sigmf_tag_injector(str(metadata_path.resolve()), gr.sizeof_gr_complex,
                                     "rx_freq", args.start)
    decisions = blocks.vector_sink_b(48)
    decoder = ieee802_11.decode_mac(False, False)
    packets = PacketSink()
    top.connect(source, convert, tags)
    artifacts = {}
    if args.backend == "npu":
        kernel = args.kernel_dir.resolve()
        for name in ("final.xclbin", "insts.bin"):
            artifacts[name] = digest(kernel / name)
        packed = mlir_aie.complex64_to_cint16(False)
        phy = mlir_aie.mlir_aie_80211_phy(
            str(kernel / "final.xclbin"), str(kernel / "insts.bin"), "MLIR_AIE",
            BATCH_SIZE, frequency, args.slots, 16, "")
        vectors = blocks.stream_to_vector(gr.sizeof_char, 48)
        top.connect(tags, packed, phy, vectors)
    else:
        # Same floating-point LS receiver used by the project's CPU flowgraph.
        from gnuradio import fft
        delay = blocks.delay(gr.sizeof_gr_complex, 16)
        conjugate = blocks.conjugate_cc()
        correlate = blocks.multiply_cc()
        average = blocks.moving_average_cc(48, 1, 4000)
        magnitude = blocks.complex_to_mag(1)
        power = blocks.complex_to_mag_squared(1)
        power_average = blocks.moving_average_ff(64, 1, 4000)
        ratio = blocks.divide_ff(1)
        short = ieee802_11.sync_short(0.56, 2, False, False)
        long_delay = blocks.delay(gr.sizeof_gr_complex, 320)
        long = ieee802_11.sync_long(320, False, False)
        symbols = blocks.stream_to_vector(gr.sizeof_gr_complex, 64)
        transform = fft.fft_vcc(64, True, (), True, 1)
        vectors = ieee802_11.frame_equalizer(ieee802_11.Equalizer(0), frequency,
                                           20e6, False, False)
        top.connect(tags, delay, conjugate, (correlate, 1))
        top.connect(tags, (correlate, 0))
        top.connect(correlate, average, magnitude, (ratio, 0))
        top.connect(tags, power, power_average, (ratio, 1))
        top.connect(delay, (short, 0))
        top.connect(average, (short, 1))
        top.connect(ratio, (short, 2))
        top.connect(short, (long, 0))
        top.connect(short, long_delay, (long, 1))
        top.connect(long, symbols, transform, vectors)
    top.connect(vectors, decisions)
    top.connect(vectors, decoder)
    top.msg_connect(decoder, "out", packets, "in")
    started = time.monotonic()
    top.start()
    top.wait()
    elapsed = time.monotonic() - started

    tag_counts = Counter(pmt.symbol_to_string(tag.key) for tag in decisions.tags())
    report = {
        "backend": args.backend, "capture": str(capture),
        "metadata": str(metadata_path.resolve()), "start_sample": args.start,
        "sample_count": args.samples, "frequency_hz": frequency, "slots": args.slots,
        "input_sha256": input_hash, "metadata_sha256": digest(metadata_path),
        "artifact_sha256": artifacts, "elapsed_seconds": elapsed,
        "decision_symbols": len(decisions.data()) // 48, "tags": dict(tag_counts),
        "crc_valid_packets": len(packets.packets), "encodings": dict(packets.encodings),
        "packet_sha256": [hashlib.sha256(packet).hexdigest() for packet in packets.packets],
    }
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for key in ("capture", "start_sample", "sample_count", "input_sha256", "metadata_sha256"):
            if key in reference and report[key] != reference[key]:
                raise ValueError(f"reference uses different input: {key}")
        actual = Counter(report["packet_sha256"])
        expected = Counter(reference["packet_sha256"])
        report["reference_comparison"] = {
            "report": str(args.reference.resolve()),
            "matching_packets": sum((actual & expected).values()),
            "missing_packets": sum((expected - actual).values()),
            "additional_packets": sum((actual - expected).values()),
            "byte_identical_packet_multiset": actual == expected,
        }
    write_pcap(args.output / "packets.pcap", packets.packets)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.save(args.output / "decisions.npy", np.asarray(decisions.data(), dtype=np.uint8))
    print(json.dumps(report, indent=2), flush=True)
    if len(packets.packets) < args.min_packets:
        raise SystemExit(f"FAIL: expected at least {args.min_packets} CRC-valid packets")
    if args.reference and not report["reference_comparison"]["byte_identical_packet_multiset"]:
        raise SystemExit("FAIL: decoded packet bytes/counts differ from the reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "AI-model-CNN-Channel-est/data/raw_capture/stress_test.sigmf-data")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=327680)
    parser.add_argument("--backend", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--kernel-dir", type=Path, default=ROOT / "AI-model-basic/kernel/build")
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--min-packets", type=int, default=1)
    parser.add_argument("--reference", type=Path, help="require identical CRC-valid packet bytes and counts")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
