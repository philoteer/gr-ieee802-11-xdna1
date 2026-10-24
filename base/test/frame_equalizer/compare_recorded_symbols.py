#!/usr/bin/env python3
"""Compare GNU Radio and NPU equalized symbols on a recorded FFT prefix.

IMPORTANT BUILD MODE: like test_symbols.py, this compares the NPU's normalized
equalized symbols against GNU Radio, so it requires the NPU built with
``DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1`` (the "symbols" build). A CSI-mode build
emits the channel estimate instead and would compare the wrong data. The
shared ``test_symbols.require_symbols_mode()`` guard refuses a mismatched build
by checking ``build/frame_equalizer_build_mode.txt``. Rebuild with
``make compare EQUALIZER_ALGORITHM=LS`` (or ``LMS``/``COMB``/``STA``) first.
"""

import argparse
from pathlib import Path

import numpy as np
import pmt
from gnuradio import blocks, gr
import ieee802_11

import test_symbols as test


def read_reference(path, header, sample_limit):
    flowgraph = gr.top_block()
    source = blocks.file_meta_source(str(path), False, True, str(header))
    head = blocks.head(gr.sizeof_gr_complex, sample_limit)
    raw = blocks.vector_sink_c()
    flowgraph.connect(source, head)
    flowgraph.connect(head, raw)
    flowgraph.run()
    tags = [tag for tag in raw.tags() if pmt.symbol_to_string(tag.key) == "wifi_start"]
    return np.asarray(raw.data(), dtype=np.complex64), tags


def run_npu_recording(handle, samples, tags):
    symbols = []
    chunks = len(samples) // test.N
    limits = np.iinfo(np.int32)
    for chunk in range(chunks):
        start = chunk * test.N
        values = samples[start:start + test.N]
        scaled = np.empty(2 * test.N, dtype=np.int32)
        scaled[0::2] = np.clip(np.rint(values.real * test.Q15),
                               limits.min, limits.max).astype(np.int32)
        scaled[1::2] = np.clip(np.rint(values.imag * test.Q15),
                               limits.min, limits.max).astype(np.int32)
        metadata = np.zeros((test.TILES, test.META_WORDS), dtype=np.int32)
        metadata[:, 0] = test.TILE
        for tag in tags:
            offset = int(tag.offset)
            if not start <= offset < start + test.N:
                continue
            local = offset - start
            tile = local // test.TILE
            index = int(metadata[tile, 1])
            if index >= 31:
                continue
            metadata[tile, 1] += 1
            metadata[tile, 2 + 2*index] = local % test.TILE
            metadata[tile, 3 + 2*index] = round(
                pmt.to_double(tag.value) * (1 << 29))
        _, chunk_symbols, _ = test.run_npu(
            handle, scaled, metadata.reshape(-1))
        symbols.extend(chunk_symbols)
    return np.asarray(symbols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    test.add_algorithm_argument(parser)
    parser.add_argument("--input", type=Path,
        default=Path("/home/piloteer/바탕화면/fft_full.iqdata"))
    parser.add_argument("--header", type=Path,
        default=Path("/home/piloteer/바탕화면/fft_full.iqdata.hdr"))
    parser.add_argument("--samples", type=int, default=2097152)
    args = parser.parse_args()
    test.require_symbols_mode(args.algorithm)
    samples, tags = read_reference(args.input, args.header, args.samples)
    handle, rtp_dir = test.load_kernel()
    matched_actual = []
    matched_reference = []
    gnu_only = 0
    npu_only = 0
    decision_disagreements = 0
    matched_evm = []
    unusable = 0
    for tag in tags:
        start = int(tag.offset)
        if start + 4 * 64 > len(samples):
            continue
        cfo = round(pmt.to_double(tag.value) * (1 << 29))
        # test.pack uses the integer FFT/Q15 domain; gnuradio_reference divides
        # that domain by Q15 and therefore receives the original complex64 data.
        candidate = samples[start:start + 4 * 64].astype(np.complex128) * test.Q15
        reference_hard, reference_symbols = test.gnuradio_reference(
            candidate, cfo, algorithm=args.algorithm)
        if len(reference_symbols) and np.max(np.abs(reference_symbols[0])) > 3.0:
            unusable += 1
            continue
        npu_hard, npu_symbols, _ = test.run_npu(
            handle, *test.pack([candidate], [0], cfo))
        if not len(reference_symbols):
            npu_only += bool(len(npu_symbols))
            continue
        if not len(npu_symbols):
            gnu_only += 1
            continue
        if not np.array_equal(reference_hard[0], npu_hard[0]):
            decision_disagreements += 1
            continue
        matched_reference.append(reference_symbols[0])
        matched_actual.append(npu_symbols[0])
        matched_evm.append(float(np.sqrt(
            np.mean(np.abs(npu_symbols[0] - reference_symbols[0]) ** 2) /
            np.mean(np.abs(reference_symbols[0]) ** 2))))
        gain = np.vdot(reference_symbols[0], npu_symbols[0]) / np.vdot(
            reference_symbols[0], reference_symbols[0])
        print(f"  offset={start} evm={100*matched_evm[-1]:.4f}% "
              f"gain={abs(gain):.5f} phase={np.angle(gain):.5f} "
              f"ref_max={np.max(np.abs(reference_symbols[0])):.3f} "
              f"npu_max={np.max(np.abs(npu_symbols[0])):.3f}")
    print(f"recorded candidates: samples={len(samples)} wifi_start={len(tags)} "
          f"matched={len(matched_actual)} GNU_only={gnu_only} "
          f"NPU_only={npu_only} hard_disagreements={decision_disagreements} "
          f"unusable_false_frames={unusable}")
    if not matched_actual:
        raise AssertionError("no mutually decoded recorded frame candidates")
    print(f"matched-frame EVM: median={100*np.median(matched_evm):.4f}% "
          f"min={100*np.min(matched_evm):.4f}% max={100*np.max(matched_evm):.4f}%")
    test.check_error("recorded FFT matched candidates",
                     np.asarray(matched_actual),
                     np.asarray(matched_reference), 5e-2)


if __name__ == "__main__":
    main()
