#!/usr/bin/env python3
"""Compare NPU equalized-symbol metadata with GNU Radio's symbols PDUs.

IMPORTANT BUILD MODE: this test compares the NPU's *normalized equalized
symbols* against GNU Radio. The NPU kernel only emits those symbols when built
with ``DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1`` (the "symbols" build). The default
production build (CSI mode, flag 0) instead emits the channel estimate per
frame, which is a different quantity; comparing it here yields a meaningless
multi-hundred-percent EVM. ``require_symbols_mode()`` refuses a mismatched build
by reading ``build/frame_equalizer_build_mode.txt``, which the Makefile writes
during the build. Use ``make test EQUALIZER_ALGORITHM=LS`` (or ``LMS``/``COMB``/
``STA``) to build and test one statically selected kernel algorithm.
"""

import argparse
import numpy as np
import pmt
import os
import sys
import tempfile
from pathlib import Path
import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel
from gnuradio import blocks, gr
import ieee802_11

TILE = 1024
TILES = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
N = TILE * TILES
META_WORDS = 64
MAX_TAGS = 16
Q15 = float(1 << 15)
LONG = np.array([
    0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,
    1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,
    0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,
    1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0], dtype=np.int32)
POLARITY = [
    1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,
    -1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,1,
    1,1,-1,1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,
    -1,1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,
    -1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,
    -1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,
    -1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1,-1,
    -1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1]
PILOTS = (11, 25, 39, 53)
DATA = [i for i in range(6, 59) if i != 32 and i not in PILOTS]
ALGORITHMS = {"ls": 0, "lms": 1, "comb": 2, "sta": 3}

CSI = np.dtype([("real", np.int32), ("imag", np.int32)], align=True)
TAG = np.dtype([
    ("offset", np.int32), ("reserved", np.uint32),
    ("frame_bytes", np.uint64), ("encoding", np.uint64),
    ("snr_linear", np.uint64), ("center_frequency_mhz", np.uint32),
    ("frequency_offset", np.int32), ("beta", np.int32),
    ("csi", CSI, (64,))], align=True)
TILE_META = np.dtype([
    ("output_length", np.int32), ("tag_count", np.int32),
    ("tags", TAG, (MAX_TAGS,))], align=True)
OUT_META_WORDS = TILES * TILE_META.itemsize // 4


def signal(rate, length):
    bits = np.zeros(24, dtype=np.uint8)
    bits[:4] = [(rate >> i) & 1 for i in range(4)]
    bits[5:17] = [(length >> i) & 1 for i in range(12)]
    bits[17] = np.bitwise_xor.reduce(bits[:17])
    coded = np.empty(48, dtype=np.uint8)
    state = 0
    for i, bit in enumerate(bits):
        reg = ((state << 1) | int(bit)) & 0x7f
        coded[2*i] = (reg & 0o155).bit_count() & 1
        coded[2*i+1] = (reg & 0o117).bit_count() & 1
        state = reg & 63
    pattern = np.array([3*(i & 15) + (i >> 4) for i in range(48)])
    interleaved = np.empty(48, dtype=np.uint8)
    interleaved[pattern] = coded
    return interleaved


def points(modulation):
    if modulation == 2:
        return np.array([-1, 1])
    if modulation == 4:
        q = 1 / np.sqrt(2)
        return np.array([-q-1j*q, q-1j*q, -q+1j*q, q+1j*q])
    if modulation == 16:
        levels = np.array([-3, 3, -1, 1]) / np.sqrt(10)
    else:
        levels = np.array([-7, 7, -1, 1, -5, 5, -3, 3]) / np.sqrt(42)
    return np.array([x + 1j*y for y in levels for x in levels])


def symbol(decisions, modulation, channel, number):
    result = np.zeros(64, dtype=np.complex128)
    result[DATA] = channel[DATA] * points(modulation)[decisions]
    p = POLARITY[(number - 2) % 127]
    result[list(PILOTS)] = channel[list(PILOTS)] * np.array([p, p, p, -p])
    return result


def frame(rate, modulation, channel, length, decisions):
    payload = [symbol(x, modulation, channel, i + 3)
               for i, x in enumerate(decisions)]
    return np.stack((LONG * channel, LONG * channel,
                     symbol(signal(rate, length), 2, channel, 2), *payload))


def algorithm_name(value):
    name = value.lower()
    if name not in ALGORITHMS:
        raise argparse.ArgumentTypeError(
            f"unknown equalizer {value!r}; choose from {', '.join(ALGORITHMS)}")
    return name


def add_algorithm_argument(parser):
    parser.add_argument(
        "--algorithm", type=algorithm_name, default="ls",
        help="equalizer algorithm compiled into the NPU image (default: ls)")


def gnuradio_reference(samples, cfo=0, algorithm="ls"):
    tag = gr.tag_t()
    tag.offset, tag.key = 0, pmt.intern("wifi_start")
    tag.value, tag.srcid = pmt.from_double(cfo / (1 << 29)), pmt.PMT_NIL
    fg = gr.top_block()
    source = blocks.vector_source_c(
        (samples.astype(np.complex64) / Q15).reshape(-1).tolist(), False, 64, [tag])
    equalizer = ieee802_11.frame_equalizer(
        ieee802_11.Equalizer(ALGORITHMS[algorithm]),
        2.417e9, 20e6, False, False)
    hard = blocks.vector_sink_b(48)
    messages = blocks.message_debug()
    fg.connect(source, equalizer, hard)
    fg.msg_connect(equalizer, "symbols", messages, "store")
    fg.run()
    symbols = [np.asarray(pmt.c32vector_elements(
        pmt.cdr(messages.get_message(i))), dtype=np.complex64)
        for i in range(messages.num_messages())]
    return np.asarray(hard.data(), dtype=np.int8).reshape(-1, 48), np.asarray(symbols)


def pack(frames, starts, cfo=0):
    samples = np.zeros(N, dtype=np.complex128)
    metadata = np.zeros((TILES, META_WORDS), dtype=np.int32)
    metadata[:, 0] = TILE
    for start, value in zip(starts, frames):
        samples[start:start + value.size] = value.reshape(-1)
        tile = start // TILE
        index = int(metadata[tile, 1])
        metadata[tile, 1] += 1
        metadata[tile, 2 + 2*index:4 + 2*index] = (start % TILE, cfo)
    components = np.empty(2*N, dtype=np.int32)
    components[0::2] = np.rint(samples.real).astype(np.int32)
    components[1::2] = np.rint(samples.imag).astype(np.int32)
    return components, metadata.reshape(-1)


def run_npu(handle, components, metadata):
    output = iron.zeros(N, dtype=np.int8)
    output_meta = iron.zeros(OUT_META_WORDS, dtype=np.int32)
    DefaultNPURuntime.run(handle, [iron.tensor(components, dtype=np.int32),
        iron.tensor(metadata, dtype=np.int32), output, output_meta])
    parsed = output_meta.numpy().view(TILE_META).reshape(TILES)
    hard, symbols, encodings = [], [], []
    raw = output.numpy()
    for tile_index, item in enumerate(parsed):
        length = int(item["output_length"])
        hard.extend(raw[tile_index*TILE:tile_index*TILE + length].reshape(-1, 48))
        for tag in item["tags"][:int(item["tag_count"])]:
            symbols.append(tag["csi"]["real"][:48] / Q15
                           + 1j * tag["csi"]["imag"][:48] / Q15)
            encodings.append(int(tag["encoding"]))
    return np.asarray(hard), np.asarray(symbols), encodings


def check_error(label, actual, reference, limit):
    error = actual - reference
    evm = np.sqrt(np.mean(abs(error)**2) / np.mean(abs(reference)**2))
    gain = np.vdot(reference.ravel(), actual.ravel()) / np.vdot(reference.ravel(), reference.ravel())
    print(f"{label}: rms_evm={100*evm:.5f}% max_abs={np.max(abs(error)):.7f} "
          f"gain={abs(gain):.8f} phase={np.angle(gain):.8g} rad")
    assert evm < limit, (label, evm)


def require_symbols_mode(algorithm="ls"):
    """Require normalized-symbol output and the requested static equalizer.

    The symbol-comparison tests below only make sense with
    ``DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1``. The build writes the active flags to
    ``build/frame_equalizer_build_mode.txt``; if it is missing or shows the CSI
    mode, stop with a clear instruction instead of producing garbage EVM.
    """
    marker = Path(__file__).parent / "build" / "frame_equalizer_build_mode.txt"
    if not marker.exists():
        raise SystemExit(
            "Build mode marker not found. The symbol-comparison tests need the "
            "NPU built in symbols mode. Rebuild with "
            "`make DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1` (or just `make test`).")
    mode = {}
    for line in marker.read_text().splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            mode[key.strip()] = val.strip()
    if mode.get("DFRAME_EQUALIZER_OUTPUT_SYMBOLS") != "1":
        raise SystemExit(
            "NPU built in CSI mode "
            f"(DFRAME_EQUALIZER_OUTPUT_SYMBOLS="
            f"{mode.get('DFRAME_EQUALIZER_OUTPUT_SYMBOLS')}), but this test "
            "compares normalized equalized symbols, which only the symbols "
            "build (DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1) emits. Rebuild with "
            "`make DFRAME_EQUALIZER_OUTPUT_SYMBOLS=1`.")
    built_algorithm = mode.get("EQUALIZER_ALGORITHM", "").lower()
    if built_algorithm != algorithm:
        raise SystemExit(
            f"NPU was built for equalizer {built_algorithm or 'unknown'}, but "
            f"the test requested {algorithm}. Rebuild with "
            f"`make test EQUALIZER_ALGORITHM={algorithm.upper()}`.")


def load_kernel():
    instructions = np.fromfile("build/insts.bin", dtype=np.uint32).copy()
    for marker, value in ((0x13579BDF, 2417),
                          (0x2468ACE0, round(20 / 2417 * (1 << 30)))):
        matches = np.flatnonzero(instructions == marker)
        assert len(matches) == 1, (hex(marker), matches)
        instructions[matches[0]] = value
    rtp_dir = tempfile.TemporaryDirectory()
    patched_insts = os.path.join(rtp_dir.name, "insts.bin")
    instructions.tofile(patched_insts)
    handle = DefaultNPURuntime.load(NPUKernel(
        "build/final.xclbin", patched_insts, kernel_name="MLIR_AIE"))
    return handle, rtp_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_algorithm_argument(parser)
    args = parser.parse_args()
    algorithm = args.algorithm
    require_symbols_mode(algorithm)
    assert TAG.itemsize == 560 and TILE_META.itemsize == 8968
    handle, rtp_dir = load_kernel()
    rng = np.random.default_rng(80211)
    rates = [(11,2), (15,2), (10,4), (14,4), (9,16), (13,16), (8,64), (12,64)]
    starts = [0,256,512,896,1344,1792,2304,3072]
    channel = np.zeros(64, dtype=np.complex128)
    channel[6:59], channel[32] = 16384, 0
    frames, refs = [], []
    for rate, modulation in rates:
        value = frame(rate, modulation, channel, 0,
                      rng.integers(0, modulation, (1,48), dtype=np.uint8))
        frames.append(value)
        refs.append(gnuradio_reference(value, algorithm=algorithm))
    hard, symbols, encodings = run_npu(handle, *pack(frames, starts))
    assert encodings == list(range(8))
    np.testing.assert_array_equal(hard, np.concatenate([x[0] for x in refs]))
    check_error("all rates, flat channel", symbols,
                np.concatenate([x[1] for x in refs]), 5e-4)

    decisions = rng.integers(0, 16, (8,48), dtype=np.uint8)
    channel[:] = 0
    occupied = [i for i in range(6,59) if i != 32]
    channel[occupied] = np.linspace(9000,21000,len(occupied)) * np.exp(
        1j*np.linspace(-0.45,0.35,len(occupied)))
    value = frame(13, 16, channel, 130, decisions)
    cfo = round(0.002 * (1 << 29))
    bins = np.arange(64) - 32
    for number in range(len(value)):
        ramp = (cfo / (1 << 29)) * 20 / 2417 * number * 80 * bins / 64
        value[number] *= np.exp(1j*(0.08 + 0.012*number - ramp))
    hard, symbols, _ = run_npu(handle, *pack([value], [0], cfo))
    ref_hard, ref_symbols = gnuradio_reference(
        value, cfo, algorithm=algorithm)
    np.testing.assert_array_equal(hard, ref_hard)
    check_error("frequency-selective tracked frame", symbols, ref_symbols, 2e-3)

    # A terminated K=7 code corrects this three-coded-bit SIGNAL error when the
    # known zero start state is enforced. The previous all-zero metric
    # initialization selected an invalid RATE path and dropped the frame.
    payload = rng.integers(0, 2, (1, 48), dtype=np.uint8)
    channel[:] = 0
    channel[6:59], channel[32] = 16384, 0
    value = frame(11, 2, channel, 0, payload)
    damaged_signal = signal(11, 0)
    damaged_signal[[0, 4, 10]] ^= 1
    value[2] = symbol(damaged_signal, 2, channel, 2)
    hard, symbols, encodings = run_npu(handle, *pack([value], [0]))
    np.testing.assert_array_equal(hard[0], payload[0])
    assert encodings[0] == 0
    print(f"frame_equalizer {algorithm} symbol comparison passed")


if __name__ == "__main__":
    main()
