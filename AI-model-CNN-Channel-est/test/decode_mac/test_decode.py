#!/usr/bin/env python3
"""Exercise the production MAC kernel on hardware with known OFDM DATA bits."""
import argparse
from pathlib import Path
import struct
import zlib

import numpy as np
import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel

BPSC = (1, 1, 2, 2, 4, 4, 6, 6)
DBPS = (24, 36, 48, 72, 96, 144, 192, 216)
INPUT_OFFSET = 0


def encode(payload, encoding, seed=93):
    psdu = payload + struct.pack("<I", zlib.crc32(payload))
    dbps, bpsc = DBPS[encoding], BPSC[encoding]
    nsym = (16 + 8 * len(psdu) + 6 + dbps - 1) // dbps
    bits = np.zeros(nsym * dbps, dtype=np.uint8)
    bits[16:16 + 8 * len(psdu)] = np.unpackbits(
        np.frombuffer(psdu, dtype=np.uint8), bitorder="little")
    for i in range(len(bits)):
        feedback = ((seed >> 6) ^ (seed >> 3)) & 1
        bits[i] ^= feedback
        seed = ((seed << 1) & 126) | feedback
    bits[16 + 8 * len(psdu):16 + 8 * len(psdu) + 6] = 0
    coded = []
    state = 0
    for bit in bits:
        state = ((state << 1) | int(bit)) & 127
        coded.extend(((state & 0o155).bit_count() & 1,
                      (state & 0o117).bit_count() & 1))
    pattern = ((1, 1) if encoding in (0, 2, 4) else
               (1, 1, 1, 0) if encoding == 6 else (1, 1, 1, 0, 0, 1))
    coded = np.array([v for i, v in enumerate(coded) if pattern[i % len(pattern)]],
                     dtype=np.uint8).reshape(nsym, 48 * bpsc)
    packed = np.zeros((nsym, 48), dtype=np.uint8)
    s = max(bpsc // 2, 1)
    for k in range(48 * bpsc):
        i = (48 * bpsc // 16) * (k % 16) + k // 16
        j = s * (i // s) + (i + 48 * bpsc - (16 * i) // (48 * bpsc)) % s
        packed[:, j // bpsc] |= coded[:, k] << (j % bpsc)
    return packed, len(psdu)


def run_frame(handle, payload, encoding, seed=93, errors=False):
    symbols, length = encode(payload, encoding, seed)
    if errors:
        # Sparse hard-decision errors must be corrected by the trellis.
        symbols.reshape(-1)[37::503] ^= 1
    result = []
    for start in range(0, len(symbols), 21):
        chunk = symbols[start:start + 21]
        data = np.full(1024 + INPUT_OFFSET, 0x55, dtype=np.int8)
        data[INPUT_OFFSET:INPUT_OFFSET + chunk.size] = chunk.reshape(-1)
        meta = np.zeros(2242, dtype=np.int32)
        meta[0] = chunk.size
        if start == 0:
            meta[1] = 1
            meta[3] = 1
            meta[4] = length
            meta[6] = encoding
        output = iron.zeros(8192, dtype=np.int8)
        output_meta = iron.zeros(2274, dtype=np.int32)
        DefaultNPURuntime.run(handle, [
            iron.tensor(data, dtype=np.int8), iron.tensor(meta, dtype=np.int32),
            output, output_meta])
        om = output_meta.numpy()
        raw = output.numpy().view(np.uint8)
        for index in range(int(om[1])):
            offset, count = om[2 + index * 142:4 + index * 142]
            result.append(raw[offset:offset + count].tobytes())
    assert result == [payload], (
        f"encoding={encoding}, length={len(payload)}, seed={seed}, "
        f"input_offset={INPUT_OFFSET}, errors={errors}: "
        f"expected one exact frame, got {[len(x) for x in result]}")


def main():
    global INPUT_OFFSET
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplementary", action="store_true")
    args = parser.parse_args()
    build = Path(__file__).resolve().parent / "build"
    for line in (build / ".device").read_text().splitlines():
        if line.startswith("INPUT_OFFSET="):
            INPUT_OFFSET = int(line.split("=")[1])
    handle = DefaultNPURuntime.load(NPUKernel(
        str(build / "final.xclbin"), str(build / "insts.bin"),
        kernel_name="MLIR_AIE"))
    rng = np.random.default_rng(80211)
    for encoding in range(8):
        for length in ((0, 10, 16, 28, 257, 1500, 4091) if args.supplementary else (10, 28, 257)):
            payload = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
            run_frame(handle, payload, encoding)
            run_frame(handle, payload, encoding, errors=True)
            print(f"PASS encoding={encoding} length={length}", flush=True)


if __name__ == "__main__":
    main()
