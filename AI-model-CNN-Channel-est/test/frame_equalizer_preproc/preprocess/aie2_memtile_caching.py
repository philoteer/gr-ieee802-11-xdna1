# Standalone frame_equalizer_preprocess: FFT samples/metadata -> prepared samples/metadata.
import os
import sys
import numpy as np
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1

tile_size = 1024
n_tiles = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
if n_tiles < 1:
    raise ValueError("AIE_TILES_PER_BATCH must be positive")
vector_size = tile_size * n_tiles
samples = np.ndarray[(vector_size * 2,), np.dtype[np.int32]]
tile = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]
fft_meta = np.ndarray[(n_tiles * 64,), np.dtype[np.int32]]
fft_tile_meta = np.ndarray[(64,), np.dtype[np.int32]]
prepared_meta = np.ndarray[(n_tiles * 81,), np.dtype[np.int32]]
prepared_tile_meta = np.ndarray[(81,), np.dtype[np.int32]]
frequency_type = np.ndarray[(2,), np.dtype[np.int32]]
frequency = Buffer(frequency_type, name="center_frequency", use_write_rtp=True)


def write_frequency(buffer):
    buffer[0] = 0x13579BDF
    buffer[1] = 0x2468ACE0


kernel = Kernel("frame_equalizer_preprocess", "frame_equalizer_preprocess.o",
                [tile, fft_tile_meta, tile, prepared_tile_meta, frequency_type, np.int32, np.int32])
input_fifo = ObjectFifo(tile, name="in")
input_meta_fifo = ObjectFifo(fft_tile_meta, name="in_meta")
output_fifo = ObjectFifo(tile, name="prepared")
output_meta_fifo = ObjectFifo(prepared_tile_meta, name="prepared_meta")


def core(src, src_meta, dst, dst_meta, center_frequency, preprocess):
    for _ in range_(n_tiles):
        a = src.acquire(1)
        b = src_meta.acquire(1)
        c = dst.acquire(1)
        d = dst_meta.acquire(1)
        preprocess(a, b, c, d, center_frequency, tile_size, 64)
        src.release(1)
        src_meta.release(1)
        dst.release(1)
        dst_meta.release(1)


worker = Worker(core, [input_fifo.cons(), input_meta_fifo.cons(), output_fifo.prod(),
                       output_meta_fifo.prod(), frequency, kernel], stack_size=2560)
runtime = Runtime()
with runtime.sequence(samples, fft_meta, samples, prepared_meta) as (a, b, c, d):
    runtime.inline_ops(write_frequency, [frequency])
    runtime.start(worker)
    runtime.fill(input_fifo.prod(), a)
    runtime.fill(input_meta_fifo.prod(), b)
    runtime.drain(output_fifo.cons(), c, wait=False)
    runtime.drain(output_meta_fifo.cons(), d, wait=True)
# Reserve one column so independently loaded, stateful stages can coexist.
device = NPU2Col1() if len(sys.argv) > 1 and sys.argv[1] == "npu2" else NPU1Col1()
print(Program(device, runtime).resolve_program(SequentialPlacer()))
