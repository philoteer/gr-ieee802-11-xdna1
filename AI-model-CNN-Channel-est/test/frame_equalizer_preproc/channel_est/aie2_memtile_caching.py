"""Standalone channel estimation: prepared samples/metadata -> CSI/SNR metadata."""
import os
import sys
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
try:
    from aie.iron.placers import SequentialPlacer
except ModuleNotFoundError as exc:
    if exc.name != "aie.iron.placers":
        raise
    SequentialPlacer = None
from aie.iron.device import NPU1Col1, NPU2Col1

tile_size = 1024
n_tiles = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
if n_tiles < 1:
    raise ValueError("AIE_TILES_PER_BATCH must be positive")
vector_size = tile_size * n_tiles
samples = np.ndarray[(vector_size * 2,), np.dtype[np.int32]]
tile = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]
prepared_meta = np.ndarray[(n_tiles * 81,), np.dtype[np.int32]]
prepared_tile_meta = np.ndarray[(81,), np.dtype[np.int32]]
# CSI for 16 symbols, copied prepared metadata, alignment padding, and 16 SNRs.
estimated_meta_words = 2168
estimated_meta = np.ndarray[(n_tiles * estimated_meta_words,), np.dtype[np.int32]]
estimated_tile_meta = np.ndarray[(estimated_meta_words,), np.dtype[np.int32]]

kernel = Kernel("frame_equalizer_channel_estimate", "frame_equalizer_channel_estimate.o",
                [tile, prepared_tile_meta, estimated_tile_meta])
input_fifo = ObjectFifo(tile, name="prepared")
input_meta_fifo = ObjectFifo(prepared_tile_meta, name="prepared_meta")
output_meta_fifo = ObjectFifo(estimated_tile_meta, name="estimated_meta", depth=1)


def core(src, src_meta, dst_meta, estimate):
    for _ in range_(n_tiles):
        a = src.acquire(1)
        b = src_meta.acquire(1)
        c = dst_meta.acquire(1)
        estimate(a, b, c)
        src.release(1)
        src_meta.release(1)
        dst_meta.release(1)


worker = Worker(core, [input_fifo.cons(), input_meta_fifo.cons(),
                       output_meta_fifo.prod(), kernel], stack_size=2560)
if hasattr(Runtime, "sequence"):
    runtime = Runtime()
    with runtime.sequence(samples, prepared_meta, estimated_meta) as (a, b, c):
        runtime.start(worker)
        runtime.fill(input_fifo.prod(), a)
        runtime.fill(input_meta_fifo.prod(), b)
        runtime.drain(output_meta_fifo.cons(), c, wait=True)
else:
    def runtime_sequence(a, b, c, source, metadata, destination):
        source.fill(a)
        metadata.fill(b)
        destination.drain(c, wait=True)

    runtime = Runtime(runtime_sequence, [samples, prepared_meta, estimated_meta,
        input_fifo.prod(), input_meta_fifo.prod(), output_meta_fifo.cons()])
device = NPU2Col1() if len(sys.argv) > 1 and sys.argv[1] == "npu2" else NPU1Col1()
program = (Program(device, runtime) if hasattr(Runtime, "sequence")
           else Program(device, runtime, workers=[worker]))
print(program.resolve_program(SequentialPlacer())
      if SequentialPlacer is not None else program.resolve_program())
