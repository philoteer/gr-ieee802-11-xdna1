# Standalone equalization: prepared samples + channel estimates -> decisions/tags.
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
# Matches frame_equalizer_estimated_tile_metadata: CSI + prepared + padding + SNR.
estimated_meta_words = 2168
prepared_meta = np.ndarray[(n_tiles * estimated_meta_words,), np.dtype[np.int32]]
prepared_tile_meta = np.ndarray[(estimated_meta_words,), np.dtype[np.int32]]
decisions = np.ndarray[(vector_size,), np.dtype[np.int8]]
decision_tile = np.ndarray[(tile_size,), np.dtype[np.int8]]
# The payload is 8968 bytes (2242 words). Pad its FIFO allocation to a 32-byte
# boundary: otherwise automatic placement puts the following int8 output at
# address % 16 == 8, and the kernel's aligned stores shift decisions by 8 bytes.
payload_meta_words = 2242
output_meta_words = 2248
output_meta = np.ndarray[(n_tiles * output_meta_words,), np.dtype[np.int32]]
output_tile_meta = np.ndarray[(output_meta_words,), np.dtype[np.int32]]
kernel = Kernel("frame_equalizer_equalize", "frame_equalizer_equalize.o",
                [tile, prepared_tile_meta, decision_tile, output_tile_meta, np.int32, np.int32])
input_fifo = ObjectFifo(tile, name="prepared")
input_meta_fifo = ObjectFifo(prepared_tile_meta, name="estimated_meta", depth=1)
output_fifo = ObjectFifo(decision_tile, name="out")
output_meta_fifo = ObjectFifo(output_tile_meta, name="out_meta")


def core(src, src_meta, dst, dst_meta, equalize):
    for _ in range_(n_tiles):
        a = src.acquire(1)
        b = src_meta.acquire(1)
        c = dst.acquire(1)
        d = dst_meta.acquire(1)
        equalize(a, b, c, d, tile_size, payload_meta_words)
        src.release(1)
        src_meta.release(1)
        dst.release(1)
        dst_meta.release(1)


worker = Worker(core, [input_fifo.cons(), input_meta_fifo.cons(), output_fifo.prod(),
                       output_meta_fifo.prod(), kernel], stack_size=2560)
if hasattr(Runtime, "sequence"):
    runtime = Runtime()
    with runtime.sequence(samples, prepared_meta, decisions, output_meta) as (a, b, c, d):
        runtime.start(worker)
        runtime.fill(input_fifo.prod(), a)
        runtime.fill(input_meta_fifo.prod(), b)
        runtime.drain(output_fifo.cons(), c, wait=False)
        runtime.drain(output_meta_fifo.cons(), d, wait=True)
else:
    def runtime_sequence(a, b, c, d, source, metadata, destination, output_metadata):
        source.fill(a)
        metadata.fill(b)
        destination.drain(c, wait=False)
        output_metadata.drain(d, wait=True)

    runtime = Runtime(runtime_sequence, [samples, prepared_meta, decisions, output_meta,
        input_fifo.prod(), input_meta_fifo.prod(), output_fifo.cons(), output_meta_fifo.cons()])
device = NPU2Col1() if len(sys.argv) > 1 and sys.argv[1] == "npu2" else NPU1Col1()
program = (Program(device, runtime) if hasattr(Runtime, "sequence")
           else Program(device, runtime, workers=[worker]))
print(program.resolve_program(SequentialPlacer())
      if SequentialPlacer is not None else program.resolve_program())
