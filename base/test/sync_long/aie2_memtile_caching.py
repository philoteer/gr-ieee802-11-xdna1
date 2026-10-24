# -*- Python -*-
#
# Standalone sync_long_cf test graph for IRON.
# Target: XDNA1 / XDNA2 (npu1 / npu2)

import numpy as np
import os

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
import aie.iron as iron

dev = iron.get_current_device()

tile_size = 1024
divider = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
if divider < 1:
    raise ValueError("AIE_TILES_PER_BATCH must be positive")
tensor_size = tile_size * divider
delay_samples = 320
max_tags_per_tile = 7
metadata_words_per_tile = 2 + 2 * max_tags_per_tile

# sync_long_cf consumes the enlarged sync_short buffer: delayed prefix + current stream.
in_tile_words = (tile_size + delay_samples) * 2
in_tensor_words = in_tile_words * divider
in_tensor_ty = np.ndarray[(in_tensor_words,), np.dtype[np.int16]]
in_tile_ty = np.ndarray[(in_tile_words,), np.dtype[np.int16]]

# sync_long_cf currently emits the normal current-stream-sized output.
out_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int16]]
out_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int16]]

meta_tensor_ty = np.ndarray[(divider * metadata_words_per_tile,), np.dtype[np.int32]]
meta_tile_ty = np.ndarray[(metadata_words_per_tile,), np.dtype[np.int32]]

sync_long_fn = Kernel(
    "sync_long_cf",
    "sync_long_cf.o",
    [in_tile_ty, meta_tile_ty, out_tile_ty, meta_tile_ty, np.int32, np.int32],
)

of_in = ObjectFifo(in_tile_ty, name="in")
of_in_meta = ObjectFifo(meta_tile_ty, name="in_meta")
of_out = ObjectFifo(out_tile_ty, name="out")
of_out_meta = ObjectFifo(meta_tile_ty, name="out_meta")


def sync_long_core_fn(of_in, of_in_meta, of_out, of_out_meta, sync_long_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        sync_long_scalar(elem_in, elem_in_meta, elem_out, elem_out_meta, tile_size, metadata_words_per_tile)

        of_in.release(1)
        of_in_meta.release(1)
        of_out.release(1)
        of_out_meta.release(1)


worker = Worker(sync_long_core_fn, [of_in.cons(), of_in_meta.cons(), of_out.prod(), of_out_meta.prod(), sync_long_fn], stack_size=2048)

rt = Runtime()
with rt.sequence(in_tensor_ty, meta_tensor_ty, out_tensor_ty, meta_tensor_ty) as (a_in, b_in_meta, c_out, d_out_meta):
    rt.start(worker)
    rt.fill(of_in.prod(), a_in)
    rt.fill(of_in_meta.prod(), b_in_meta)
    rt.drain(of_out.cons(), c_out, wait=False)
    rt.drain(of_out_meta.cons(), d_out_meta, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
