# -*- Python -*-
# Standalone sync_short_cf test graph for XDNA1 / XDNA2.

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer
import aie.iron as iron

dev = iron.get_current_device()

tensor_size = 4096
tile_size = tensor_size // 4
delay_samples = 320
metadata_words = 16

# Pack all three logical inputs into one DMA stream. In int32 words each tile
# contains cint16[tile_size], cint32[tile_size], then int8[tile_size].
threshold_words = (tile_size + 3) // 4
in_tile_words = 3 * tile_size + threshold_words
in_tensor_ty = np.ndarray[(4 * in_tile_words,), np.dtype[np.int32]]
in_tile_ty = np.ndarray[(in_tile_words,), np.dtype[np.int32]]

# Pack the variable-length data area and its metadata into one output stream.
out_tile_words = tile_size + delay_samples + metadata_words
out_tensor_ty = np.ndarray[(4 * out_tile_words,), np.dtype[np.int32]]
out_tile_ty = np.ndarray[(out_tile_words,), np.dtype[np.int32]]

sync_short_fn = Kernel(
    "sync_short_test",
    "sync_short_test.o",
    [in_tile_ty, out_tile_ty, np.int32],
)

of_in = ObjectFifo(in_tile_ty, name="packed_in")
of_out = ObjectFifo(out_tile_ty, name="packed_out")


def sync_short_core_fn(of_in, of_out, sync_short_scalar):
    for _ in range_(4):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        sync_short_scalar(elem_in, elem_out, tile_size)
        of_in.release(1)
        of_out.release(1)


worker = Worker(sync_short_core_fn, [of_in.cons(), of_out.prod(), sync_short_fn], stack_size=2048)

rt = Runtime()
with rt.sequence(in_tensor_ty, out_tensor_ty) as (a_in, b_out):
    rt.start(worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
