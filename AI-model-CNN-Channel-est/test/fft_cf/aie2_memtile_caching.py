# -*- Python -*-
#
# Standalone fft_cf test graph for IRON.
# Target: XDNA1 / XDNA2 (npu1 / npu2)

"""Standalone IRON graph for the fft_cf kernel.

Feeds interleaved int16 tiles and their FFT metadata to fft_cf and drains the
int32 bin tiles with their output metadata.
"""

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
max_tags_per_tile = 31
metadata_words_per_tile = 2 + 2 * max_tags_per_tile

# Complex values are represented as interleaved real/imaginary components.
in_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int16]]
in_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int16]]
out_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int32]]
out_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]
meta_tensor_ty = np.ndarray[(divider * metadata_words_per_tile,), np.dtype[np.int32]]
meta_tile_ty = np.ndarray[(metadata_words_per_tile,), np.dtype[np.int32]]

fft_fn = Kernel(
    "fft_cf",
    "fft_cf.o",
    [in_tile_ty, meta_tile_ty, out_tile_ty, meta_tile_ty, np.int32, np.int32],
)

of_in = ObjectFifo(in_tile_ty, name="in")
of_in_meta = ObjectFifo(meta_tile_ty, name="in_meta")
of_out = ObjectFifo(out_tile_ty, name="out")
of_out_meta = ObjectFifo(meta_tile_ty, name="out_meta")


def fft_core_fn(of_in, of_in_meta, of_out, of_out_meta, fft_scalar):
    """Runs fft_cf over each tile, pairing data and metadata FIFOs.

    Args:
        of_in: Time-domain cint16 ObjectFifo consumer.
        of_in_meta: Input metadata ObjectFifo consumer.
        of_out: FFT bin ObjectFifo producer.
        of_out_meta: Output metadata ObjectFifo producer.
        fft_scalar: fft_cf kernel handle.
    """
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        fft_scalar(
            elem_in,
            elem_in_meta,
            elem_out,
            elem_out_meta,
            tile_size,
            metadata_words_per_tile,
        )

        of_in.release(1)
        of_in_meta.release(1)
        of_out.release(1)
        of_out_meta.release(1)


worker = Worker(
    fft_core_fn,
    [of_in.cons(), of_in_meta.cons(), of_out.prod(), of_out_meta.prod(), fft_fn],
)

rt = Runtime()
with rt.sequence(
    in_tensor_ty, meta_tensor_ty, out_tensor_ty, meta_tensor_ty
) as (a_in, b_in_meta, c_out, d_out_meta):
    rt.start(worker)
    rt.fill(of_in.prod(), a_in)
    rt.fill(of_in_meta.prod(), b_in_meta)
    rt.drain(of_out.cons(), c_out, wait=False)
    rt.drain(of_out_meta.cons(), d_out_meta, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
