# -*- Python -*-
#
# Standalone autocorr_cf test graph for IRON.
# Target: XDNA1 / XDNA2 (npu1 / npu2)

import numpy as np
import ml_dtypes
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

# Complex cint16 data is represented as interleaved real/imag components.
in_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int16]]
in_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int16]]
# The output is complex Q7.24 with interleaved int32 components.
out_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int32]]
out_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]

autocorr_fn = Kernel(
    "autocorr_cf",
    "autocorr_cf.o",
    [in_tile_ty, out_tile_ty, np.int32],
)

of_in = ObjectFifo(in_tile_ty, name="in")
of_out = ObjectFifo(out_tile_ty, name="out")


def autocorr_core_fn(of_in, of_out, autocorr_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)

        autocorr_scalar(elem_in, elem_out, tile_size)

        of_in.release(1)
        of_out.release(1)


worker = Worker(autocorr_core_fn, [of_in.cons(), of_out.prod(), autocorr_fn])

rt = Runtime()
with rt.sequence(in_tensor_ty, out_tensor_ty) as (a_in, c_out):
    rt.start(worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
