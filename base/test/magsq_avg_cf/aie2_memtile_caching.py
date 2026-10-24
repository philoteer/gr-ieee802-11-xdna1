# -*- Python -*-
#
# Standalone magsq_avg_cf test graph for IRON.
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
# magsq_avg_cf emits the boxcar sum as Q8.23 int32.
out_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
out_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
# Delayed complex cint16 data uses interleaved real/imag components.
delay_out_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int16]]
delay_out_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int16]]

magsq_fn = Kernel(
    "magsq_avg_cf",
    "magsq_avg_cf.o",
    [in_tile_ty, out_tile_ty, delay_out_tile_ty, np.int32],
)

of_in = ObjectFifo(in_tile_ty, name="in")
of_out = ObjectFifo(out_tile_ty, name="out")
of_out_delay = ObjectFifo(delay_out_tile_ty, name="out_delay")


def magsq_core_fn(of_in, of_out, of_out_delay, magsq_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_delay = of_out_delay.acquire(1)

        magsq_scalar(elem_in, elem_out, elem_out_delay, tile_size)

        of_in.release(1)
        of_out.release(1)
        of_out_delay.release(1)


worker = Worker(magsq_core_fn, [of_in.cons(), of_out.prod(), of_out_delay.prod(), magsq_fn])

rt = Runtime()
with rt.sequence(in_tensor_ty, out_tensor_ty, delay_out_tensor_ty) as (a_in, c_out, d_out):
    rt.start(worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=False)
    rt.drain(of_out_delay.cons(), d_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
