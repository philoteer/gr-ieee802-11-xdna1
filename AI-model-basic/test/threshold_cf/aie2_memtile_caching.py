# -*- Python -*-
#
# Standalone threshold_cf test graph for IRON.
# Target: XDNA1 / XDNA2 (npu1 / npu2)

import numpy as np
import sys
import os
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
try:
    from aie.iron.placers import SequentialPlacer
except ModuleNotFoundError as exc:
    if exc.name != "aie.iron.placers":
        raise
    SequentialPlacer = None
from aie.iron.controlflow import range_
import aie.iron as iron

if len(sys.argv) > 1:
    from aie.iron.device import NPU1, NPU2

    dev = {"npu": NPU1, "npu1": NPU1, "npu2": NPU2}[sys.argv[1]]()
else:
    dev = iron.get_current_device()

tile_size = 1024
divider = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
if divider < 1:
    raise ValueError("AIE_TILES_PER_BATCH must be positive")
tensor_size = tile_size * divider

# Autocorrelation input is complex Q7.24 with interleaved int32 components.
ac_in_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int32]]
ac_in_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]
# Magnitude input is the Q8.23 int32 boxcar sum from magsq_avg_cf.
mag_in_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
mag_in_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
out_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
out_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int8]]

threshold_fn = Kernel(
    "threshold_cf",
    "threshold_cf.o",
    [ac_in_tile_ty, mag_in_tile_ty, out_tile_ty, np.int32],
)

of_ac_in = ObjectFifo(ac_in_tile_ty, name="ac_in")
of_mag_in = ObjectFifo(mag_in_tile_ty, name="mag_in")
of_out = ObjectFifo(out_tile_ty, name="out")


def threshold_core_fn(of_ac_in, of_mag_in, of_out, threshold_scalar):
    for _ in range_(divider):
        elem_ac = of_ac_in.acquire(1)
        elem_mag = of_mag_in.acquire(1)
        elem_out = of_out.acquire(1)

        threshold_scalar(elem_ac, elem_mag, elem_out, tile_size)

        of_ac_in.release(1)
        of_mag_in.release(1)
        of_out.release(1)


worker = Worker(threshold_core_fn, [of_ac_in.cons(), of_mag_in.cons(), of_out.prod(), threshold_fn])

if hasattr(Runtime, "sequence"):
    rt = Runtime()
    with rt.sequence(ac_in_tensor_ty, mag_in_tensor_ty, out_tensor_ty) as (a_in, b_in, c_out):
        rt.start(worker)
        rt.fill(of_ac_in.prod(), a_in)
        rt.fill(of_mag_in.prod(), b_in)
        rt.drain(of_out.cons(), c_out, wait=True)
    my_program = Program(dev, rt)
else:
    def runtime_sequence(a_in, b_in, c_out, ac_fifo, mag_fifo, output_fifo):
        ac_fifo.fill(a_in)
        mag_fifo.fill(b_in)
        output_fifo.drain(c_out, wait=True)

    rt = Runtime(runtime_sequence, [
        ac_in_tensor_ty, mag_in_tensor_ty, out_tensor_ty,
        of_ac_in.prod(), of_mag_in.prod(), of_out.cons(),
    ])
    my_program = Program(dev, rt, workers=[worker])

module = (my_program.resolve_program(SequentialPlacer())
          if SequentialPlacer is not None else my_program.resolve_program())
print(module)
