# -*- Python -*-
# Standalone trig-estimator accuracy graph for XDNA1 / XDNA2.

import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
try:
    from aie.iron.placers import SequentialPlacer
except ModuleNotFoundError as exc:
    if exc.name != "aie.iron.placers":
        raise
    SequentialPlacer = None
import aie.iron as iron

if len(sys.argv) > 1:
    from aie.iron.device import NPU1, NPU2

    dev = {"npu": NPU1, "npu1": NPU1, "npu2": NPU2}[sys.argv[1]]()
else:
    dev = iron.get_current_device()

sample_count = 4096
tile_count = 16
tile_size = sample_count // tile_count
# Each input tile holds phase, y, and x. Outputs contain minimax and LUT
# sine/cosine, both atan2 implementations, and one cycle count for each path.
in_tensor_ty = np.ndarray[(3 * sample_count,), np.dtype[np.int32]]
in_tile_ty = np.ndarray[(3 * tile_size,), np.dtype[np.int32]]
out_tensor_ty = np.ndarray[(10 * sample_count,), np.dtype[np.int32]]
out_tile_ty = np.ndarray[(10 * tile_size,), np.dtype[np.int32]]

trig_utils_fn = Kernel(
    "trig_utils_test",
    "trig_utils_test.o",
    [in_tile_ty, out_tile_ty, np.int32],
)

of_in = ObjectFifo(in_tile_ty, name="trig_in")
of_out = ObjectFifo(out_tile_ty, name="trig_out")


def trig_core_fn(of_in, of_out, trig_utils_scalar):
    for _ in range_(tile_count):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        trig_utils_scalar(elem_in, elem_out, tile_size)
        of_in.release(1)
        of_out.release(1)


# The kernel's local LUT state plus nested trig calls exceed the default 1 KiB
# stack on AIE2P. Match the production preprocess worker's stack reservation.
worker = Worker(trig_core_fn, [of_in.cons(), of_out.prod(), trig_utils_fn],
                stack_size=2560)

if hasattr(Runtime, "sequence"):
    rt = Runtime()
    with rt.sequence(in_tensor_ty, out_tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)
    program = Program(dev, rt)
else:
    def runtime_sequence(a_in, b_out, input_fifo, output_fifo):
        input_fifo.fill(a_in)
        output_fifo.drain(b_out, wait=True)

    rt = Runtime(runtime_sequence, [
        in_tensor_ty, out_tensor_ty, of_in.prod(), of_out.cons(),
    ])
    program = Program(dev, rt, workers=[worker])

module = (program.resolve_program(SequentialPlacer())
          if SequentialPlacer is not None else program.resolve_program())
print(module)
