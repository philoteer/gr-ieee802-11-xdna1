# -*- Python -*-
# Standalone trig-estimator accuracy graph for XDNA1 / XDNA2.

"""Standalone accuracy graph for the aie_trig_utils.h estimators.

Feeds packed phase/y/x tiles through trig_utils_test, which writes minimax
and linear-LUT sine/cosine, both atan2 implementations, and per-path cycle
counts for test_accuracy.py.
"""

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer
import aie.iron as iron

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
    """Runs trig_utils_test over each input tile.

    Args:
        of_in: Packed phase/y/x ObjectFifo consumer.
        of_out: Packed estimator output ObjectFifo producer.
        trig_utils_scalar: trig_utils_test kernel handle.
    """
    for _ in range_(tile_count):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        trig_utils_scalar(elem_in, elem_out, tile_size)
        of_in.release(1)
        of_out.release(1)


worker = Worker(trig_core_fn, [of_in.cons(), of_out.prod(), trig_utils_fn])

rt = Runtime()
with rt.sequence(in_tensor_ty, out_tensor_ty) as (a_in, b_out):
    rt.start(worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

program = Program(dev, rt)
module = program.resolve_program(SequentialPlacer())
print(module)
