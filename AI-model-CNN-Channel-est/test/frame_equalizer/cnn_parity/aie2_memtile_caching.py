"""Standalone IRON graph for the cnn parity test kernel.

Streams two LTS symbols through cnn_parity_test and drains the channel
estimate it produces via channel_estimator/select.h.
"""

import numpy as np

import aie.iron as iron
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


dev = iron.get_current_device()

in_ty = np.ndarray[(2 * 64 * 2,), np.dtype[np.int32]]
out_ty = np.ndarray[(64 * 2,), np.dtype[np.int32]]

cnn_fn = Kernel("cnn_parity_test", "cnn_parity_test.o", [in_ty, out_ty])
of_in = ObjectFifo(in_ty, name="cnn_lts_in")
of_out = ObjectFifo(out_ty, name="cnn_channel_out")


def core(of_in, of_out, cnn):
    """Run one direct LTS-to-channel invocation on the AIE tile."""
    lts = of_in.acquire(1)
    channel = of_out.acquire(1)
    cnn(lts, channel)
    of_in.release(1)
    of_out.release(1)


worker = Worker(core, [of_in.cons(), of_out.prod(), cnn_fn])
runtime = Runtime()
with runtime.sequence(in_ty, out_ty) as (lts, channel):
    runtime.start(worker)
    runtime.fill(of_in.prod(), lts)
    runtime.drain(of_out.cons(), channel, wait=True)

program = Program(dev, runtime)
print(program.resolve_program(SequentialPlacer()))
