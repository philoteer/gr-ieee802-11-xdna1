import numpy as np
import sys

import aie.iron as iron
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
try:
    from aie.iron.placers import SequentialPlacer
except ModuleNotFoundError as exc:
    if exc.name != "aie.iron.placers":
        raise
    SequentialPlacer = None


if len(sys.argv) > 1:
    from aie.iron.device import NPU1, NPU2

    dev = {"npu": NPU1, "npu1": NPU1, "npu2": NPU2}[sys.argv[1]]()
else:
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
if hasattr(Runtime, "sequence"):
    runtime = Runtime()
    with runtime.sequence(in_ty, out_ty) as (lts, channel):
        runtime.start(worker)
        runtime.fill(of_in.prod(), lts)
        runtime.drain(of_out.cons(), channel, wait=True)
    program = Program(dev, runtime)
else:
    def runtime_sequence(lts, channel, input_fifo, output_fifo):
        input_fifo.fill(lts)
        output_fifo.drain(channel, wait=True)

    runtime = Runtime(runtime_sequence, [
        in_ty, out_ty, of_in.prod(), of_out.cons(),
    ])
    program = Program(dev, runtime, workers=[worker])

module = (program.resolve_program(SequentialPlacer())
          if SequentialPlacer is not None else program.resolve_program())
print(module)
