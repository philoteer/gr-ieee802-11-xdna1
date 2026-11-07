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
in_ty = np.ndarray[(7 * 64 * 2,), np.dtype[np.int32]]
out_ty = np.ndarray[(2 * 64 * 2,), np.dtype[np.int32]]
kernel = Kernel("updater_parity_test", "updater_parity_test.o", [in_ty, out_ty])
of_in = ObjectFifo(in_ty, name="updater_symbols_in")
of_out = ObjectFifo(out_ty, name="updater_channel_out")


def core(source, destination, function):
    symbols = source.acquire(1)
    channel = destination.acquire(1)
    function(symbols, channel)
    source.release(1)
    destination.release(1)


worker = Worker(core, [of_in.cons(), of_out.prod(), kernel])
if hasattr(Runtime, "sequence"):
    runtime = Runtime()
    with runtime.sequence(in_ty, out_ty) as (symbols, channel):
        runtime.start(worker)
        runtime.fill(of_in.prod(), symbols)
        runtime.drain(of_out.cons(), channel, wait=True)
    program = Program(dev, runtime)
else:
    def runtime_sequence(symbols, channel, source, destination):
        source.fill(symbols)
        destination.drain(channel, wait=True)

    runtime = Runtime(runtime_sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    program = Program(dev, runtime, workers=[worker])
print(program.resolve_program(SequentialPlacer())
      if SequentialPlacer is not None else program.resolve_program())
