import numpy as np

import aie.iron as iron
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


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
runtime = Runtime()
with runtime.sequence(in_ty, out_ty) as (symbols, channel):
    runtime.start(worker)
    runtime.fill(of_in.prod(), symbols)
    runtime.drain(of_out.cons(), channel, wait=True)

print(Program(dev, runtime).resolve_program(SequentialPlacer()))
