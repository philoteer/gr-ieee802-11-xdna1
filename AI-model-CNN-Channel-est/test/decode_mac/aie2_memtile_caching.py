"""Stateful MAC decoder accepting the frame equalizer's data/metadata ABI."""

import sys
import os
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2

try:
    from aie.iron.placers import SequentialPlacer
except ModuleNotFoundError as exc:
    if exc.name != "aie.iron.placers":
        raise
    SequentialPlacer = None

device = {"npu": NPU1, "npu1": NPU1, "npu2": NPU2}[
    sys.argv[1] if len(sys.argv) > 1 else "npu"]()
input_offset = int(os.environ.get("DECODE_MAC_INPUT_OFFSET", "0"))
tile_ty = np.ndarray[(1024 + input_offset,), np.dtype[np.int8]]
# sizeof(frame_equalizer_tile_metadata) == 8968, including aligned tag padding.
meta_ty = np.ndarray[(2242,), np.dtype[np.int32]]
out_ty = np.ndarray[(8192,), np.dtype[np.int8]]
out_meta_ty = np.ndarray[(2274,), np.dtype[np.int32]]
kernel = Kernel("decode_mac", "decode_mac.o",
                [tile_ty, meta_ty, out_ty, out_meta_ty, np.int32, np.int32])
source = ObjectFifo(tile_ty, name="decisions", depth=1)
source_meta = ObjectFifo(meta_ty, name="equalizer_metadata", depth=1)
destination = ObjectFifo(out_ty, name="mac_frames", depth=1)
destination_meta = ObjectFifo(out_meta_ty, name="mac_metadata", depth=1)


def copy_tile(source, source_meta, destination, destination_meta, kernel):
    for _ in range_(1):
        incoming = source.acquire(1)
        metadata = source_meta.acquire(1)
        outgoing = destination.acquire(1)
        outgoing_meta = destination_meta.acquire(1)
        kernel(incoming, metadata, outgoing, outgoing_meta, 1024, 2242)
        source.release(1)
        source_meta.release(1)
        destination.release(1)
        destination_meta.release(1)


worker = Worker(copy_tile, [source.cons(), source_meta.cons(),
                           destination.prod(), destination_meta.prod(), kernel],
                stack_size=2048)  # Match the fused production MAC worker.
if hasattr(Runtime, "sequence"):
    runtime = Runtime()
    with runtime.sequence(tile_ty, meta_ty, out_ty, out_meta_ty) as (
            incoming, metadata, outgoing, outgoing_meta):
        runtime.start(worker)
        runtime.fill(source.prod(), incoming)
        runtime.fill(source_meta.prod(), metadata)
        runtime.drain(destination.cons(), outgoing, wait=False)
        runtime.drain(destination_meta.cons(), outgoing_meta, wait=True)
    program = Program(device, runtime)
else:
    def sequence(incoming, metadata, outgoing, outgoing_meta,
                 source, source_meta, destination, destination_meta):
        source.fill(incoming)
        source_meta.fill(metadata)
        destination.drain(outgoing, wait=False)
        destination_meta.drain(outgoing_meta, wait=True)

    runtime = Runtime(sequence, [tile_ty, meta_ty, out_ty, out_meta_ty,
                                 source.prod(), source_meta.prod(),
                                 destination.cons(), destination_meta.cons()])
    program = Program(device, runtime, workers=[worker])

print(program.resolve_program(SequentialPlacer())
      if SequentialPlacer is not None else program.resolve_program())
