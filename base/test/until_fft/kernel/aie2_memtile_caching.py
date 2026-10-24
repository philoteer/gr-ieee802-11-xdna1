# -*- Python -*-
#
# GNU Radio - Complex to Magnitude Squared via IRON
# Target: XDNA1 / XDNA2 (npu1 / npu2)

import numpy as np
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
delay_samples = 320
max_tags_per_tile = 7
sync_short_metadata_words_per_tile = 2 + 2 * max_tags_per_tile
fft_max_tags_per_tile = 31
fft_metadata_words_per_tile = 2 + 2 * fft_max_tags_per_tile

# Define tensor types
# Input is complex cint16 (requires *2 for real/imag interleaved components)
in_tensor_ty = np.ndarray[(tensor_size*2,), np.dtype[np.int16]]
in_tile_ty = np.ndarray[(tile_size*2,), np.dtype[np.int16]]

# magsq_avg_cf emits the boxcar sum as Q8.23 int32.
magsq_out_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
# Delayed complex cint16 data uses interleaved real/imag components.
magsq_delay_out_tile_ty = np.ndarray[(tile_size*2,), np.dtype[np.int16]]

# autocorr_cf emits complex Q7.24 as interleaved int32 components.
autocorr_out_tile_ty = np.ndarray[(tile_size*2,), np.dtype[np.int32]]

# threshold_cf emits one threshold-met flag per sample.
threshold_out_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
threshold_out_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int8]]

# sync_long_cf intentionally performs no runtime buffer or metadata bounds checks.
# Its input must hold delay_samples + tile_size complex samples, its metadata must
# hold 2 + 2 * max_tags_per_tile int32 words, and tag offsets must be ordered and
# within the reported valid-sample count. With tile_size=1024 and min_gap=480,
# its worst-case output is 896 complex samples; do not reduce its output below that.
sync_short_out_tile_words = (tile_size + delay_samples) * 2
sync_short_out_tile_ty_1 = np.ndarray[(sync_short_out_tile_words,), np.dtype[np.int16]]
sync_long_out_tile_ty = np.ndarray[(tile_size*2,), np.dtype[np.int16]]
fft_out_tile_ty = np.ndarray[(tile_size*2,), np.dtype[np.int32]]
fft_out_tensor_ty = np.ndarray[(tensor_size*2,), np.dtype[np.int32]]

sync_short_out_meta_tensor_ty = np.ndarray[(divider * sync_short_metadata_words_per_tile,), np.dtype[np.int32]]
sync_short_out_meta_tile_ty = np.ndarray[(sync_short_metadata_words_per_tile,), np.dtype[np.int32]]
fft_out_meta_tensor_ty = np.ndarray[(divider * fft_metadata_words_per_tile,), np.dtype[np.int32]]
fft_out_meta_tile_ty = np.ndarray[(fft_metadata_words_per_tile,), np.dtype[np.int32]]

# External, binary kernel definition
mag_sq_fn = Kernel(
    "magsq_avg_cf",
    "magsq_avg_cf.o",
    [in_tile_ty, magsq_out_tile_ty, magsq_delay_out_tile_ty, np.int32],
)

autocorr_fn = Kernel(
    "autocorr_cf",
    "autocorr_cf.o",
    [in_tile_ty, autocorr_out_tile_ty, np.int32],
)

threshold_fn = Kernel(
    "threshold_cf",
    "threshold_cf.o",
    [autocorr_out_tile_ty, magsq_out_tile_ty, threshold_out_tile_ty, np.int32],
)

sync_short_fn = Kernel(
    "sync_short_cf",
    "sync_short_cf.o",
    [magsq_delay_out_tile_ty, autocorr_out_tile_ty, threshold_out_tile_ty, sync_short_out_tile_ty_1, sync_short_out_meta_tile_ty, np.int32],
)

sync_long_fn = Kernel(
    "sync_long_cf",
    "sync_long_cf.o",
    [sync_short_out_tile_ty_1, sync_short_out_meta_tile_ty, sync_long_out_tile_ty, sync_short_out_meta_tile_ty, np.int32, np.int32],
)

fft_fn = Kernel(
    "fft_cf",
    "fft_cf.o",
    [sync_long_out_tile_ty, sync_short_out_meta_tile_ty, fft_out_tile_ty, fft_out_meta_tile_ty, np.int32, np.int32],
)

# Input data movement
of_in = ObjectFifo(in_tile_ty, name="in")
of_in_mem = of_in.cons().forward(name="in_mem")

# Output data movement
of_out_magsq = ObjectFifo(magsq_out_tile_ty, name="out_magsq")
of_out_magsq_delay = ObjectFifo(magsq_delay_out_tile_ty, name="out_magsq_delay")
of_out_autocorr = ObjectFifo(autocorr_out_tile_ty, name="out_autocorr")
of_out_threshold = ObjectFifo(threshold_out_tile_ty, name="threshold_met")
of_out_sync_short_1 = ObjectFifo(sync_short_out_tile_ty_1, name="sync_short_out")
of_out_sync_short_2 = ObjectFifo(sync_short_out_meta_tile_ty, name="sync_short_out_meta")
of_out_sync_long_1 = ObjectFifo(sync_long_out_tile_ty, name="sync_long_out")
of_out_sync_long_2 = ObjectFifo(sync_short_out_meta_tile_ty, name="sync_long_out_meta")
of_out_fft_1 = ObjectFifo(fft_out_tile_ty, name="fft_out")
of_out_fft_2 = ObjectFifo(fft_out_meta_tile_ty, name="fft_out_meta")

# 2. Define independent core tasks
def autocorr_core_fn(of_in, of_out_autocorr, autocorr_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_out2 = of_out_autocorr.acquire(1)
        
        autocorr_scalar(elem_in, elem_out2, tile_size)
        of_in.release(1)
        of_out_autocorr.release(1)

def magsq_core_fn(of_in, of_out_magsq, of_out_magsq_delay, mag_sq_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_out = of_out_magsq.acquire(1)
        elem_out_delay = of_out_magsq_delay.acquire(1)
        
        mag_sq_scalar(elem_in, elem_out, elem_out_delay, tile_size)
        of_in.release(1)
        of_out_magsq.release(1)
        of_out_magsq_delay.release(1)

def threshold_core_fn(of_ac_in, of_mag_in, of_out_threshold, threshold_scalar):
    for _ in range_(divider):
        elem_ac = of_ac_in.acquire(1)
        elem_mag = of_mag_in.acquire(1)
        elem_out_threshold = of_out_threshold.acquire(1)
        
        threshold_scalar(elem_ac, elem_mag, elem_out_threshold, tile_size)
        
        of_ac_in.release(1)
        of_mag_in.release(1)
        of_out_threshold.release(1)

def sync_short_core_fn(of_in, of_abs_in, of_threshold_met_in, of_out, of_out_meta, sync_short_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_abs_in = of_abs_in.acquire(1)
        elem_threshold_met = of_threshold_met_in.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        sync_short_scalar(elem_in, elem_abs_in, elem_threshold_met, elem_out, elem_out_meta, tile_size)

        of_in.release(1)
        of_abs_in.release(1)
        of_threshold_met_in.release(1)
        of_out.release(1)
        of_out_meta.release(1)

def sync_long_core_fn(of_in, of_in_meta, of_out, of_out_meta, sync_long_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        sync_long_scalar(elem_in, elem_in_meta, elem_out, elem_out_meta, tile_size, sync_short_metadata_words_per_tile)

        of_in.release(1)
        of_in_meta.release(1)
        of_out.release(1)
        of_out_meta.release(1)

def fft_core_fn(of_in, of_in_meta, of_out, of_out_meta, fft_scalar):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        fft_scalar(elem_in, elem_in_meta, elem_out, elem_out_meta, tile_size, fft_metadata_words_per_tile)

        of_in.release(1)
        of_in_meta.release(1)
        of_out.release(1)
        of_out_meta.release(1)
        
        
# Create a worker to perform the task
worker_autocorr = Worker(autocorr_core_fn, [of_in_mem.cons(), of_out_autocorr.prod(), autocorr_fn])
worker_magsq = Worker(magsq_core_fn, [of_in_mem.cons(), of_out_magsq.prod(), of_out_magsq_delay.prod(), mag_sq_fn])
worker_threshold = Worker(threshold_core_fn, [of_out_autocorr.cons(), of_out_magsq.cons(), of_out_threshold.prod(), threshold_fn])
worker_sync_short = Worker(sync_short_core_fn, [of_out_magsq_delay.cons(), of_out_autocorr.cons(), of_out_threshold.cons(), of_out_sync_short_1.prod(), of_out_sync_short_2.prod(), sync_short_fn], stack_size=2048)
worker_sync_long = Worker(sync_long_core_fn, [of_out_sync_short_1.cons(), of_out_sync_short_2.cons(), of_out_sync_long_1.prod(), of_out_sync_long_2.prod(), sync_long_fn], stack_size=2048)
worker_fft = Worker(fft_core_fn, [of_out_sync_long_1.cons(), of_out_sync_long_2.cons(), of_out_fft_1.prod(), of_out_fft_2.prod(), fft_fn])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(in_tensor_ty, fft_out_tensor_ty, fft_out_meta_tensor_ty) as (a_in, c_out2, d_out3):
    # Start cores in parallel
    rt.start(worker_autocorr)
    rt.start(worker_magsq)
    rt.start(worker_threshold)
    rt.start(worker_sync_short)
    rt.start(worker_sync_long)
    rt.start(worker_fft)

    # Push input once; MemTile implicitly broadcasts it to both cores
    rt.fill(of_in.prod(), a_in)
    
    # Gather remaining outputs back to the host
    rt.drain(of_out_fft_1.cons(), c_out2, wait=False)    # FFT output
    rt.drain(of_out_fft_2.cons(), d_out3, wait=True)     # FFT tag metadata

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components and generate the MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
