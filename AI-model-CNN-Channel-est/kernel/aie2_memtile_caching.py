# -*- Python -*-
#
# Production IRON graph for the IEEE 802.11 receiver pipeline: magnitude
# squared, autocorrelation, threshold, short/long preamble sync, 64-bin FFT,
# and the split frame equalizer.
# Target: XDNA1 / XDNA2 (npu1 / npu2)

"""Generates the production receiver pipeline as IRON MLIR.

The graph chains magsq_avg_cf, autocorr_cf, threshold_cf, sync_short_cf,
sync_long_cf, fft_cf, frame_equalizer_preprocess, and
frame_equalizer_equalize, broadcasting the input to the first two kernels and
returning only the frame-equalizer outputs to the host. AIE_TILE_SIZE,
AIE_TILES_PER_BATCH, AIE_FIFO_DEPTH, and AIE_FIFO_DEPTHS tune the tile
geometry and FIFO depths.
"""

import numpy as np
import os

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
import aie.iron as iron

dev = iron.get_current_device()

tile_size = int(os.environ.get("AIE_TILE_SIZE", "1024"))
tiles_per_batch = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
if tile_size != 1024:
    raise ValueError("AIE_TILE_SIZE must be 1024")
if tiles_per_batch < 1:
    raise ValueError("AIE_TILES_PER_BATCH must be positive")
tensor_size = tile_size * tiles_per_batch

fifo_default_depth = int(os.environ.get("AIE_FIFO_DEPTH", "2"))
if fifo_default_depth < 1:
    raise ValueError("AIE_FIFO_DEPTH must be positive")
fifo_depth_overrides = {}
for item in os.environ.get("AIE_FIFO_DEPTHS", "").split(","):
    item = item.strip()
    if not item:
        continue
    name, separator, value = item.partition("=")
    if not separator or not name or int(value) < 1:
        raise ValueError("AIE_FIFO_DEPTHS must contain name=positive_depth entries")
    fifo_depth_overrides[name.strip()] = int(value)


def fifo_depth(name):
    """Returns the configured depth for a FIFO.

    Args:
        name: FIFO name used to look up AIE_FIFO_DEPTHS overrides.

    Returns:
        The per-FIFO override when present, otherwise the AIE_FIFO_DEPTH
        default.
    """
    return fifo_depth_overrides.get(name, fifo_default_depth)


delay_samples = 320
max_tags_per_tile = 7
sync_short_metadata_words_per_tile = 2 + 2 * max_tags_per_tile
fft_max_tags_per_tile = 31
# FFT metadata layout: two header words plus 31 (offset, CFO) pairs.
fft_metadata_words_per_tile = 2 + 2 * fft_max_tags_per_tile
# Each tile contains 1024 FFT bins, or sixteen 64-bin OFDM symbols. The
# preprocess kernel writes one beta value for each of those tile-local slots.
frame_equalizer_symbols_per_tile = tile_size // 64
# Handoff metadata = copied FFT metadata + one beta per symbol + one center
# frequency word. This is separate from the much larger final output metadata.
frame_equalizer_prepared_metadata_words_per_tile = (
    fft_metadata_words_per_tile + frame_equalizer_symbols_per_tile + 1
)
frame_equalizer_meta_words_per_tile = (
    frame_equalizer_symbols_per_tile * 64 * 2
    + frame_equalizer_prepared_metadata_words_per_tile
    + 7
    + frame_equalizer_symbols_per_tile * 2
)
frame_equalizer_max_tags_per_tile = 16
frame_equalizer_tag_metadata_words = 140
frame_equalizer_metadata_words_per_tile = (
    2 + frame_equalizer_max_tags_per_tile * frame_equalizer_tag_metadata_words
)

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
frame_equalizer_prepared_tile_ty = np.ndarray[(tile_size*2,), np.dtype[np.int32]]
frame_equalizer_meta_tile_ty = np.ndarray[
    (frame_equalizer_meta_words_per_tile,), np.dtype[np.int32]
]
frame_equalizer_out_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int8]]
frame_equalizer_out_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]

sync_short_out_meta_tensor_ty = np.ndarray[(tiles_per_batch * sync_short_metadata_words_per_tile,), np.dtype[np.int32]]
sync_short_out_meta_tile_ty = np.ndarray[(sync_short_metadata_words_per_tile,), np.dtype[np.int32]]
fft_out_meta_tensor_ty = np.ndarray[(tiles_per_batch * fft_metadata_words_per_tile,), np.dtype[np.int32]]
fft_out_meta_tile_ty = np.ndarray[(fft_metadata_words_per_tile,), np.dtype[np.int32]]
# Sideband metadata connecting preprocess to equalize; one object accompanies
# each prepared data tile.
frame_equalizer_prepared_meta_tile_ty = np.ndarray[
    (frame_equalizer_prepared_metadata_words_per_tile,), np.dtype[np.int32]
]
frame_equalizer_out_meta_tensor_ty = np.ndarray[(tiles_per_batch * frame_equalizer_metadata_words_per_tile,), np.dtype[np.int32]]
frame_equalizer_out_meta_tile_ty = np.ndarray[(frame_equalizer_metadata_words_per_tile,), np.dtype[np.int32]]
center_frequency_ty = np.ndarray[(2,), np.dtype[np.int32]]
center_frequency_rtp = Buffer(
    center_frequency_ty,
    name="center_frequency",
    use_write_rtp=True,
)

center_frequency_marker = 0x13579BDF
center_frequency_reciprocal_marker = 0x2468ACE0

def write_center_frequency_rtp(buffer):
    """Fills the center-frequency RTP buffer with its sentinel markers.

    The host adapter replaces 0x13579BDF with the center frequency in MHz and
    0x2468ACE0 with the Q30 sample-rate/center-frequency ratio while patching
    insts.bin.

    Args:
        buffer: Two-word int32 RTP buffer consumed by frame_equalizer_preprocess.
    """
    buffer[0] = center_frequency_marker
    buffer[1] = center_frequency_reciprocal_marker

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

frame_equalizer_preprocess_fn = Kernel(
    "frame_equalizer_preprocess",
    "frame_equalizer_preprocess.o",
    [fft_out_tile_ty, fft_out_meta_tile_ty, frame_equalizer_prepared_tile_ty, frame_equalizer_prepared_meta_tile_ty, center_frequency_ty, np.int32, np.int32],
)

frame_equalizer_channel_estimate_fn = Kernel(
    "frame_equalizer_channel_estimate",
    "frame_equalizer_channel_estimate.o",
    [frame_equalizer_prepared_tile_ty, frame_equalizer_prepared_meta_tile_ty, frame_equalizer_meta_tile_ty],
)

frame_equalizer_equalize_fn = Kernel(
    "frame_equalizer_equalize",
    "frame_equalizer_equalize.o",
    [frame_equalizer_prepared_tile_ty, frame_equalizer_meta_tile_ty, frame_equalizer_out_tile_ty, frame_equalizer_out_meta_tile_ty, np.int32, np.int32],
)

# Input data movement
of_in = ObjectFifo(in_tile_ty, depth=fifo_depth("in"), name="in")
of_in_mem = of_in.cons().forward(depth=fifo_depth("in_mem"), name="in_mem")

# Output data movement
of_out_magsq = ObjectFifo(magsq_out_tile_ty, depth=fifo_depth("out_magsq"), name="out_magsq")
of_out_magsq_delay = ObjectFifo(magsq_delay_out_tile_ty, depth=fifo_depth("out_magsq_delay"), name="out_magsq_delay")
of_out_autocorr = ObjectFifo(autocorr_out_tile_ty, depth=fifo_depth("out_autocorr"), name="out_autocorr")
of_out_threshold = ObjectFifo(threshold_out_tile_ty, depth=fifo_depth("threshold_met"), name="threshold_met")
of_out_sync_short_1 = ObjectFifo(sync_short_out_tile_ty_1, depth=fifo_depth("sync_short_out"), name="sync_short_out")
of_out_sync_short_2 = ObjectFifo(sync_short_out_meta_tile_ty, depth=fifo_depth("sync_short_out_meta"), name="sync_short_out_meta")
of_out_sync_long_1 = ObjectFifo(sync_long_out_tile_ty, depth=fifo_depth("sync_long_out"), name="sync_long_out")
of_out_sync_long_2 = ObjectFifo(sync_short_out_meta_tile_ty, depth=fifo_depth("sync_long_out_meta"), name="sync_long_out_meta")
of_out_fft_1 = ObjectFifo(fft_out_tile_ty, depth=fifo_depth("fft_out"), name="fft_out")
of_out_fft_2 = ObjectFifo(fft_out_meta_tile_ty, depth=fifo_depth("fft_out_meta"), name="fft_out_meta")
# These two FIFOs are a paired handoff. The data FIFO carries phase-corrected
# FFT samples; the metadata FIFO carries tags, beta values, and center frequency.
of_frame_equalizer_prepared = ObjectFifo(frame_equalizer_prepared_tile_ty, depth=fifo_depth("frame_equalizer_prepared"), name="frame_equalizer_prepared")
of_frame_equalizer_prepared_meta = ObjectFifo(frame_equalizer_prepared_meta_tile_ty, depth=fifo_depth("frame_equalizer_prepared_meta"), name="frame_equalizer_prepared_meta")
of_frame_equalizer_meta = ObjectFifo(frame_equalizer_meta_tile_ty, depth=1, name="frame_equalizer_meta")
of_out_frame_equalizer_1 = ObjectFifo(frame_equalizer_out_tile_ty, depth=fifo_depth("frame_equalizer_out"), name="frame_equalizer_out")
of_out_frame_equalizer_2 = ObjectFifo(frame_equalizer_out_meta_tile_ty, depth=fifo_depth("frame_equalizer_out_meta"), name="frame_equalizer_out_meta")

# 2. Define independent core tasks
def autocorr_core_fn(of_in, of_out_autocorr, autocorr_scalar):
    """Runs autocorr_cf over each input tile.

    Args:
        of_in: Input tile ObjectFifo consumer shared by the broadcast.
        of_out_autocorr: Q7.24 autocorrelation ObjectFifo producer.
        autocorr_scalar: autocorr_cf kernel handle.
    """
    for _ in range_(tiles_per_batch):
        elem_in = of_in.acquire(1)
        elem_out2 = of_out_autocorr.acquire(1)
        
        autocorr_scalar(elem_in, elem_out2, tile_size)
        of_in.release(1)
        of_out_autocorr.release(1)

def magsq_core_fn(of_in, of_out_magsq, of_out_magsq_delay, mag_sq_scalar):
    """Runs magsq_avg_cf over each input tile, producing both output streams.

    Args:
        of_in: Input tile ObjectFifo consumer shared by the broadcast.
        of_out_magsq: Q8.23 boxcar-sum ObjectFifo producer.
        of_out_magsq_delay: Delayed cint16 ObjectFifo producer.
        mag_sq_scalar: magsq_avg_cf kernel handle.
    """
    for _ in range_(tiles_per_batch):
        elem_in = of_in.acquire(1)
        elem_out = of_out_magsq.acquire(1)
        elem_out_delay = of_out_magsq_delay.acquire(1)
        
        mag_sq_scalar(elem_in, elem_out, elem_out_delay, tile_size)
        of_in.release(1)
        of_out_magsq.release(1)
        of_out_magsq_delay.release(1)

def threshold_core_fn(of_ac_in, of_mag_in, of_out_threshold, threshold_scalar):
    """Runs threshold_cf over each input tile, pairing the two inputs.

    Args:
        of_ac_in: Q7.24 autocorrelation ObjectFifo consumer.
        of_mag_in: Q8.23 magnitude ObjectFifo consumer.
        of_out_threshold: Threshold-flag ObjectFifo producer.
        threshold_scalar: threshold_cf kernel handle.
    """
    for _ in range_(tiles_per_batch):
        elem_ac = of_ac_in.acquire(1)
        elem_mag = of_mag_in.acquire(1)
        elem_out_threshold = of_out_threshold.acquire(1)
        
        threshold_scalar(elem_ac, elem_mag, elem_out_threshold, tile_size)
        
        of_ac_in.release(1)
        of_mag_in.release(1)
        of_out_threshold.release(1)

def sync_short_core_fn(of_in, of_abs_in, of_threshold_met_in, of_out, of_out_meta, sync_short_scalar):
    """Runs sync_short_cf over each input tile, pairing all five FIFOs.

    Args:
        of_in: Delayed cint16 ObjectFifo consumer.
        of_abs_in: Q7.24 autocorrelation ObjectFifo consumer.
        of_threshold_met_in: Threshold-flag ObjectFifo consumer.
        of_out: Delay-prefixed output ObjectFifo producer.
        of_out_meta: Output metadata ObjectFifo producer.
        sync_short_scalar: sync_short_cf kernel handle.
    """
    for _ in range_(tiles_per_batch):
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
    """Runs sync_long_cf over each input tile, pairing data and metadata.

    Args:
        of_in: Delay-prefixed cint16 ObjectFifo consumer.
        of_in_meta: Input metadata ObjectFifo consumer.
        of_out: Rotated output ObjectFifo producer.
        of_out_meta: Output metadata ObjectFifo producer.
        sync_long_scalar: sync_long_cf kernel handle.
    """
    for _ in range_(tiles_per_batch):
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
    """Runs fft_cf over each input tile, pairing data and metadata.

    Args:
        of_in: Time-domain cint16 ObjectFifo consumer.
        of_in_meta: Input metadata ObjectFifo consumer.
        of_out: FFT bin ObjectFifo producer.
        of_out_meta: Output metadata ObjectFifo producer.
        fft_scalar: fft_cf kernel handle.
    """
    for _ in range_(tiles_per_batch):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        fft_scalar(elem_in, elem_in_meta, elem_out, elem_out_meta, tile_size, fft_metadata_words_per_tile)

        of_in.release(1)
        of_in_meta.release(1)
        of_out.release(1)
        of_out_meta.release(1)

def frame_equalizer_preprocess_core_fn(of_in, of_in_meta, of_out, of_out_meta, center_frequency, preprocess_scalar):
    """Runs frame_equalizer_preprocess over each FFT tile.

    Args:
        of_in: FFT bin ObjectFifo consumer.
        of_in_meta: FFT metadata ObjectFifo consumer.
        of_out: Prepared data ObjectFifo producer.
        of_out_meta: Prepared metadata ObjectFifo producer.
        center_frequency: RTP buffer with the center-frequency snapshot.
        preprocess_scalar: frame_equalizer_preprocess kernel handle.
    """
    for _ in range_(tiles_per_batch):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        preprocess_scalar(elem_in, elem_in_meta, elem_out, elem_out_meta, center_frequency, tile_size, fft_metadata_words_per_tile)

        of_in.release(1)
        of_in_meta.release(1)
        of_out.release(1)
        of_out_meta.release(1)

def frame_equalizer_channel_estimate_core_fn(
    of_in, of_in_meta, of_meta, estimate_scalar
):
    for _ in range_(tiles_per_batch):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_meta = of_meta.acquire(1)

        estimate_scalar(
            elem_in, elem_in_meta, elem_meta
        )

        of_in.release(1)
        of_in_meta.release(1)
        of_meta.release(1)

def frame_equalizer_equalize_core_fn(of_in, of_meta, of_out, of_out_meta, equalize_scalar):
    """Runs frame_equalizer_equalize over each prepared tile.

    Args:
        of_in: Prepared data ObjectFifo consumer.
        of_in_meta: Prepared metadata ObjectFifo consumer.
        of_out: Equalized output ObjectFifo producer.
        of_out_meta: Tag metadata ObjectFifo producer.
        equalize_scalar: frame_equalizer_equalize kernel handle.
    """
    for _ in range_(tiles_per_batch):
        elem_meta = of_meta.acquire(1)
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        equalize_scalar(elem_in, elem_meta, elem_out, elem_out_meta, tile_size, frame_equalizer_metadata_words_per_tile)

        of_meta.release(1)
        of_in.release(1)
        of_out.release(1)
        of_out_meta.release(1)
        
        
# Create a worker to perform the task
worker_autocorr = Worker(autocorr_core_fn, [of_in_mem.cons(), of_out_autocorr.prod(), autocorr_fn])
worker_magsq = Worker(magsq_core_fn, [of_in_mem.cons(), of_out_magsq.prod(), of_out_magsq_delay.prod(), mag_sq_fn])
worker_threshold = Worker(threshold_core_fn, [of_out_autocorr.cons(), of_out_magsq.cons(), of_out_threshold.prod(), threshold_fn])
worker_sync_short = Worker(sync_short_core_fn, [of_out_magsq_delay.cons(), of_out_autocorr.cons(), of_out_threshold.cons(), of_out_sync_short_1.prod(), of_out_sync_short_2.prod(), sync_short_fn], stack_size=2048)
worker_sync_long = Worker(sync_long_core_fn, [of_out_sync_short_1.cons(), of_out_sync_short_2.cons(), of_out_sync_long_1.prod(), of_out_sync_long_2.prod(), sync_long_fn], stack_size=2048)
worker_fft = Worker(fft_core_fn, [of_out_sync_long_1.cons(), of_out_sync_long_2.cons(), of_out_fft_1.prod(), of_out_fft_2.prod(), fft_fn])
worker_frame_equalizer_preprocess = Worker(frame_equalizer_preprocess_core_fn, [of_out_fft_1.cons(), of_out_fft_2.cons(), of_frame_equalizer_prepared.prod(), of_frame_equalizer_prepared_meta.prod(), center_frequency_rtp, frame_equalizer_preprocess_fn], stack_size=2560)
worker_frame_equalizer_channel_estimate = Worker(frame_equalizer_channel_estimate_core_fn, [of_frame_equalizer_prepared.cons(), of_frame_equalizer_prepared_meta.cons(), of_frame_equalizer_meta.prod(), frame_equalizer_channel_estimate_fn], stack_size=2048)
worker_frame_equalizer_equalize = Worker(frame_equalizer_equalize_core_fn, [of_frame_equalizer_prepared.cons(), of_frame_equalizer_meta.cons(), of_out_frame_equalizer_1.prod(), of_out_frame_equalizer_2.prod(), frame_equalizer_equalize_fn], stack_size=4096)

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(in_tensor_ty, frame_equalizer_out_tensor_ty, frame_equalizer_out_meta_tensor_ty) as (a_in, c_out2, d_out3):
    rt.inline_ops(write_center_frequency_rtp, [center_frequency_rtp])
    # Start cores in parallel
    rt.start(worker_autocorr)
    rt.start(worker_magsq)
    rt.start(worker_threshold)
    rt.start(worker_sync_short)
    rt.start(worker_sync_long)
    rt.start(worker_fft)
    rt.start(worker_frame_equalizer_preprocess)
    rt.start(worker_frame_equalizer_channel_estimate)
    rt.start(worker_frame_equalizer_equalize)

    # Push input once; MemTile implicitly broadcasts it to both cores.
    rt.fill(of_in.prod(), a_in)
    
    # Gather remaining outputs back to the host
    rt.drain(of_out_frame_equalizer_1.cons(), c_out2, wait=False)
    rt.drain(of_out_frame_equalizer_2.cons(), d_out3, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Placement is intentionally unchanged while the core count is being expanded;
# revisit this after the new cores land because it can invalidate placement A/Bs.
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
