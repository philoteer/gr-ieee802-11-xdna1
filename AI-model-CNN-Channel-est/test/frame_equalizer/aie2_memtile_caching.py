# -*- Python -*-
#
# Standalone frame_equalizer test graph for IRON.
# Target: XDNA1 / XDNA2 (npu1 / npu2)

"""Standalone IRON graph for the split frame equalizer.

Runs frame_equalizer_preprocess and frame_equalizer_equalize on separate AIE
tiles, streaming FFT tiles and their metadata through the prepared handoff
FIFOs and draining the equalized output and tag metadata.
"""

import numpy as np
import os

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
import aie.iron as iron

dev = iron.get_current_device()

tile_size = 1024
divider = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
if divider < 1:
    raise ValueError("AIE_TILES_PER_BATCH must be positive")
tensor_size = tile_size * divider
fft_max_tags_per_tile = 31
# FFT metadata layout: two header words plus 31 (offset, CFO) pairs.
fft_metadata_words_per_tile = 2 + 2 * fft_max_tags_per_tile
# Each 1024-sample tile contains sixteen 64-bin OFDM symbols. Beta is stored
# in physical tile order, rather than by frame-relative symbol number.
frame_equalizer_symbols_per_tile = tile_size // 64
# Handoff metadata = copied FFT metadata + one beta per symbol + one center
# frequency word. This is separate from the final output metadata.
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

# Complex cint32 values use interleaved real/imaginary int32 components.
in_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int32]]
in_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]
in_meta_tensor_ty = np.ndarray[(divider * fft_metadata_words_per_tile,), np.dtype[np.int32]]
in_meta_tile_ty = np.ndarray[(fft_metadata_words_per_tile,), np.dtype[np.int32]]
prepared_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]
meta_tile_ty = np.ndarray[
    (frame_equalizer_meta_words_per_tile,), np.dtype[np.int32]
]
# Sideband metadata accompanying each prepared data tile.
prepared_meta_tile_ty = np.ndarray[
    (frame_equalizer_prepared_metadata_words_per_tile,), np.dtype[np.int32]
]
out_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
out_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int8]]
out_meta_tensor_ty = np.ndarray[(divider * frame_equalizer_metadata_words_per_tile,), np.dtype[np.int32]]
out_meta_tile_ty = np.ndarray[(frame_equalizer_metadata_words_per_tile,), np.dtype[np.int32]]
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

    The host flowgraph replaces 0x13579BDF with the center frequency in MHz
    and 0x2468ACE0 with the Q30 sample-rate/center-frequency ratio while
    patching insts.bin.

    Args:
        buffer: Two-word int32 RTP buffer consumed by frame_equalizer_preprocess.
    """
    buffer[0] = center_frequency_marker
    buffer[1] = center_frequency_reciprocal_marker

frame_equalizer_preprocess_fn = Kernel(
    "frame_equalizer_preprocess",
    "frame_equalizer_preprocess.o",
    [in_tile_ty, in_meta_tile_ty, prepared_tile_ty, prepared_meta_tile_ty, center_frequency_ty, np.int32, np.int32],
)

frame_equalizer_channel_estimate_fn = Kernel(
    "frame_equalizer_channel_estimate",
    "frame_equalizer_channel_estimate.o",
    [prepared_tile_ty, prepared_meta_tile_ty, meta_tile_ty],
)

frame_equalizer_equalize_fn = Kernel(
    "frame_equalizer_equalize",
    "frame_equalizer_equalize.o",
    [prepared_tile_ty, meta_tile_ty, out_tile_ty, out_meta_tile_ty, np.int32, np.int32],
)

of_in = ObjectFifo(in_tile_ty, name="in")
of_in_meta = ObjectFifo(in_meta_tile_ty, name="in_meta")
# Prepared data and prepared metadata are paired: the first carries corrected
# FFT samples, while the second carries tags, beta values, and center frequency.
of_prepared = ObjectFifo(prepared_tile_ty, name="prepared")
of_prepared_meta = ObjectFifo(prepared_meta_tile_ty, name="prepared_meta")
of_meta = ObjectFifo(meta_tile_ty, depth=1, name="meta")
of_out = ObjectFifo(out_tile_ty, name="out")
of_out_meta = ObjectFifo(out_meta_tile_ty, name="out_meta")


def frame_equalizer_preprocess_core_fn(
    of_in, of_in_meta, of_out, of_out_meta, center_frequency, preprocess_scalar
):
    """Runs frame_equalizer_preprocess over each FFT tile.

    Args:
        of_in: FFT bin ObjectFifo consumer.
        of_in_meta: FFT metadata ObjectFifo consumer.
        of_out: Prepared data ObjectFifo producer.
        of_out_meta: Prepared metadata ObjectFifo producer.
        center_frequency: RTP buffer with the center-frequency snapshot.
        preprocess_scalar: frame_equalizer_preprocess kernel handle.
    """
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        preprocess_scalar(
            elem_in,
            elem_in_meta,
            elem_out,
            elem_out_meta,
            center_frequency,
            tile_size,
            fft_metadata_words_per_tile,
        )

        of_in.release(1)
        of_in_meta.release(1)
        of_out.release(1)
        of_out_meta.release(1)


def frame_equalizer_channel_estimate_core_fn(
    of_in, of_in_meta, of_meta, estimate_scalar
):
    for _ in range_(divider):
        elem_in = of_in.acquire(1)
        elem_in_meta = of_in_meta.acquire(1)
        elem_meta = of_meta.acquire(1)

        estimate_scalar(
            elem_in,
            elem_in_meta,
            elem_meta,
        )

        of_in.release(1)
        of_in_meta.release(1)
        of_meta.release(1)


def frame_equalizer_equalize_core_fn(
    of_in, of_meta, of_out, of_out_meta, equalize_scalar
):
    """Runs frame_equalizer_equalize over each prepared tile.

    Args:
        of_in: Prepared data ObjectFifo consumer.
        of_in_meta: Prepared metadata ObjectFifo consumer.
        of_out: Equalized output ObjectFifo producer.
        of_out_meta: Tag metadata ObjectFifo producer.
        equalize_scalar: frame_equalizer_equalize kernel handle.
    """
    for _ in range_(divider):
        elem_meta = of_meta.acquire(1)
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out_meta = of_out_meta.acquire(1)

        equalize_scalar(
            elem_in,
            elem_meta,
            elem_out,
            elem_out_meta,
            tile_size,
            frame_equalizer_metadata_words_per_tile,
        )

        of_meta.release(1)
        of_in.release(1)
        of_out.release(1)
        of_out_meta.release(1)


preprocess_worker = Worker(
    frame_equalizer_preprocess_core_fn,
    [
        of_in.cons(),
        of_in_meta.cons(),
        of_prepared.prod(),
        of_prepared_meta.prod(),
        center_frequency_rtp,
        frame_equalizer_preprocess_fn,
    ],
    stack_size=2560,
)

channel_estimate_worker = Worker(
    frame_equalizer_channel_estimate_core_fn,
    [
        of_prepared.cons(),
        of_prepared_meta.cons(),
        of_meta.prod(),
        frame_equalizer_channel_estimate_fn,
    ],
    stack_size=2048,
)

equalize_worker = Worker(
    frame_equalizer_equalize_core_fn,
    [
        of_prepared.cons(),
        of_meta.cons(),
        of_out.prod(),
        of_out_meta.prod(),
        frame_equalizer_equalize_fn,
    ],
    stack_size=4096,
)

rt = Runtime()
with rt.sequence(
    in_tensor_ty, in_meta_tensor_ty, out_tensor_ty, out_meta_tensor_ty
) as (a_in, b_in_meta, c_out, d_out_meta):
    rt.inline_ops(write_center_frequency_rtp, [center_frequency_rtp])
    rt.start(preprocess_worker)
    rt.start(channel_estimate_worker)
    rt.start(equalize_worker)
    rt.fill(of_in.prod(), a_in)
    rt.fill(of_in_meta.prod(), b_in_meta)
    rt.drain(of_out.cons(), c_out, wait=False)
    rt.drain(of_out_meta.cons(), d_out_meta, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
