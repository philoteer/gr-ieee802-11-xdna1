#!/usr/bin/env python3

import os

import numpy as np

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel


TILE_SIZE = 1024
TILES = int(os.environ.get("AIE_TILES_PER_BATCH", "16"))
TENSOR_SIZE = TILE_SIZE * TILES
FFT_SIZE = 64
MAX_TAGS_PER_TILE = 31
META_WORDS = 2 + 2 * MAX_TAGS_PER_TILE
MAX_INTEGER_ERROR = 16


def make_input(samples, counts, tags, start):
    data = np.zeros(TENSOR_SIZE * 2, dtype=np.int16)
    metadata = np.zeros((TILES, META_WORDS), dtype=np.int32)
    cursor = start
    for tile, count in enumerate(counts):
        tile_samples = samples[cursor:cursor + count]
        tile_start = tile * TILE_SIZE * 2
        data[tile_start:tile_start + 2 * count:2] = tile_samples.real
        data[tile_start + 1:tile_start + 2 * count:2] = tile_samples.imag
        metadata[tile, 0] = count
        metadata[tile, 1] = len(tags[tile])
        for tag_index, (offset, value) in enumerate(tags[tile]):
            metadata[tile, 2 + 2 * tag_index] = offset
            metadata[tile, 3 + 2 * tag_index] = value
        cursor += count
    return data, metadata.reshape(-1), cursor


def run(kernel_handle, data, metadata):
    output = iron.zeros(TENSOR_SIZE * 2, dtype=np.int32)
    output_metadata = iron.zeros(TILES * META_WORDS, dtype=np.int32)
    DefaultNPURuntime.run(
        kernel_handle,
        [
            iron.tensor(data, dtype=np.int16),
            iron.tensor(metadata, dtype=np.int32),
            output,
            output_metadata,
        ],
    )
    return output.numpy().copy(), output_metadata.numpy().reshape(TILES, META_WORDS).copy()


def gather(output, metadata):
    chunks = []
    tags = []
    logical_start = 0
    for tile in range(TILES):
        count = int(metadata[tile, 0])
        tile_start = tile * TILE_SIZE * 2
        components = output[tile_start:tile_start + 2 * count]
        chunks.append(components[::2] + 1j * components[1::2])
        for tag_index in range(int(metadata[tile, 1])):
            offset = int(metadata[tile, 2 + 2 * tag_index])
            value = int(metadata[tile, 3 + 2 * tag_index])
            tags.append((logical_start + offset, value))
        logical_start += count
    return np.concatenate(chunks), tags


def main():
    rng = np.random.default_rng(80211)
    source = (
        rng.integers(-1000, 1001, 1216, dtype=np.int16).astype(np.int32)
        + 1j * rng.integers(-1000, 1001, 1216, dtype=np.int16).astype(np.int32)
    )

    first_counts = [31, 97, 1000, 13] + [0] * (TILES - 4)
    first_tags = [[(10, 111)], [(33, 222), (96, 333)],
                  [(100, 444), (990, 555)], [(5, 666)]]
    first_tags += [[]] * (TILES - 4)
    first_data, first_meta, cursor = make_input(source, first_counts, first_tags, 0)
    second_data, second_meta, cursor = make_input(
        source, [11] + [0] * (TILES - 1), [[]] * TILES, cursor
    )
    third_data, third_meta, cursor = make_input(
        source, [64] + [0] * (TILES - 1), [[]] * TILES, cursor
    )
    assert cursor == len(source)

    kernel = NPUKernel("build/final.xclbin", "build/insts.bin", kernel_name="MLIR_AIE")
    kernel_handle = DefaultNPURuntime.load(kernel)
    first_output, first_output_meta = run(kernel_handle, first_data, first_meta)
    second_output, second_output_meta = run(kernel_handle, second_data, second_meta)
    third_output, third_output_meta = run(kernel_handle, third_data, third_meta)

    first_values, first_tags = gather(first_output, first_output_meta)
    second_values, second_tags = gather(second_output, second_output_meta)
    third_values, third_tags = gather(third_output, third_output_meta)
    actual = np.concatenate([first_values, second_values, third_values])

    expected = np.fft.fftshift(
        np.fft.fft(source.reshape(-1, FFT_SIZE), axis=1), axes=1
    ).reshape(-1)
    real_error = np.max(np.abs(actual.real - np.rint(expected.real)))
    imag_error = np.max(np.abs(actual.imag - np.rint(expected.imag)))
    if real_error > MAX_INTEGER_ERROR or imag_error > MAX_INTEGER_ERROR:
        real_index = int(np.argmax(np.abs(actual.real - np.rint(expected.real))))
        imag_index = int(np.argmax(np.abs(actual.imag - np.rint(expected.imag))))
        raise AssertionError(
            f"FFT error too large: real={real_error}, imag={imag_error}, "
            f"limit={MAX_INTEGER_ERROR}; actual[:8]={actual[:8]}, "
            f"expected[:8]={expected[:8]}; real_index={real_index}, "
            f"actual={actual[real_index]}, expected={expected[real_index]}; "
            f"imag_index={imag_index}, actual={actual[imag_index]}, "
            f"expected={expected[imag_index]}; actual[64:72]={actual[64:72]}, "
            f"expected[64:72]={expected[64:72]}"
        )

    expected_tags = [(64, 111), (64, 222), (128, 333), (256, 444),
                     (1152, 555), (1152, 666)]
    actual_tags = first_tags + [
        (len(first_values) + offset, value) for offset, value in second_tags
    ] + [
        (len(first_values) + len(second_values) + offset, value)
        for offset, value in third_tags
    ]
    if actual_tags != expected_tags:
        raise AssertionError(f"tag mismatch: {actual_tags} != {expected_tags}")

    print(
        f"fft_cf test passed: {len(actual)} samples, "
        f"max integer error real={real_error}, imag={imag_error}"
    )


if __name__ == "__main__":
    main()
