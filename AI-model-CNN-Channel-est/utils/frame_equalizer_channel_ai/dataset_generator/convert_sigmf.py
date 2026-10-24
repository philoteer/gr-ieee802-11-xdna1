#!/usr/bin/env python3
"""Build channel-estimator training datasets from collected SigMF recordings.

The converter uses CRC-valid 52-tone Wi-Fi frames, forms two-symbol channel
observations, and estimates each target from nearby reference symbols that do
not overlap the target pair.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from sigmf import sigmffile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.frame_equalizer_channel_ai.common.dataset import (
    ChannelCorpus, FFT_SIZE, LONG, OCCUPIED, concatenate_corpora, corpus_digest,
    usable_pair_indices,
)
from utils.frame_equalizer_channel_ai.dataset_generator.corpus import _build_corpus, _round_complex


TONES_PER_SYMBOL = int(np.count_nonzero(OCCUPIED))
DEFAULT_RECORDINGS = PROJECT_ROOT / 'data' / 'datasets' / 'channel_estimator' / '20260927_room'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'build'


@dataclass(frozen=True)
class Frame:
    """A converted frame and its source location.

    Attributes:
        source: Path to the source SigMF recording.
        offset: Starting sample offset of the frame in the recording.
        channel: Wi-Fi channel number, or ``None`` if it cannot be identified.
        corpus: Training examples generated from the frame.
    """

    source: str
    offset: int
    channel: int | None
    corpus: ChannelCorpus
    samples: np.ndarray | None = None


def wifi_channel(frequency: object) -> int | None:
    """Map a nominal 2.4 GHz Wi-Fi frequency to its channel number.

    Args:
        frequency: Center frequency in Hz.

    Returns:
        Channel 1 through 13 for a frequency within 1 MHz of a channel center;
        otherwise, ``None``.
    """
    if not isinstance(frequency, (int, float)) or not np.isfinite(frequency):
        return None
    channel = round((frequency - 2412e6) / 5e6) + 1
    expected = 2412e6 + 5e6 * (channel - 1)
    return channel if 1 <= channel <= 13 and abs(frequency - expected) < 1e6 else None


def frame_examples(samples: np.ndarray, reference_radius: int,
                   min_reference_symbols: int) -> ChannelCorpus | None:
    """Convert one frame into paired observations and channel-reference labels.

    Each example predicts two consecutive symbols. Reference symbols are drawn
    from the surrounding window while excluding both predicted symbols, so the
    label does not reuse its own observations. For 3-channel captures, the
    middle channel is ignored; channel 0 is received IQ and the last channel is
    transmitted ground truth.

    Args:
        samples: Frame shaped ``(symbols, 52, channels)`` with 2 or 3 channels.
        reference_radius: Number of candidate reference symbols on each side.
        min_reference_symbols: Minimum number of references required per pair.

    Returns:
        A ``ChannelCorpus`` of examples, or ``None`` if the frame has invalid
        samples or yields no usable reference windows.

    Raises:
        ValueError: If the sample array has an unsupported shape.
    """
    if samples.ndim != 3 or samples.shape[1] != TONES_PER_SYMBOL or samples.shape[2] not in (2, 3):
        raise ValueError(f'expected (symbols, {TONES_PER_SYMBOL}, 2 or 3) SigMF samples; got {samples.shape}')
    received = samples[:, :, 0]
    transmitted = samples[:, :, -1]
    if (not np.all(np.isfinite(received)) or not np.all(np.isfinite(transmitted)) or
            np.any(np.abs(transmitted) < 1e-5)):
        return None
    channel = received / transmitted
    amplitude = float(np.median(np.abs(channel)))
    if not np.isfinite(amplitude) or amplitude < 1e-5:
        return None
    # Match the median response to the corpus scale while leaving headroom below
    # signed-Q15 full scale for unusually large per-tone peaks.
    scale = min(12000.0 / amplitude, 30000.0 / float(np.max(np.abs(channel))))
    observations = []
    truths = []
    for index in usable_pair_indices(len(samples), reference_radius, min_reference_symbols):
        left = np.arange(max(0, index - reference_radius), index)
        right = np.arange(index + 2, min(len(samples), index + 2 + reference_radius))
        references = np.concatenate((left, right))
        signal = received[references] * np.conj(transmitted[references])
        power = np.abs(transmitted[references]) ** 2
        reference = np.sum(signal, axis=0, dtype=np.complex128) / np.sum(power, axis=0)
        two_observations = np.zeros((2, FFT_SIZE), dtype=np.complex128)
        two_observations[:, OCCUPIED] = channel[index:index + 2] * (scale * LONG[OCCUPIED])
        truth = np.zeros(FFT_SIZE, dtype=np.complex128)
        truth[OCCUPIED] = reference * scale
        observations.append(_round_complex(two_observations))
        truths.append(_round_complex(truth))
    if not observations:
        return None
    return _build_corpus(np.stack(observations), np.stack(truths))


def read_frames(paths: list[Path], reference_radius: int,
                min_reference_symbols: int,
                retain_samples: bool = False) -> tuple[list[Frame], Counter]:
    """Read annotated Wi-Fi frames and convert usable frames into corpora.

    Args:
        paths: SigMF ``.sigmf-meta`` files to process.
        reference_radius: Reference-window radius passed to ``frame_examples``.
        min_reference_symbols: Minimum independent references per target pair.

    Returns:
        A list of converted frames and a counter describing skipped frames.

    Raises:
        ValueError: If a recording has unsupported format or invalid frame
            annotations, or if no usable frame is found.
    """
    frames = []
    skipped = Counter()
    for path in paths:
        recording = sigmffile.fromfile(path, skip_checksum=True)
        channels = recording.get_global_field('core:num_channels', 1)
        if channels not in (2, 3) or recording.get_global_field('core:datatype') != 'cf32_le':
            raise ValueError(f'{path}: expected 2- or 3-channel cf32_le data')
        captures = recording.get_captures()
        capture_starts = [capture['core:sample_start'] for capture in captures]
        if capture_starts != sorted(capture_starts):
            raise ValueError(f'{path}: capture sample starts are out of order')
        for annotation in recording.get_annotations():
            if annotation.get('core:label') != 'wifi-frame':
                continue
            start = annotation['core:sample_start']
            count = annotation.get('core:sample_count', 0)
            if start < 0 or count <= 0 or start + count > len(recording) or count % TONES_PER_SYMBOL:
                raise ValueError(f'{path}: invalid 52-tone frame at sample {start} ({count} samples)')
            capture_index = bisect_right(capture_starts, start) - 1
            channel_number = (wifi_channel(captures[capture_index].get('core:frequency'))
                              if capture_index >= 0 else None)
            samples = np.asarray(recording.read_samples(start, count)).reshape(-1, TONES_PER_SYMBOL, channels)
            corpus = frame_examples(samples, reference_radius, min_reference_symbols)
            if corpus is None:
                skipped['short_or_invalid'] += 1
                continue
            frames.append(Frame(str(path), start, channel_number, corpus,
                                samples if retain_samples else None))
    if not frames:
        raise ValueError('no usable CRC-valid frames with independent reference symbols')
    return frames, skipped


def main():
    """Convert requested SigMF recordings and save split channel corpora."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--recordings', type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset-name', default='sigmf_collected')
    parser.add_argument('--split-mode', choices=('by_frame', 'by_channel'), default='by_frame')
    parser.add_argument('--reference-radius', type=int, default=8)
    parser.add_argument('--min-reference-symbols', type=int, default=4)
    args = parser.parse_args()
    if (not args.dataset_name or Path(args.dataset_name).name != args.dataset_name or
            args.dataset_name in ('.', '..') or args.reference_radius < 1 or
            args.min_reference_symbols < 1):
        parser.error('invalid dataset name or reference-window settings')

    from utils.frame_equalizer_channel_ai.dataset_generator.split_sigmf import load_splits
    splits, skipped = load_splits(args.recordings, args.split_mode, args.reference_radius,
                                  args.min_reference_symbols)
    directory = args.dataset_dir / args.dataset_name
    summary = {'dataset': args.dataset_name, 'split_mode': args.split_mode,
               'recordings': str(args.recordings / args.split_mode),
               'reference_radius': args.reference_radius,
               'min_reference_symbols': args.min_reference_symbols,
               'skipped_frames': {name: dict(value) for name, value in skipped.items()}, 'splits': {}}
    for name, selected in splits.items():
        corpus = concatenate_corpora(*(frame.corpus for frame in selected))
        corpus.save(directory / f'{"validation" if name == "val" else name}.npz')
        summary['splits'][name] = {
            'frames': len(selected), 'samples': len(corpus.inputs_q15),
            'channels': {str(channel) if channel is not None else 'unknown': count
                         for channel, count in sorted(Counter(frame.channel for frame in selected).items(),
                                                      key=lambda item: -1 if item[0] is None else item[0])},
            'frame_keys': [f'{frame.source}:{frame.offset}' for frame in selected],
            'digest': corpus_digest(corpus),
        }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'splits'} |
                     {'splits': {name: {key: value for key, value in split.items()
                                       if key != 'frame_keys'}
                                 for name, split in summary['splits'].items()}}, indent=2))


if __name__ == '__main__':
    main()
