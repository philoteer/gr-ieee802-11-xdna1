#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.frame_equalizer_channel_ai.common.corpora import DEFAULT_DATASET_DIRECTORY
from utils.frame_equalizer_channel_ai.common.updater import example_indices
from utils.frame_equalizer_channel_ai.dataset_generator.split_sigmf import load_splits


DATASETS = {
    'room': ('20260927_room', None),
    'bus_206': ('20260928_bus', '206'),
    'bus_207': ('20260928_bus', '207'),
}


def count_samples(recordings: Path, split_mode: str = 'by_frame',
                  reference_radius: int = 8, min_reference_symbols: int = 4,
                  target_mode: str = 'averaged', model: str = 'both',
                  recording: str | None = None) -> dict:
    if reference_radius < 1 or min_reference_symbols < 1:
        raise ValueError('reference-window settings must be positive')
    if model not in ('cnn', 'mlp', 'both'):
        raise ValueError('unknown model')
    if target_mode not in ('averaged', 'single_symbol'):
        raise ValueError('unknown updater target mode')
    splits, skipped = load_splits(recordings, split_mode, reference_radius,
                                  min_reference_symbols, retain_samples=model != 'cnn')
    models = ('cnn', 'mlp') if model == 'both' else (model,)
    counts = {name: {} for name in models}
    frames = {name: {} for name in models}
    for split, selected in splits.items():
        name = 'validation' if split == 'val' else split
        if recording is not None:
            selected = [frame for frame in selected if Path(frame.source).stem == recording]
        for kind in models:
            sizes = []
            for frame in selected:
                if kind == 'cnn':
                    sizes.append(len(frame.corpus.inputs_q15))
                else:
                    _, indices, _ = example_indices(
                        len(frame.samples), reference_radius, min_reference_symbols, target_mode)
                    sizes.append(len(indices))
            counts[kind][name] = sum(sizes)
            frames[kind][name] = sum(size > 0 for size in sizes)
    return {'recordings': str(recordings / split_mode), 'recording': recording,
            'split_mode': split_mode, 'reference_radius': reference_radius,
            'min_reference_symbols': min_reference_symbols, 'target_mode': target_mode,
            'samples': counts, 'frames': frames,
            'skipped_frames_collection': {
                'validation' if name == 'val' else name: dict(value)
                for name, value in skipped.items()}}


def parser():
    result = argparse.ArgumentParser(description='Count prepared CNN/MLP samples in saved SigMF splits.')
    result.add_argument('--dataset', choices=tuple(DATASETS), default='room')
    result.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIRECTORY)
    result.add_argument('--recordings', type=Path)
    result.add_argument('--model', choices=('cnn', 'mlp', 'both'), default='both')
    result.add_argument('--split-mode', choices=('by_frame', 'by_channel'), default='by_frame')
    result.add_argument('--reference-radius', type=int, default=8)
    result.add_argument('--min-reference-symbols', type=int, default=4)
    result.add_argument('--target-mode', choices=('averaged', 'single_symbol'), default='averaged')
    result.add_argument('--report', type=Path)
    return result


def main():
    command = parser()
    args = command.parse_args()
    if args.reference_radius < 1 or args.min_reference_symbols < 1:
        command.error('reference-window settings must be positive')
    collection, recording = DATASETS[args.dataset]
    summary = count_samples(
        args.recordings if args.recordings is not None else args.dataset_dir / collection,
        args.split_mode, args.reference_radius, args.min_reference_symbols,
        args.target_mode, args.model, recording)
    summary['dataset'] = args.dataset
    output = json.dumps(summary, indent=2) + '\n'
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
