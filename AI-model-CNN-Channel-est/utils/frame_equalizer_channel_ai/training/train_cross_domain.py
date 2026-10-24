#!/usr/bin/env python3
"""Train bus→room and room→bus CNN/MLP pairs with source-only model selection.

Use saved source train+test for fitting and source val for epochs/gates.
The entire target domain is reserved for final evaluation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.frame_equalizer_channel_ai.common.dataset import concatenate_corpora, corpus_digest
from utils.frame_equalizer_channel_ai.common.cross_domain import domain_policy, source_frames, target_frames
from utils.frame_equalizer_channel_ai.dataset_generator.split_sigmf import load_splits


def train_cnns():
    package = ROOT / 'utils/frame_equalizer_channel_ai'
    datasets = ROOT / 'data/datasets/channel_estimator'
    domains = {}
    provenance = {}
    for domain, collection in (('room', '20260927_room'), ('bus', '20260928_bus')):
        splits, skipped = load_splits(datasets / collection, 'by_frame', 8, 4)
        domains[domain] = splits
        provenance[domain] = {
            'collection': collection,
            'partitions': {name: {'frames': len(frames),
                                 'samples': sum(len(frame.corpus.inputs_q15) for frame in frames)}
                           for name, frames in splits.items()},
            'skipped': {key: dict(value) for key, value in skipped.items()},
        }
    for source, target in (('bus', 'room'), ('room', 'bus')):
        name = f'{source}_to_{target}'
        directory = datasets / name
        selected = source_frames(domains[source])
        selected['test'] = target_frames(domains[target])
        manifest = {'domain_policy': domain_policy(name), 'domains': provenance,
                    'reference_radius': 8, 'min_reference_symbols': 4, 'splits': {}}
        for split, frames in selected.items():
            corpus = concatenate_corpora(*(frame.corpus for frame in frames))
            corpus.save(directory / f'{"validation" if split == "val" else split}.npz')
            manifest['splits'][split] = {
                'frames': len(frames), 'samples': len(corpus.inputs_q15),
                'digest': corpus_digest(corpus),
                'frame_keys': [f'{frame.source}:{frame.offset}' for frame in frames],
            }
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        artifacts = package / 'training/artifacts' / name
        subprocess.run([
            sys.executable, str(package / 'training/cli.py'), 'train',
            '--dataset', name, '--checkpoint', str(artifacts / 'channel_estimator.pt'),
            '--domain-manifest', str(directory / 'manifest.json'),
            '--report', str(artifacts / 'channel_estimator_report.json'),
            '--artifact-prefix', str(artifacts / 'channel_estimator_weights'),
            '--header', str(ROOT / f'kernel/channel_estimator/model/frame_equalizer_channel_ai_weights_{name}.h'),
            '--cpu-header', str(ROOT.parent / f'gr-ieee802-11-modified/lib/equalizer/model/channel_cnn_weights_{name}.h'),
        ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updater-only', action='store_true',
                        help='reuse saved CNNs and train only matching MLP updaters')
    args = parser.parse_args()
    if not args.updater_only:
        train_cnns()
    for mode in ('bus_to_room', 'room_to_bus'):
        # Reusing a CNN selected on the target would taint the MLP experiment too.
        checkpoint = ROOT / f'utils/frame_equalizer_channel_ai/training/artifacts/{mode}/channel_estimator.pt'
        metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)['metadata']
        if metadata.get('domain_policy') != domain_policy(mode):
            parser.error(f'{mode}: retrain CNNs with source-only validation before --updater-only')
        subprocess.run([
            sys.executable, str(ROOT / 'utils/frame_equalizer_channel_ai/training/updater.py'),
            '--split-mode', mode,
        ], check=True)
    subprocess.run([sys.executable, str(ROOT.parent /
                    'gr-ieee802-11-modified/utils/export_channel_int8.py')], check=True)


if __name__ == '__main__':
    main()
