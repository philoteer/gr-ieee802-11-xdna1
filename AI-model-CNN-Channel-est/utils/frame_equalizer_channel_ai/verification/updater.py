"""Evaluates an exported MLP channel updater on decoded-symbol recordings."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.frame_equalizer_channel_ai.common.updater import load_cnn_header, load_mlp_header
from utils.frame_equalizer_channel_ai.common.cross_domain import domain_policy
from utils.frame_equalizer_channel_ai.training.updater import (
    CROSS_DOMAIN_MODES, SPLIT_MODES, assemble, infer, infer_float, load_checkpoint,
    load_recording_splits, resolve_paths, score,
)


def main():
    """Loads exported models, assembles examples, and prints MSE metrics.

    The command-line inputs select a pre-split decoded-symbol SigMF collection;
    optional arguments select model headers and reference-window settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--recordings', type=Path, nargs='+')
    parser.add_argument('--split-mode', choices=SPLIT_MODES, default='by_frame')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--cnn-header', type=Path)
    parser.add_argument('--header', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference-radius', type=int, default=8)
    parser.add_argument('--min-reference-symbols', type=int, default=4)
    parser.add_argument('--target-mode', choices=('averaged', 'single_symbol'), default='averaged')
    arguments = parser.parse_args()
    resolve_paths(arguments)
    cnn = load_cnn_header(arguments.cnn_header)
    weights, gate = load_mlp_header(arguments.header)
    splits, counts = load_recording_splits(arguments.recordings, arguments.split_mode,
                                          arguments.reference_radius, arguments.min_reference_symbols)
    frames, skipped = splits[arguments.split], counts[arguments.split]
    data = assemble(frames, cnn, arguments.reference_radius, arguments.min_reference_symbols,
                    arguments.target_mode)
    mse, baseline = score(data, infer(data[0], weights), gate)
    result = {'split_mode': arguments.split_mode,
               'recordings': [str(path / ('by_frame' if arguments.split_mode in CROSS_DOMAIN_MODES
                                          else arguments.split_mode)) for path in arguments.recordings],
              'frames': len(frames), 'samples': len(data[0]), 'skipped': dict(skipped),
                       'target_mode': arguments.target_mode,
                      'gate_snr_ratio_x16': gate, 'null_mse_q15': baseline,
                      'mlp_mse_q15': mse, 'relative_mse': mse / baseline}
    if arguments.split_mode in CROSS_DOMAIN_MODES:
        result['domain_policy'] = domain_policy(arguments.split_mode)
    if arguments.checkpoint.is_file():
        import torch
        torch.set_num_threads(2)
        models, metadata = load_checkpoint(arguments.checkpoint)
        if metadata['split_mode'] != arguments.split_mode:
            raise ValueError('checkpoint split policy does not match --split-mode')
        if arguments.split_mode in CROSS_DOMAIN_MODES and metadata.get('domain_policy') != domain_policy(arguments.split_mode):
            raise ValueError('checkpoint predates source-only cross-domain validation; retrain it')
        if metadata['cnn_header_sha256'] != hashlib.sha256(arguments.cnn_header.read_bytes()).hexdigest():
            raise ValueError('checkpoint was trained against a different CNN baseline')
        float_mse, _ = score(data, infer_float(data[0], models), metadata['cpu_gate_snr_ratio_x16'])
        result['float_mlp_mse_q15'] = float_mse
        result['float_relative_mse'] = float_mse / baseline
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
