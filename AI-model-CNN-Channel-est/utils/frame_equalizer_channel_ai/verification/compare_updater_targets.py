"""Train and compare updater supervision on saved channel-held-out splits."""

from contextlib import redirect_stdout
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.frame_equalizer_channel_ai.common.updater import load_cnn_header, frame_examples, PILOT_BINS
from utils.frame_equalizer_channel_ai.common.dataset import usable_pair_indices
from utils.frame_equalizer_channel_ai.dataset_generator.split_sigmf import load_splits
from utils.frame_equalizer_channel_ai.training import updater


def metrics(data, prediction, gate):
    labels = data[1].astype(np.float64)
    correction = np.where((data[3] < gate)[:, None, None], prediction, 0)
    error = labels - correction
    baseline = labels ** 2
    occupied = float(np.mean(error ** 2))
    null = float(np.mean(baseline))
    occupied_bins = np.r_[np.arange(6, 32), np.arange(33, 59)]
    data_mask = ~np.isin(occupied_bins, (11, 25, 39, 53))
    error = error.reshape(len(error), 52, 2)[:, data_mask]
    baseline = baseline.reshape(len(baseline), 52, 2)[:, data_mask]
    data_mse = float(np.mean(error ** 2))
    data_null = float(np.mean(baseline))
    return {'mse_q15': occupied, 'baseline_mse_q15': null,
            'relative_mse': occupied / null if null else 0,
            'data_mse_q15': data_mse, 'data_baseline_mse_q15': data_null,
            'data_relative_mse': data_mse / data_null if data_null else 0,
            'active_fraction': float(np.mean(data[3] < gate))}


def symbol_decisions(frames, cnn, models, reports):
    occupied_bins = np.r_[np.arange(6, 32), np.arange(33, 59)]
    data_mask = ~np.isin(occupied_bins, PILOT_BINS)
    constellations = [np.asarray([-1, 1], dtype=np.complex128)]
    for levels, power in (([-1, 1], 2), ([-3, -1, 1, 3], 10),
                          ([-7, -5, -3, -1, 1, 3, 5, 7], 42)):
        constellations.append(np.asarray([r + 1j * i for r in levels for i in levels]) / np.sqrt(power))
    counts = {name: {'errors': 0, 'symbols': 0} for name in ('baseline', *models)}
    weights = {mode: updater.quantize(model) for mode, model in models.items()}
    for frame in frames:
        features, _, baseline, snr = frame_examples(frame, cnn, 8, 4)
        if not len(features):
            continue
        indices = []
        for index in usable_pair_indices(len(frame.samples), 8, 4):
            current = index + 2
            if current >= len(frame.samples):
                continue
            window = np.arange(max(0, current - 8), min(len(frame.samples), current + 9))
            if np.count_nonzero((window < index) | (window > current)) >= 4:
                indices.append(current)
        received = frame.samples[:, :, 0]
        transmitted = frame.samples[:, :, -1]
        channel = received / transmitted
        scale = min(12000.0 / float(np.median(np.abs(channel))),
                    30000.0 / float(np.max(np.abs(channel))))
        current = channel[indices]
        pilots = np.rint(np.stack((current[:, np.searchsorted(occupied_bins, PILOT_BINS)].real,
                                  current[:, np.searchsorted(occupied_bins, PILOT_BINS)].imag), axis=-1) * scale)
        maximum = np.maximum(np.max(np.abs(baseline.astype(np.int64)), axis=(1, 2)),
                             np.max(np.abs(pilots), axis=(1, 2)))
        shifts = 15 - np.asarray([int(value).bit_length() for value in maximum])
        reference = transmitted[indices][:, data_mask]
        probe = reference.ravel()[:4096]
        constellation = min(constellations, key=lambda values:
                            np.mean(np.min(np.abs(probe[:, None] - values) ** 2, axis=1)))
        expected = np.argmin(np.abs(reference[:, :, None] - constellation) ** 2, axis=2)
        estimates = {'baseline': baseline.astype(np.float64)}
        for mode in models:
            prediction = updater.infer(features, weights[mode]).reshape(len(features), 52, 2)
            active = snr < reports[mode]['gate_snr_ratio_x16']
            correction = np.where(active[:, None, None], prediction, 0).astype(np.int64)
            for row, shift in enumerate(shifts):
                correction[row] = correction[row] >> int(shift) if shift >= 0 else correction[row] << int(-shift)
            estimates[mode] = baseline + correction
        for mode, estimate in estimates.items():
            response = (estimate[:, :, 0] + 1j * estimate[:, :, 1])[:, data_mask] / scale
            equalized = received[indices][:, data_mask] / np.where(np.abs(response) > 1e-12, response, 1e-12)
            actual = np.argmin(np.abs(equalized[:, :, None] - constellation) ** 2, axis=2)
            counts[mode]['errors'] += int(np.count_nonzero(actual != expected))
            counts[mode]['symbols'] += reference.size
    for count in counts.values():
        count['ser'] = count['errors'] / count['symbols'] if count['symbols'] else 0
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--evaluate-only', action='store_true')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'split_mode': 'by_channel', 'seed': 40,
              'epochs': arguments.epochs, 'reference_radius': 8,
              'min_reference_symbols': 4, 'collections': {}}
    for name, collection in (('room', '20260927_room'), ('bus', '20260928_bus')):
        recordings = ROOT / 'data/datasets/channel_estimator' / collection
        models = {}
        reports = {}
        for mode in ('single_symbol', 'averaged'):
            directory = arguments.output_dir / name / mode
            directory.mkdir(parents=True, exist_ok=True)
            args = updater.parser().parse_args([
                '--recordings', str(recordings), '--split-mode', 'by_channel',
                '--target-mode', mode, '--epochs', str(arguments.epochs),
                '--checkpoint', str(directory / 'updater.pt'),
                '--report', str(directory / 'training.json'),
                '--header', str(directory / 'npu_weights.h'),
                 '--cpu-header', str(directory / 'cpu_weights.h')])
            updater.resolve_paths(args)
            if not arguments.evaluate_only:
                with (directory / 'training.log').open('w') as log, redirect_stdout(log):
                    updater.train(args)
            models[mode], reports[mode] = updater.load_checkpoint(args.checkpoint)
            print(f'{name}: trained {mode}', flush=True)
        cnn = load_cnn_header(args.cnn_header)
        splits, _ = load_splits(recordings, 'by_channel', 8, 4, retain_samples=True)
        evaluation = {}
        for target in ('single_symbol', 'averaged'):
            evaluation[target] = {}
            for split in ('val', 'test'):
                data = updater.assemble(splits[split], cnn, 8, 4, target)
                comparisons = {}
                for mode in models:
                    weights = updater.quantize(models[mode])
                    integer = updater.infer(data[0], weights)
                    floating = updater.infer_float(data[0], models[mode])
                    gate = reports[mode]['gate_snr_ratio_x16']
                    cpu_gate = reports[mode]['cpu_gate_snr_ratio_x16']
                    comparisons[mode] = {
                        'integer': metrics(data, integer, gate),
                        'float': metrics(data, floating, cpu_gate),
                        'ungated_integer': metrics(data, integer, 2**64 - 1)}
                evaluation[target][split] = {'samples': len(data[0]), 'models': comparisons}
        result['collections'][name] = {
            'training': {mode: {key: report[key] for key in (
                'samples', 'gate_snr_ratio_x16', 'cpu_gate_snr_ratio_x16',
                'cnn_header_sha256', 'learning_rate', 'batch_size', 'noise_std')}
                         for mode, report in reports.items()},
            'evaluation': evaluation}
        result['collections'][name]['test_symbol_decisions'] = symbol_decisions(
            splits['test'], cnn, models, reports)
        (arguments.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'{name}: evaluated both models on both target definitions', flush=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
