"""Training and export pipeline for the CNN channel-residual MLP updater."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.frame_equalizer_channel_ai.common.dataset import snr_gate_candidates
from utils.frame_equalizer_channel_ai.common.cross_domain import (
    SOURCE_TRAIN_PARTITIONS, domain_policy, source_frames, target_frames,
)
from utils.frame_equalizer_channel_ai.common.updater import frame_examples, load_cnn_header, load_mlp_header
from utils.frame_equalizer_channel_ai.dataset_generator.convert_sigmf import (
    DEFAULT_RECORDINGS,
)
from utils.frame_equalizer_channel_ai.dataset_generator.split_sigmf import load_splits
from utils.frame_equalizer_channel_ai.training.exporter import (
    DEFAULT_CPU_DIRECTORY, _initializer, export_trained_updater,
)
from utils.frame_equalizer_channel_ai.training.model import _augment_inputs

CROSS_DOMAIN_MODES = ('bus_to_room', 'room_to_bus')
SPLIT_MODES = ('by_frame', 'by_channel') + CROSS_DOMAIN_MODES


class Updater(nn.Module):
    """Single-group residual MLP with a 61-to-16-to-26 layer layout."""

    def __init__(self):
        """Initializes bias-free dense layers and a hidden ReLU."""
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(61, 16, bias=False), nn.ReLU(),
                                    nn.Linear(16, 26, bias=False))

    def forward(self, values):
        """Predicts the 26 real/imaginary residual components for one group.

        Args:
            values: Tensor whose final dimension contains 61 input features.

        Returns:
            Tensor with the same leading dimensions and 26 output components.
        """
        hidden = self.layers[1](self.layers[0](values)).clamp(max=32767 / 32768)
        return self.layers[2](hidden).clamp(min=-1, max=32767 / 32768)


def quantize(models):
    """Converts four trained group models to the kernel's padded Q7 layout.

    Args:
        models: Four trained ``Updater`` instances, one per 13-bin group.

    Returns:
        A pair of ``int8`` arrays: ``W1`` shaped ``(4, 64, 16)`` and ``W2``
        shaped ``(4, 16, 32)``. Unused input/output lanes are zero-padded.
    """
    first = np.zeros((4, 64, 16), dtype=np.int8)
    second = np.zeros((4, 16, 32), dtype=np.int8)
    for index, model in enumerate(models):
        first[index, :61] = np.clip(np.rint(model.layers[0].weight.detach().numpy().T * 128), -128, 127)
        second[index, :, :26] = np.clip(np.rint(model.layers[2].weight.detach().numpy().T * 128), -128, 127)
    return first, second


def infer(features, weights):
    """Runs the integer Q7 MLP reference on encoded group features.

    Args:
        features: ``int16`` input array shaped ``(N, 4, 61)``.
        weights: Pair of padded Q7 arrays returned by :func:`quantize`.

    Returns:
        ``int16`` residual predictions shaped ``(N, 4, 26)``.
    """
    result = np.zeros((len(features), 4, 26), dtype=np.int16)
    w1, w2 = weights
    for group in range(4):
        inputs = features[:, group].astype(np.int32)
        hidden = np.maximum((inputs @ w1[group, :61].astype(np.int32)) >> 7, 0)
        hidden = np.clip(hidden, 0, 32767).astype(np.int32)
        output = (hidden @ w2[group, :, :26].astype(np.int32)) >> 7
        result[:, group] = np.clip(output, -32768, 32767)
    return result


def infer_float(features, models):
    result = np.empty((len(features), 4, 26), dtype=np.float32)
    with torch.no_grad():
        for group, model in enumerate(models):
            for start in range(0, len(features), 4096):
                inputs = torch.from_numpy(features[start:start + 4096, group].astype(np.float32) / 32768)
                result[start:start + 4096, group] = model(inputs).numpy() * 32768
    return result


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    models = []
    for state in payload['state_dicts']:
        model = Updater()
        model.load_state_dict(state)
        models.append(model.eval())
    if len(models) != 4:
        raise ValueError('updater checkpoint must contain four groups')
    return models, payload['metadata']


def export_models(models, gate, cpu_gate, args):
    weights = quantize(models)
    args.header.parent.mkdir(parents=True, exist_ok=True)
    lines = ['#ifndef FRAME_EQUALIZER_CHANNEL_UPDATER_MLP_WEIGHTS_H',
             '#define FRAME_EQUALIZER_CHANNEL_UPDATER_MLP_WEIGHTS_H', '#include <stdint.h>',
             'namespace frame_equalizer_channel_updater_mlp {',
             f'constexpr uint64_t kGateSnrRatioX16 = {gate}ull;',
             f'alignas(32) static constexpr int8_t kW1[4][64][16] = {_initializer(weights[0])};',
             f'alignas(32) static constexpr int8_t kW2[4][16][32] = {_initializer(weights[1])};',
             '}', '#endif', '']
    args.header.write_text('\n'.join(lines))
    export_trained_updater(models, cpu_gate, args.cpu_header)


def assemble(frames, cnn, radius, minimum, target_mode='averaged'):
    """Concatenates nonempty per-frame examples into one training split.

    Args:
        frames: Frames whose decoded samples are used to build examples.
        cnn: Quantized CNN estimator used for each frame's baseline.
        radius: Maximum reference-symbol radius around candidate pairs.
        minimum: Minimum number of surrounding reference symbols.

    Returns:
        Tuple of concatenated feature, label, baseline, and SNR arrays.

    Raises:
        ValueError: If no frame contributes usable examples.
    """
    parts = [frame_examples(frame, cnn, radius, minimum, target_mode) for frame in frames]
    parts = [part for part in parts if len(part[0])]
    if not parts:
        raise ValueError("split has no data symbols following LTS substitutes")
    return tuple(np.concatenate([part[index] for part in parts]) for index in range(4))


def score(data, prediction, gate):
    """Measures gated residual MSE against the no-updater baseline.

    Args:
        data: Tuple containing features, residual labels, baselines, and SNRs.
        prediction: Predicted residuals shaped ``(N, 4, 26)``.
        gate: SNR threshold; predictions at or above it are suppressed.

    Returns:
        Pair ``(mlp_mse, null_mse)`` in the Q15 residual domain.
    """
    _, labels, _, snr = data
    errors = labels.astype(np.float64).reshape(len(labels), 104)
    active = (snr < gate)[:, None]
    residual = np.where(active, prediction.reshape(len(labels), 104), 0).astype(np.float64)
    return float(np.mean((errors - residual) ** 2)), float(np.mean(errors ** 2))


def load_target_test(directory, radius, minimum):
    selected, skipped = load_splits(directory, 'by_frame', radius, minimum,
                                   retain_samples=True)
    return target_frames(selected), sum(skipped.values(), Counter())


def load_recording_splits(recordings, split_mode, radius, minimum, include_test=True):
    """Load saved partitions, or pool source/target collections for domain runs."""
    if split_mode in CROSS_DOMAIN_MODES:
        if len(recordings) != 2:
            raise ValueError('cross-domain mode requires source then target collection')
        selected, skipped = load_splits(recordings[0], 'by_frame', radius, minimum,
                                       retain_samples=True)
        splits = source_frames(selected)
        counts = {'train': sum((skipped[name] for name in SOURCE_TRAIN_PARTITIONS), Counter()),
                  'val': skipped['val']}
        if include_test:
            splits['test'], counts['test'] = load_target_test(recordings[1], radius, minimum)
        return splits, counts
    splits = {name: [] for name in ('train', 'val', 'test')}
    skipped = {name: Counter() for name in splits}
    for directory in recordings:
        selected, counts = load_splits(directory, split_mode, radius, minimum,
                                      retain_samples=True)
        for name in splits:
            splits[name].extend(selected[name])
            skipped[name].update(counts[name])
    return splits, skipped


def resolve_paths(args):
    """Default all model/artifact paths to the requested split policy."""
    model_dir = ROOT / 'kernel/channel_estimator/model'
    artifacts = Path(__file__).resolve().parent / 'artifacts' / args.split_mode
    defaults = {
        'cnn_header': model_dir / f'frame_equalizer_channel_ai_weights_{args.split_mode}.h',
        'header': model_dir / f'frame_equalizer_channel_updater_mlp_weights_{args.split_mode}.h',
        'cpu_header': DEFAULT_CPU_DIRECTORY / f'channel_updater_mlp_weights_{args.split_mode}.h',
        'report': artifacts / 'updater_report.json',
        'checkpoint': artifacts / 'channel_updater.pt',
    }
    for name, path in defaults.items():
        if getattr(args, name, path) is None:
            setattr(args, name, path)
    if args.recordings is None:
        if args.split_mode in CROSS_DOMAIN_MODES:
            collections = {'room': '20260927_room', 'bus': '20260928_bus'}
            args.recordings = [ROOT / 'data/datasets/channel_estimator' / collections[domain]
                               for domain in args.split_mode.split('_to_')]
        else:
            args.recordings = [DEFAULT_RECORDINGS]


def train(args):
    """Trains four group-specific MLPs, selects an SNR gate, and exports.

    Args:
        args: Parsed training options, including pre-split recordings,
            optimization settings, and output paths.

    Returns:
        None. Writes the generated C++ weights header and JSON training report.
    """
    resolve_paths(args)
    torch.set_num_threads(args.threads)
    if args.export_checkpoint is not None:
        models, metadata = load_checkpoint(args.export_checkpoint)
        if metadata['split_mode'] != args.split_mode:
            raise ValueError('checkpoint split policy does not match --split-mode')
        if args.split_mode in CROSS_DOMAIN_MODES and metadata.get('domain_policy') != domain_policy(args.split_mode):
            raise ValueError('checkpoint predates source-only cross-domain validation; retrain it')
        export_models(models, metadata['gate_snr_ratio_x16'], metadata['cpu_gate_snr_ratio_x16'], args)
        print(args.header)
        print(args.cpu_header)
        return
    if args.noise_std < 0:
        raise ValueError('noise must be nonnegative')
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    cnn = load_cnn_header(args.cnn_header)
    splits, skipped = load_recording_splits(args.recordings, args.split_mode, args.reference_radius,
                                           args.min_reference_symbols,
                                           include_test=args.split_mode not in CROSS_DOMAIN_MODES)
    splits['validation'] = splits.pop('val')
    data = {name: assemble(selected, cnn, args.reference_radius, args.min_reference_symbols,
                           args.target_mode)
            for name, selected in splits.items() if name != 'test'}
    models = []
    training_history = []
    best_epochs = []
    for group in range(4):
        model = Updater()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        x = torch.from_numpy(data['train'][0][:, group].astype(np.float32) / 32768.0)
        y = torch.from_numpy(data['train'][1][:, group].astype(np.float32) / 32768.0)
        vx = torch.from_numpy(data['validation'][0][:, group].astype(np.float32) / 32768.0)
        vy = torch.from_numpy(data['validation'][1][:, group].astype(np.float32) / 32768.0)
        rng = torch.Generator().manual_seed(args.seed + group)
        noise_generator = torch.Generator().manual_seed(args.seed + 4 + group)
        best = float('inf')
        best_weights = deepcopy(model.state_dict())
        stalled = 0
        best_epoch = 0
        history = []
        for epoch in range(args.epochs):
            model.train()
            for indices in torch.randperm(len(x), generator=rng).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                inputs = _augment_inputs(x[indices], args.noise_std, noise_generator, signal_features=60)
                loss = torch.mean((model(inputs) - y[indices]) ** 2)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation = float(torch.mean((model(vx) - vy) ** 2))
            history.append(validation)
            if validation < best - args.min_delta:
                best, best_weights, stalled = validation, deepcopy(model.state_dict()), 0
                best_epoch = epoch + 1
            else:
                stalled += 1
            if stalled == args.patience:
                break
        model.load_state_dict(best_weights)
        models.append(model.eval())
        training_history.append(history)
        best_epochs.append(best_epoch)
    weights = quantize(models)
    validation_prediction = infer(data['validation'][0], weights)
    candidates = snr_gate_candidates()
    gate = min(candidates, key=lambda value: score(data['validation'], validation_prediction, value)[0])
    float_validation = infer_float(data['validation'][0], models)
    cpu_gate = min(candidates, key=lambda value: score(data['validation'], float_validation, value)[0])
    # Only now, with all four models and both gates fixed, read the target domain.
    if args.split_mode in CROSS_DOMAIN_MODES:
        splits['test'], skipped['test'] = load_target_test(
            args.recordings[1], args.reference_radius, args.min_reference_symbols)
    data['test'] = assemble(splits['test'], cnn, args.reference_radius,
                           args.min_reference_symbols, args.target_mode)
    report = {'seed': args.seed, 'epochs': args.epochs, 'patience': args.patience,
              'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
              'noise_std': args.noise_std,
              'min_delta': args.min_delta, 'threads': args.threads,
              'best_epoch_by_group': best_epochs,
              'epochs_trained_by_group': [len(history) for history in training_history],
              'split_mode': args.split_mode,
               'recordings': [str(path / ('by_frame' if args.split_mode in CROSS_DOMAIN_MODES
                                          else args.split_mode)) for path in args.recordings],
              'reference_radius': args.reference_radius,
              'min_reference_symbols': args.min_reference_symbols,
              'target_mode': args.target_mode,
              'cnn_header_sha256': hashlib.sha256(args.cnn_header.read_bytes()).hexdigest(),
              'cnn_header': str(args.cnn_header),
              'data_sha256': {name: hashlib.sha256(b''.join(
                  value.tobytes() for value in values)).hexdigest() for name, values in data.items()},
              'skipped': {name: dict(value) for name, value in skipped.items()},
              'frames': {name: len(value) for name, value in splits.items()},
              'samples': {name: len(value[0]) for name, value in data.items()},
              'gate_snr_ratio_x16': gate, 'cpu_gate_snr_ratio_x16': cpu_gate,
              'validation_loss_by_group': training_history,
              'validation_candidates': {
                  str(value): score(data['validation'], validation_prediction, value)[0] for value in candidates},
              'float_validation_candidates': {
                  str(value): score(data['validation'], float_validation, value)[0] for value in candidates}}
    if args.split_mode in CROSS_DOMAIN_MODES:
        report['domain_policy'] = domain_policy(args.split_mode)
    for name in ('validation', 'test'):
        mse, baseline = score(data[name], infer(data[name][0], weights), gate)
        report[name] = {'null_mse_q15': baseline, 'mlp_mse_q15': mse,
                         'relative_mse': mse / baseline if baseline else 0}
        float_mse, _ = score(data[name], infer_float(data[name][0], models), cpu_gate)
        report[name]['float_mlp_mse_q15'] = float_mse
        report[name]['float_relative_mse'] = float_mse / baseline if baseline else 0
    if args.split_mode not in CROSS_DOMAIN_MODES and args.header.is_file():
        previous, previous_gate = load_mlp_header(args.header)
        mse, baseline = score(data['test'], infer(data['test'][0], previous), previous_gate)
        report['previous_updater_on_current_cnn_test'] = {'mlp_mse_q15': mse, 'relative_mse': mse / baseline}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dicts': [model.state_dict() for model in models], 'metadata': report}, args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    export_models(models, gate, cpu_gate, args)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def parser():
    """Creates the command-line parser for updater training."""
    result = argparse.ArgumentParser()
    result.add_argument('--recordings', type=Path, nargs='+',
                        help='collections with saved partitions; cross-domain: source then target')
    result.add_argument('--cnn-header', type=Path)
    result.add_argument('--header', type=Path)
    result.add_argument('--cpu-header', type=Path)
    result.add_argument('--report', type=Path)
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--export-checkpoint', type=Path)
    result.add_argument('--reference-radius', type=int, default=8)
    result.add_argument('--min-reference-symbols', type=int, default=4)
    result.add_argument('--target-mode', choices=('averaged', 'single_symbol'), default='averaged')
    result.add_argument('--split-mode', choices=SPLIT_MODES, default='by_frame')
    result.add_argument('--seed', type=int, default=40)
    result.add_argument('--epochs', type=int, default=40)
    result.add_argument('--patience', type=int, default=6)
    result.add_argument('--min-delta', type=float, default=1e-7)
    result.add_argument('--batch-size', type=int, default=256)
    result.add_argument('--learning-rate', type=float, default=0.001)
    result.add_argument('--noise-std', type=float, default=0.02)
    result.add_argument('--threads', type=int, default=2)
    return result


if __name__ == '__main__':
    train(parser().parse_args())
