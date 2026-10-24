#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.frame_equalizer_channel_ai.common.corpora import DEFAULT_DATASET_DIRECTORY, load_corpus
from utils.frame_equalizer_channel_ai.common.dataset import LONG, OCCUPIED
from utils.frame_equalizer_channel_ai.common.updater import load_cnn_header
from utils.frame_equalizer_channel_ai.verification.quantized import integer_estimate


def main():
    parser = argparse.ArgumentParser(description="Compare compiled CPU CNN and deployed Q7 CNN on held-out corpora")
    parser.add_argument('--dataset', default='both')
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    parser.add_argument('--runner', type=Path, default=ROOT.parent.parent / '_gr-ieee802-11/build/lib/qa_channel_ai')
    parser.add_argument('--cnn-header', type=Path, default=ROOT / 'kernel/channel_estimator/model/frame_equalizer_channel_ai_weights_by_frame.h')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'utils/frame_equalizer_channel_ai/training/artifacts/channel_estimator.pt')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    corpus = load_corpus(args.dataset_dir, args.dataset, args.split)
    corrected = corpus.observations.astype(np.float32) * LONG[None, None, :, None]
    values = corrected[..., 0] + 1j * corrected[..., 1]
    cases = []
    for pair, ratio in zip(values, corpus.snr_ratio_x16):
        pilots = (pair[0, (11, 25, 39, 53)] + pair[1, (11, 25, 39, 53)]) * 0.5
        inputs = np.r_[pair.ravel(), pilots]
        cases.append(f'{int(ratio)} 2\n' + ' '.join(f'{v:.9g}' for v in np.stack((inputs.real, inputs.imag), axis=-1).ravel()))
    result = subprocess.run([str(args.runner)], input=str(len(cases)) + '\n' + '\n'.join(cases) + '\n',
                            check=True, capture_output=True, text=True)
    cpu = np.fromstring(result.stdout, sep=' ').reshape(len(cases), -1, 2)[:, :64]
    model = load_cnn_header(args.cnn_header)
    quantized = integer_estimate(model, corpus)
    truth = corpus.truth[:, OCCUPIED].astype(np.float64)
    power = np.sum(truth**2, axis=(1, 2))

    def metrics(estimate):
        error = estimate[:, OCCUPIED].astype(np.float64) - truth
        nmse = np.sum(error**2, axis=(1, 2)) / np.maximum(power, 1e-30)
        channel = estimate[:, OCCUPIED, 0] + 1j * estimate[:, OCCUPIED, 1]
        reference = truth[..., 0] + 1j * truth[..., 1]
        evm = np.sqrt(np.mean(np.abs(np.divide(reference, channel,
            out=np.zeros_like(reference), where=np.abs(channel) > 0) - 1)**2))
        return {'mean_nmse': float(np.mean(nmse)), 'mean_nmse_db': float(10 * np.log10(np.mean(nmse))),
                'channel_equalization_evm_percent': float(100 * evm)}

    report = {'dataset': args.dataset, 'split': args.split, 'samples': len(cases),
              'ls': metrics(corpus.ls), 'aie_quantized_cnn': metrics(quantized),
              'cpu_float_cnn': metrics(cpu),
              'float_vs_quantized_relative_rms': float(np.sqrt(
                  np.sum((cpu[:, OCCUPIED] - quantized[:, OCCUPIED])**2) /
                  np.sum(quantized[:, OCCUPIED].astype(np.float64)**2)))}
    if args.checkpoint.is_file():
        import torch
        from utils.frame_equalizer_channel_ai.common.model import ChannelEstimator, float_estimate
        torch.set_num_threads(2)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        floating = ChannelEstimator()
        floating.load_state_dict(payload['state_dict'])
        reference = float_estimate(floating.eval(), corpus)
        gated = corpus.snr_ratio_x16 >= payload['metadata']['cpu_gate_snr_ratio_x16']
        reference[gated] = corpus.ls[gated]
        report['checkpoint_float_cnn'] = metrics(reference)
        relative = float(np.sqrt(np.sum((cpu[:, OCCUPIED] - reference[:, OCCUPIED])**2) /
                                 np.sum(reference[:, OCCUPIED].astype(np.float64)**2)))
        report['cpu_vs_checkpoint_relative_rms'] = relative
        if relative > 2e-4:
            raise AssertionError(f'compiled CPU/checkpoint mismatch: {relative}')
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
