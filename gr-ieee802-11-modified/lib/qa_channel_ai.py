#!/usr/bin/env python3
from pathlib import Path
import ast
import re
import subprocess
import sys

import numpy as np

OCCUPIED = np.r_[6:32, 33:59]
PILOTS = np.array([11, 25, 39, 53])
QMAX = np.float32(32767 / 32768)


def load(path, shapes):
    source = path.read_text()
    weights = []
    for index, shape in enumerate(shapes, 1):
        values = re.search(rf'kW{index}\[\d+\] = \{{(.*?)\}};', source, re.S)[1]
        weights.append(np.array([float(v.rstrip('f')) for v in values.replace('\n', '').split(',')],
                                dtype=np.float32).reshape(shape))
    gate = int(re.search(r'kGateSnrRatioX16 = (\d+)', source)[1])
    return weights, gate


def scale_of(values):
    maximum = float(np.max(np.abs(np.stack([values.real, values.imag]))))
    return 2.0 ** np.frexp(maximum)[1] if maximum else 1.0


def load_int8(split, kind):
    directory = Path(__file__).resolve().parents[2] / 'AI-model-CNN-Channel-est/kernel/channel_estimator/model'
    stem = 'ai' if kind == 'cnn' else 'updater_mlp'
    text = (directory / f'frame_equalizer_channel_{stem}_weights_{split}.h').read_text()
    weights = [np.array(ast.literal_eval(body.replace('{', '[').replace('}', ']')), dtype=np.int64)
               for body in re.findall(r'int8_t kW\d+\[.*?= (\{.*?\});', text, re.S)]
    if kind == 'mlp':
        weights = [weights[0][:, :61, :16].transpose(0, 2, 1),
                   weights[1][:, :16, :26].transpose(0, 2, 1)]
    return weights, int(re.search(r'k(?:Ls)?GateSnrRatioX16 = (\d+)', text)[1])


def q15(x):
    return np.floor(np.clip(np.asarray(x) * 32768, -32768, 32767)).astype(np.int64)


def feature(ratio):
    if ratio <= 32:
        return 0
    if ratio >= 320000:
        return QMAX
    raw = int(ratio)
    exponent = raw.bit_length() - 1
    mantissa = raw >> (exponent - 8) if exponent >= 8 else raw << (8 - exponent)
    return min(32767, (exponent * 256 + mantissa - 256 - 1280) * 32767 // 3402) / 32768


def refine(first, second, ratio, weights, gate, integer=False):
    estimate = np.zeros(64, dtype=np.complex64)
    estimate[OCCUPIED] = (first[OCCUPIED] * 0.5 + second[OCCUPIED] * 0.5)
    if ratio >= gate or not np.any(np.r_[first[OCCUPIED], second[OCCUPIED]]):
        return estimate
    scale = scale_of(np.r_[first[OCCUPIED], second[OCCUPIED]])
    x = np.zeros((6, 64), dtype=np.float32)
    x[0, OCCUPIED], x[1, OCCUPIED] = first[OCCUPIED].real / scale, first[OCCUPIED].imag / scale
    x[2, OCCUPIED], x[3, OCCUPIED] = second[OCCUPIED].real / scale, second[OCCUPIED].imag / scale
    x[4, OCCUPIED] = QMAX
    x[5] = feature(ratio)
    if integer:
        x = q15(x)
    for layer, w in enumerate(weights):
        taps = w.shape[-1]
        patches = np.lib.stride_tricks.sliding_window_view(np.pad(x, ((0, 0), (taps // 2, taps // 2))), taps, axis=1)
        x = np.einsum('oik,ibk->ob', w, patches, optimize=True)
        if integer:
            x = np.clip(x >> 7, -2147483648, 2147483647)
        if layer < 2:
            x = np.maximum(x, 0)
    estimate[OCCUPIED] += (x[0, OCCUPIED] + 1j * x[1, OCCUPIED]) * (scale / 32768 if integer else scale)
    return estimate


def update(baseline, pilots, ratio, weights, gate, integer=False):
    result = baseline.copy()
    if ratio >= gate:
        return result
    scale = scale_of(np.r_[baseline[OCCUPIED], pilots])
    features = np.zeros(61, dtype=np.float32)
    delta = (pilots - baseline[PILOTS]) / scale
    features[52:60] = np.clip(np.stack([delta.real, delta.imag], axis=-1).ravel(), -1, QMAX)
    features[60] = feature(ratio)
    if integer:
        features = q15(features)
        p, b = pilots / scale, baseline[PILOTS] / scale
        features[52:60] = np.clip(q15(np.stack([p.real, p.imag], axis=-1)).ravel() -
                                      q15(np.stack([b.real, b.imag], axis=-1)).ravel(), -32768, 32767)
    for group in range(4):
        half = OCCUPIED[(group // 2) * 26:(group // 2 + 1) * 26]
        normalized = baseline[half] / scale
        values = np.stack([normalized.real, normalized.imag], axis=-1).ravel()
        features[:52] = q15(values) if integer else values
        if integer:
            hidden = np.clip((weights[0][group] @ features) >> 7, 0, 32767)
            residual = (np.clip((weights[1][group] @ hidden) >> 7, -32768, 32767) / 32768).reshape(13, 2)
        else:
            hidden = np.clip(weights[0][group] @ features, 0, QMAX)
            residual = np.clip(weights[1][group] @ hidden, -1, QMAX).reshape(13, 2)
        bins = OCCUPIED[group * 13:(group + 1) * 13]
        result[bins] += (residual[:, 0] + 1j * residual[:, 1]) * scale
    return result


def main():
    directory = Path(__file__).parent / 'equalizer/model'
    selection = sys.argv[2] if len(sys.argv) > 2 else 'by_frame'
    suffix = '_' + selection
    cnn, cnn_gate = load(directory / f'channel_cnn_weights{suffix}.h', [(12, 6, 7), (12, 12, 5), (2, 12, 3)])
    mlp, mlp_gate = load(directory / f'channel_updater_mlp_weights{suffix}.h', [(4, 16, 61), (4, 26, 16)])
    integer = '--int8' in sys.argv
    if integer:
        cnn, cnn_gate = load_int8(selection, 'cnn')
        mlp, mlp_gate = load_int8(selection, 'mlp')
    rng = np.random.default_rng(43)
    cases, expected = [], []
    # UINT64_MAX disables gating; its +/-1 integer boundaries cannot be
    # represented by the C++ double SNR interface. Exercise finite gates only.
    boundaries = [gate + delta for gate in (cnn_gate, mlp_gate) if gate < 2**53
                  for delta in (-1, 0, 1) if gate + delta >= 0]
    for amplitude in [0, 2**-20, 0.5, 1, 2, 2**20]:
        for ratio in sorted(set([0, 32, 64, 1600, 320000] + boundaries)):
            first = (rng.normal(size=64) + 1j * rng.normal(size=64)).astype(np.complex64) * amplitude
            second = (rng.normal(size=64) + 1j * rng.normal(size=64)).astype(np.complex64) * amplitude
            pilots = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(np.complex64) * amplitude
            # Make the component maximum cross exact power-of-two boundaries.
            if amplitude and ratio == 64:
                first[:] = amplitude
                second[:] = amplitude * 0.75
            number = [2, 3, 128, 129, 256][len(cases) % 5]
            pair = np.r_[first, second, pilots]
            cases.append(f'{ratio} {number}\n' + ' '.join(f'{v:.9g}' for v in np.stack([pair.real, pair.imag], axis=-1).ravel()))
            baseline = refine(first, second, ratio, cnn, cnn_gate, integer)
            corrected = update(baseline, pilots, ratio, mlp, mlp_gate, integer)
            signal = np.sum(np.abs(first[OCCUPIED].astype(np.complex128) + second[OCCUPIED])**2)
            noise = np.sum(np.abs(first[OCCUPIED].astype(np.complex128) - second[OCCUPIED])**2)
            actual_ratio = 16 * signal / noise if noise else (np.inf if signal else 0)
            actual_baseline = refine(first, second, actual_ratio, cnn, cnn_gate, integer)
            equalizer = update(actual_baseline, pilots, actual_ratio, mlp, mlp_gate, integer)[OCCUPIED]
            expected.append(np.r_[baseline, corrected, equalizer])
    result = subprocess.run([sys.argv[1], selection] + (['--int8'] if integer else []), input=str(len(cases)) + '\n' + '\n'.join(cases) + '\n',
                            text=True, check=True, capture_output=True)
    actual = np.fromstring(result.stdout, sep=' ').reshape(len(cases), -1, 2)
    actual = actual[:, :, 0] + 1j * actual[:, :, 1]
    for index, (got, want) in enumerate(zip(actual, expected)):
        amplitude = max(1e-20, float(np.max(np.abs(want))))
        np.testing.assert_allclose(got / amplitude, want / amplitude, rtol=2e-5, atol=2e-6,
                                   err_msg=f'case {index}')
    print(f'{len(cases)} CNN/MLP/reference cases passed; reset, constellations, polarity, and Eigen allocation checks passed')


if __name__ == '__main__':
    main()
