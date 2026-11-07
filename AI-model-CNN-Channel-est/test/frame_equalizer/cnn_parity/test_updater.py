import argparse
from pathlib import Path
import sys

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_parity import make_lts
from utils.frame_equalizer_channel_ai.common.dataset import (
    ChannelCorpus, LONG, OCCUPIED, _shared_q15, average_lts,
    estimate_snr_ratio_x16, snr_feature_q15,
)
from utils.frame_equalizer_channel_ai.common.updater import (
    OCCUPIED_BINS, PILOT_BINS, infer, load_cnn_header, load_mlp_header, normalize,
)
from utils.frame_equalizer_channel_ai.verification.quantized import integer_estimate


def expected(symbols, cnn, weights, gate, number=2):
    lts = symbols[:2]
    corrected = lts * LONG[None, :, None]
    inputs_four, shifts = _shared_q15(corrected[None])
    snr = estimate_snr_ratio_x16(lts[None])
    inputs = np.empty((1, 6, 64), dtype=np.int32)
    inputs[:, :4] = inputs_four.transpose(0, 1, 3, 2).reshape(1, 4, 64)
    inputs[:, 4] = OCCUPIED.astype(np.int32) * 32767
    inputs[:, 5] = snr_feature_q15(snr)[:, None]
    ls = average_lts(corrected[0], corrected[1])[None]
    corpus = ChannelCorpus(inputs, lts[None], snr, ls, ls.copy(),
                           np.zeros_like(ls), shifts)
    baseline = integer_estimate(cnn, corpus)[0]
    if int(snr[0]) >= gate:
        return baseline
    pilot = symbols[number, PILOT_BINS].copy()
    pilot *= 1 if number < 6 else -1
    pilot[3] *= -1
    magnitude = np.max(np.abs(np.concatenate((baseline[OCCUPIED_BINS].reshape(-1), pilot.reshape(-1))).astype(np.int64)))
    shift = 15 - int(magnitude).bit_length()
    base_q15 = normalize(baseline[None, OCCUPIED_BINS], np.array([shift]))[0]
    pilot_q15 = normalize(pilot[None], np.array([shift]))[0]
    features = np.zeros((1, 4, 61), dtype=np.int16)
    for group in range(4):
        features[0, group, :52] = base_q15[(group // 2) * 26:(group // 2 + 1) * 26].reshape(52)
        delta = pilot_q15.astype(np.int32) - base_q15[np.searchsorted(OCCUPIED_BINS, PILOT_BINS)].astype(np.int32)
        features[0, group, 52:60] = np.clip(delta, -32768, 32767).reshape(8)
        features[0, group, 60] = int(snr_feature_q15(snr)[0])
    prediction = infer(features, weights)[0].reshape(52, 2).astype(np.int32)
    correction = prediction >> shift if shift >= 0 else prediction * (1 << -shift)
    result = baseline.copy()
    result[OCCUPIED_BINS] = np.clip(result[OCCUPIED_BINS].astype(np.int64) + correction,
                                    -2147483648, 2147483647).astype(np.int32)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xclbin', type=Path, required=True)
    parser.add_argument('--insts', type=Path, required=True)
    parser.add_argument('--split-mode', choices=('by_frame', 'by_channel', 'bus_to_room', 'room_to_bus'), default='by_frame')
    args = parser.parse_args()
    cnn = load_cnn_header(ROOT / f'kernel/channel_estimator/model/frame_equalizer_channel_ai_weights_{args.split_mode}.h')
    weights, gate = load_mlp_header(ROOT / f'kernel/channel_estimator/model/frame_equalizer_channel_updater_mlp_weights_{args.split_mode}.h')
    handle = DefaultNPURuntime.load(NPUKernel(str(args.xclbin), str(args.insts), kernel_name='MLIR_AIE'))
    rng = np.random.default_rng(8021141)
    offsets = np.array([[1900, -1200], [-800, 2100], [1500, 800], [1100, -1600]])
    # A newly trained ReLU network can output zero on one fixed pilot profile.
    # Keep deterministic alternatives so active-update coverage survives retraining.
    profiles = [offsets, -offsets] + [rng.integers(-4000, 4001, (4, 2)) for _ in range(16)]
    for high_snr, scale in ((False, 1), (False, 4), (False, 0.25), (True, 1)):
        lts = np.rint(make_lts() * scale).astype(np.int32)
        if high_snr:
            lts[1] = lts[0]
        symbols = np.zeros((7, 64, 2), dtype=np.int32)
        symbols[:2] = lts
        baseline = expected(symbols, cnn, weights, 0)
        for offsets in profiles:
            for number in range(2, 7):
                symbols[number, PILOT_BINS] = baseline[PILOT_BINS] + offsets * (number - 1)
                symbols[number, 53] *= -1
                if number == 6:
                    symbols[number, PILOT_BINS] *= -1
            reference = np.stack((expected(symbols, cnn, weights, gate, 2),
                                  expected(symbols, cnn, weights, gate, 6)))
            if high_snr or not np.array_equal(reference[0], baseline):
                break
        if not high_snr and np.array_equal(reference[0], baseline):
            raise AssertionError('low-SNR example did not exercise the MLP')
        if high_snr and not np.array_equal(reference[0], baseline):
            raise AssertionError('high-SNR example did not exercise the gate')
        output = iron.zeros(2 * 64 * 2, dtype=np.int32)
        DefaultNPURuntime.run(handle, [iron.tensor(symbols.reshape(-1), dtype=np.int32), output])
        actual = output.numpy().reshape(2, 64, 2)
        if not np.array_equal(actual, reference):
            mismatch = np.argwhere(actual != reference)
            raise AssertionError(f"updater parity: {len(mismatch)} mismatches, first {mismatch[:8].tolist()} "
                                 f"actual {actual[tuple(mismatch[0])]}, expected {reference[tuple(mismatch[0])]}")
    print('MLP updater parity PASS: repeated symbols, pilot polarity and SNR gate match Python exactly')


if __name__ == '__main__':
    main()
