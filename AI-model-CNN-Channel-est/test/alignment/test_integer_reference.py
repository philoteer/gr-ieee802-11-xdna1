"""Validate exported-weight loading and the NumPy-only MLP reference."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.frame_equalizer_channel_ai.common.updater import (
    infer, load_cnn_header, load_mlp_header,
)


class IntegerReferenceTest(unittest.TestCase):
    def test_all_exports_match_scalar_mlp(self):
        features = np.random.default_rng(80211).integers(
            -32768, 32768, (3, 4, 61), dtype=np.int16)
        features[0] = 32767
        features[1] = -32768
        model_dir = ROOT / "kernel/channel_estimator/model"
        for split in ("by_frame", "by_channel", "bus_to_room", "room_to_bus"):
            with self.subTest(split=split):
                cnn = load_cnn_header(model_dir / f"frame_equalizer_channel_ai_weights_{split}.h")
                self.assertEqual([weight.shape for weight in cnn.weights],
                                 [(12, 6, 7), (12, 12, 5), (2, 12, 3)])
                weights, _ = load_mlp_header(
                    model_dir / f"frame_equalizer_channel_updater_mlp_weights_{split}.h")
                expected = np.empty((3, 4, 26), dtype=np.int16)
                for sample in range(3):
                    for group in range(4):
                        hidden = [min(32767, max(0, sum(
                            int(features[sample, group, i]) * int(weights[0][group, i, j])
                            for i in range(61)) >> 7)) for j in range(16)]
                        for output in range(26):
                            value = sum(hidden[i] * int(weights[1][group, i, output])
                                        for i in range(16)) >> 7
                            expected[sample, group, output] = min(32767, max(-32768, value))
                np.testing.assert_array_equal(infer(features, weights), expected)


if __name__ == "__main__":
    unittest.main()
