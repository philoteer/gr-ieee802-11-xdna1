#!/usr/bin/env python3
"""Train matching CNN/MLP pairs on existing frame/channel-split datasets.

This wrapper does not create or change dataset partitions. It converts the
saved room/bus SigMF train/validation/test partitions to trainer-ready NPZ
files, then trains a CNN and its MLP updater for each split policy. Each run
exports separate NPU and GNU Radio weights, checkpoints, and training reports.
The MLP always uses its matching CNN to generate baseline channel estimates.
"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "utils/frame_equalizer_channel_ai"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updater-only', action='store_true',
                        help='reuse the saved CNNs and train only their matching MLP updaters')
    args = parser.parse_args()
    # These select existing partition directories, not new splitting operations.
    for mode in ("by_frame", "by_channel"):
        names = []
        recordings = []
        for collection in ("20260927_room", "20260928_bus"):
            name = f"{collection}_{mode}"
            names.append(name)
            recordings.append(str(ROOT / "data/datasets/channel_estimator" / collection))
            if args.updater_only:
                continue
            # Convert the saved partitions to NPZ without reassigning frames
            # or channels. Existing converted NPZ files are regenerated.
            subprocess.run([
                sys.executable, str(PACKAGE / "dataset_generator/convert_sigmf.py"),
                "--recordings", str(ROOT / "data/datasets/channel_estimator" / collection),
                "--split-mode", mode, "--dataset-dir", str(ROOT / "data/datasets/channel_estimator"),
                "--dataset-name", name,
            ], check=True)
        # Combine room and bus data within each matching partition: training
        # fits weights, validation selects the epoch/gates, and test reports
        # held-out metrics. Keep all outputs separate for each split policy.
        artifacts = PACKAGE / "training/artifacts" / mode
        if not args.updater_only:
            subprocess.run([
                sys.executable, str(PACKAGE / "training/cli.py"), "train",
                "--dataset", ",".join(names),
                "--checkpoint", str(artifacts / "channel_estimator.pt"),
                "--report", str(artifacts / "channel_estimator_report.json"),
                "--artifact-prefix", str(artifacts / "channel_estimator_weights"),
                "--header", str(ROOT / f"kernel/channel_estimator/model/frame_equalizer_channel_ai_weights_{mode}.h"),
                "--cpu-header", str(ROOT.parent / f"gr-ieee802-11-modified/lib/equalizer/model/channel_cnn_weights_{mode}.h"),
            ], check=True)
        # The updater reads the same saved room/bus partitions directly because
        # its examples also need the following payload symbol and its pilots.
        # Passing this mode's CNN prevents cross-policy baseline/model mixing.
        subprocess.run([
            sys.executable, str(PACKAGE / "training/updater.py"),
            "--recordings", *recordings, "--split-mode", mode,
        ], check=True)


if __name__ == "__main__":
    main()
