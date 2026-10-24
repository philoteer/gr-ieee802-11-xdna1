#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.frame_equalizer_channel_ai.common.model import ChannelEstimator
from utils.frame_equalizer_channel_ai.training.updater import load_checkpoint
from utils.frame_equalizer_channel_ai.training.exporter import (
    DEFAULT_CPU_DIRECTORY, export_trained_cnn, export_trained_updater,
)


def main():
    parser = argparse.ArgumentParser(description="Export trained checkpoints directly as GNU Radio float weights")
    artifacts = Path(__file__).resolve().parent / 'artifacts'
    parser.add_argument("--split-mode", choices=('by_frame', 'by_channel'), default='by_frame')
    parser.add_argument("--cnn-checkpoint", type=Path)
    parser.add_argument("--updater-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CPU_DIRECTORY)
    args = parser.parse_args()
    args.cnn_checkpoint = args.cnn_checkpoint or artifacts / args.split_mode / 'channel_estimator.pt'
    args.updater_checkpoint = args.updater_checkpoint or artifacts / args.split_mode / 'channel_updater.pt'
    payload = torch.load(args.cnn_checkpoint, map_location='cpu', weights_only=False)
    model = ChannelEstimator()
    model.load_state_dict(payload['state_dict'])
    print(export_trained_cnn(model, payload['metadata']['cpu_gate_snr_ratio_x16'],
                            args.output_dir / f"channel_cnn_weights_{args.split_mode}.h"))
    models, metadata = load_checkpoint(args.updater_checkpoint)
    if metadata['split_mode'] != args.split_mode:
        raise ValueError('updater checkpoint split policy does not match --split-mode')
    print(export_trained_updater(models, metadata['cpu_gate_snr_ratio_x16'],
                                args.output_dir / f"channel_updater_mlp_weights_{args.split_mode}.h"))


if __name__ == "__main__":
    main()
