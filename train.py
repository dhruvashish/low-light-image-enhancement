"""
Root CLI entry point for training low-light image enhancement models.

Usage:
    python train.py --config configs/config_encoder_decoder.yaml
    python train.py --config configs/config_nafnet_channel.yaml
    python train.py --config configs/config_nafnet.yaml
    python train.py --config configs/config_nafnet_gating.yaml
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train import main

if __name__ == "__main__":
    main()
