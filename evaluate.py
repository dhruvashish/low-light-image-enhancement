"""
Root CLI entry point for benchmark evaluation and analytics compilation on LOL eval15.

Usage:
    python evaluate.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate import evaluate_all_final

if __name__ == "__main__":
    evaluate_all_final()
