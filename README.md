# Low-Light Image Enhancement: Benchmarking & Sub-Pixel Architecture

Official research repository for the comprehensive low-light image enhancement (LLIE) benchmark and efficient sub-pixel architecture evaluation on the LOL dataset.

---

## Overview

Real-world deployment of deep learning models for low-light enhancement is often constrained by high inference latency, large memory footprints, and numerical instability under mixed-precision training. 

This repository provides:
1. **Multi-Stage Controlled Benchmark**: A rigorous evaluation across 8 initial architectures, progressing to a controlled precision audit and a standardized Stage 3 benchmark under uniform native Bfloat16 (BF16) mixed precision.
2. **Precision Audit Findings**: Analysis exposing numerical overflow issues in LayerNorm layers under FP16 training and standardizing a robust BF16 training protocol with single-precision loss computation.
3. **Pareto-Optimal Architecture**: An efficient Encoder–Decoder network with sub-pixel PixelShuffle upsampling that achieves restoration quality comparable to heavy attention models while delivering over **5.8× faster inference** and **65% lower GPU memory consumption**.

---

## Quantitative Benchmark (LOL `eval15`)

All models were evaluated under identical conditions (150 epochs, native BF16, AdamW optimizer, batch size 4) on an NVIDIA RTX 4050 GPU:

| Architecture | Params (M) | GFLOPs ($256^2$) | Peak Val SSIM | Test PSNR (dB) | Test SSIM | Latency (ms) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Encoder–Decoder (Sub-Pixel)** | **7.85** | 12.41 | **0.8616** | **21.75** | **0.8284** | **6.26 (160 FPS)** | **1,525** |
| NAFNet + Channel Attention | 7.88 | **11.61** | 0.8551 | **21.86** | 0.8247 | 36.40 (27 FPS) | 4,409 |
| NAFNet Baseline | 8.79 | **11.61** | 0.8599 | 21.70 | **0.8287** | 34.26 (29 FPS) | 4,421 |
| NAFNet + Adaptive Gating | 8.96 | 12.14 | 0.8578 | 21.39 | 0.8213 | 35.73 (28 FPS) | 4,636 |

---

## Repository Structure

```text
├── configs/             # Experiment YAML configurations
├── data/                # Dataset directory & splits (LOLdataset)
├── docs/                # Research paper, IEEE LaTeX source, PDF & novelty summary
│   ├── figures/         # Publication figures (fig1 to fig17)
│   ├── IEEE_RESEARCH_PAPER_FROM_DARKNESS_TO_DETAIL.tex
│   └── IEEE_RESEARCH_PAPER_FROM_DARKNESS_TO_DETAIL_BENCHMARK.pdf
├── experiments/         # Structured CSV training logs & audit manifest
├── models/              # PyTorch neural network architectures
├── results/             # Evaluation metrics, summary CSVs & output samples
├── src/                 # Core training, evaluation, loss & dataset modules
├── train.py             # CLI runner for training
├── evaluate.py          # CLI runner for evaluation
├── requirements.txt     # Python package dependencies
├── environment.yml      # Conda environment specification
└── README.md
```

---

## Installation & Setup

### Using Conda
```bash
conda env create -f environment.yml
conda activate lowlight-benchmark
```

### Using Pip
```bash
pip install -r requirements.txt
```

---

## Usage

### Training
Train models using standard configuration files:
```bash
# Train proposed Encoder-Decoder architecture
python train.py --config configs/config_encoder_decoder.yaml

# Train NAFNet with Channel Attention
python train.py --config configs/config_nafnet_channel.yaml
```

### Evaluation
Run evaluations across test sets and compile comparative performance metrics:
```bash
python evaluate.py
```

---

## Research Documentation
The complete 10-page research paper, LaTeX manuscript, and novelty analysis are available in the `docs/` directory:
- [IEEE LaTeX Paper Source](docs/IEEE_RESEARCH_PAPER_FROM_DARKNESS_TO_DETAIL.tex)
- [Compiled IEEE Research Paper (PDF)](docs/IEEE_RESEARCH_PAPER_FROM_DARKNESS_TO_DETAIL_BENCHMARK.pdf)
- [Novelty Summary Document (DOCX)](docs/NOVELTY_SUMMARY.docx)

---

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
