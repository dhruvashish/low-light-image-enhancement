# Dataset Instructions

This directory houses the Low-Light (LOL) benchmark dataset used throughout the 3-stage controlled empirical study.

## Dataset Structure

```text
data/
└── LOLdataset/
    ├── controlled_train/   # 435 paired low/high images (frozen seed=42)
    │   ├── low/
    │   └── high/
    ├── controlled_val/     # 50 paired validation images
    │   ├── low/
    │   └── high/
    └── eval15/             # 15 official benchmark test images (strictly isolated)
        ├── low/
        └── high/
```

## Dataset Information
- **LOL Dataset Source**: [Wei et al., BMVC 2018](https://daoosrn.github.io/dataset/lol.html)
- **Controlled Partition**: 435 pairs training, 50 pairs validation, and 15 official test pairs (`eval15`).
- **Data Augmentation**: $256 \times 256$ random crops, random horizontal/vertical flips, and 90-degree rotations during training.
