# ALL EXPERIMENTAL TRAINING LOGS & CSV ARCHIVE

This directory provides a consolidated, organized repository of all training logs, per-image test metric CSVs, and summary tables across the entire Low-Light Image Enhancement research project on the LOL Dataset.

---

## Directory Structure

```
ALL_EXPERIMENTS_LOGS/
│
├── 01_Stage1_Preliminary_8_Models_20_Epochs/
│   ├── summary_stage1_metrics.csv
│   ├── autoencoder/ (training_log.csv, per_image_metrics.csv)
│   ├── encoder_decoder/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet_channel/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet_dual/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet_gating/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet_spatial/ (training_log.csv, per_image_metrics.csv)
│   └── unet/ (training_log.csv, per_image_metrics.csv)
│
├── 02_Stage2_Controlled_4_Models_100_150_Epochs/
│   ├── summary_stage2_metrics.csv
│   ├── encoder_decoder/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet_channel/ (training_log.csv, per_image_metrics.csv)
│   └── nafnet_gating/ (training_log.csv, per_image_metrics.csv)
│
├── 03_Stage3_Final_Corrected_Benchmark_150_Epochs_BF16/
│   ├── summary_stage3_metrics.csv
│   ├── encoder_decoder/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet/ (training_log.csv, per_image_metrics.csv)
│   ├── nafnet_channel/ (training_log.csv, per_image_metrics.csv)
│   └── nafnet_gating/ (training_log.csv, per_image_metrics.csv)
│
├── ALL_MODELS_MASTER_LOGS_MANIFEST.csv
└── README_LOGS_ORGANIZATION.md
```

---

## Master Summary Table Across All Three Stages

| Stage | Architecture | Target Epochs | Params (M) | GFLOPs | Peak Val PSNR | Peak Val SSIM | Official Test PSNR | Official Test SSIM | Latency (ms) | Peak VRAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Stage 1 (20-Epoch Screen) | **unet** | 10 | 4.32 | 10.119 | N/A | N/A | **13.2917** | **0.6302** | 3.6 | 3461.4 |
| Stage 1 (20-Epoch Screen) | **nafnet** | 20 | 8.79 | 11.605 | N/A | N/A | **19.3343** | **0.7497** | 23.16 | 753.4 |
| Stage 1 (20-Epoch Screen) | **nafnet_spatial** | 20 | 8.79 | 11.641 | N/A | N/A | **18.6784** | **0.7396** | 28.69 | 420.2 |
| Stage 1 (20-Epoch Screen) | **nafnet_channel** | 20 | 7.88 | 11.605 | N/A | N/A | **19.5185** | **0.7505** | 25.67 | 404.7 |
| Stage 1 (20-Epoch Screen) | **nafnet_dual** | 20 | 7.89 | 11.640 | N/A | N/A | **18.6208** | **0.7486** | 30.5 | 403.8 |
| Stage 1 (20-Epoch Screen) | **nafnet_gating** | 20 | 8.96 | 12.142 | N/A | N/A | **18.9185** | **0.7535** | 26.3 | 422.5 |
| Stage 1 (20-Epoch Screen) | **autoencoder** | 20 | 5.92 | 6.623 | N/A | N/A | **14.3492** | **0.6321** | 2.58 | 278.5 |
| Stage 1 (20-Epoch Screen) | **encoder_decoder** | 20 | 7.85 | 12.406 | N/A | N/A | **18.8889** | **0.7583** | 5.16 | 376.2 |
| Stage 2 (Controlled Audit) | **encoder_decoder** | 150 | 7.85 | 12.406 | 22.0774 | 0.856006 | **20.9851** | **0.821537** | 5.5 | 1454.0 |
| Stage 2 (Controlled Audit) | **nafnet_channel** | 100 | 7.88 | 11.605 | 20.734 | 0.749388 | **19.8325** | **0.731573** | 29.21 | 4134.0 |
| Stage 2 (Controlled Audit) | **nafnet** | 100 | 8.79 | 11.605 | 20.7358 | 0.785015 | **19.7731** | **0.753171** | 25.1 | 4146.0 |
| Stage 2 (Controlled Audit) | **nafnet_gating** | 150 | 8.96 | 12.142 | 19.7451 | 0.716787 | **18.6234** | **0.689241** | 26.6 | 4359.4 |
| Stage 3 (Final Corrected BF16) | **encoder_decoder** | 150 | 7.85 | 12.406 | 22.3316 | 0.861603 | **21.7484** | **0.828352** | 6.26 | 1524.9 |
| Stage 3 (Final Corrected BF16) | **nafnet** | 150 | 8.79 | 11.605 | 22.5862 | 0.859894 | **21.6986** | **0.828675** | 34.26 | 4421.2 |
| Stage 3 (Final Corrected BF16) | **nafnet_channel** | 150 | 7.88 | 11.605 | 22.5606 | 0.855066 | **21.8559** | **0.824722** | 36.4 | 4409.1 |
| Stage 3 (Final Corrected BF16) | **nafnet_gating** | 150 | 8.96 | 12.142 | 22.3104 | 0.857825 | **21.3854** | **0.821283** | 35.73 | 4635.5 |

