# Historical Experiment Logs

This directory organizes all historical training logs across the three benchmarking phases:

1. `01_Stage1_Preliminary_8_Models_20_Epochs/`: Preliminary 20-epoch triage across 8 candidate architectures.
2. `02_Stage2_Controlled_4_Models_100_150_Epochs/`: Controlled audit exposing FP16 LayerNorm overflow in NAFNet models.
3. `03_Stage3_Final_Corrected_Benchmark_150_Epochs_BF16/`: Corrected benchmark under uniform native BF16 precision and monotonic cosine schedules.
4. `ALL_MODELS_MASTER_LOGS_MANIFEST.csv`: Master manifest linking every model to its parameters, FLOPs, latency, and test scores.
