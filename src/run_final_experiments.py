import os
import sys
import time
import json
import subprocess
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable

def run_cmd(cmd_list, desc):
    print("\n" + "=" * 80)
    print(f"EXECUTING: {desc}")
    print(f"Command: {' '.join(cmd_list)}")
    print("=" * 80, flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list, cwd=BASE_DIR)
    elapsed = (time.time() - t0) / 60
    if res.returncode != 0:
        print(f"[ERROR] Step failed with return code {res.returncode}: {desc}")
        sys.exit(res.returncode)
    else:
        print(f"[SUCCESS] Step completed in {elapsed:.1f} min: {desc}", flush=True)

def is_arch_completed(arch, expected_epochs=150):
    report_path = os.path.join(BASE_DIR, "outputs_final", arch, "experiment_report.json")
    if not os.path.isfile(report_path):
        return False
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
        log_path = os.path.join(BASE_DIR, "logs_final", arch, "training_log.csv")
        if os.path.isfile(log_path):
            with open(log_path, "r") as lf:
                lines = lf.readlines()
                # Check header + 150 lines
                if len(lines) >= expected_epochs + 1:
                    return True
        return "test_mean_psnr" in data
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Final Corrected Benchmark Sequential Orchestrator")
    parser.add_argument(
        "--archs", 
        nargs="+", 
        default=["nafnet", "nafnet_channel", "nafnet_gating", "encoder_decoder"],
        help="List of architectures to train in sequence"
    )
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain even if 150 epochs completed")
    args = parser.parse_args()

    print("=" * 85)
    print("STARTING FINAL CORRECTED BENCHMARK SEQUENCE (STAGE 3 — BF16)")
    print(f"Target Architectures: {args.archs}")
    print(f"Python Executable: {PYTHON_EXEC}")
    print(f"Base Directory: {BASE_DIR}")
    print("=" * 85, flush=True)

    master_t0 = time.time()

    for arch in args.archs:
        config_path = os.path.join("configs_final", f"config_{arch}.yaml")
        if not os.path.isfile(os.path.join(BASE_DIR, config_path)):
            print(f"[Error] Configuration file missing: {config_path}")
            sys.exit(1)

        completed = is_arch_completed(arch)
        if completed and not args.force_retrain:
            print(f"\n[INFO] Architecture '{arch}' already completed 150 epochs. Skipping training phase.", flush=True)
        else:
            print(f"\n>>> INITIATING TRAINING FOR ARCHITECTURE: {arch.upper()} <<<", flush=True)
            train_cmd = [PYTHON_EXEC, os.path.join(BASE_DIR, "train_final.py"), "--config", config_path]
            run_cmd(train_cmd, f"150-Epoch Native BF16 Training for {arch.upper()}")

        # Generate individual model report and visual artifacts
        rep_cmd = [PYTHON_EXEC, os.path.join(BASE_DIR, "generate_final_model_report.py"), arch]
        run_cmd(rep_cmd, f"Per-Model Artifacts Generation for {arch.upper()}")

    # After all models complete, generate consolidated analytics
    print("\n" + "=" * 85)
    print("ALL TARGET ARCHITECTURES FINISHED. GENERATING CONSOLIDATED SUITE...")
    print("=" * 85, flush=True)

    eval_cmd = [PYTHON_EXEC, os.path.join(BASE_DIR, "evaluate_final.py")]
    run_cmd(eval_cmd, "Consolidated Final Evaluation Compilation")

    graphs_cmd = [PYTHON_EXEC, os.path.join(BASE_DIR, "generate_final_graphs.py")]
    run_cmd(graphs_cmd, "Master Comparative Visualizations")

    docx_cmd = [PYTHON_EXEC, os.path.join(BASE_DIR, "generate_final_docx_report.py")]
    run_cmd(docx_cmd, "Publication-Grade Word Research Report")

    total_time_hr = (time.time() - master_t0) / 3600
    print("\n" + "=" * 85)
    print(f"FINAL CORRECTED BENCHMARK PIPELINE COMPLETE IN {total_time_hr:.2f} HOURS!")
    print("Outputs generated:")
    print("  - Individual Reports: reports_final/<arch>/")
    print("  - Consolidated Metrics: reports_final/consolidated_research_report.md")
    print("  - Master Plots: graphs_final/*.png")
    print("  - Word Report: FINAL_CORRECTED_BENCHMARK_RESEARCH_REPORT.docx")
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()

