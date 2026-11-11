#!/usr/bin/env bash
# ==============================================================================
# run_all_scaling.sh
# Autonomous Scaling and Domain Ablation Runner
# Uses CUDA_VISIBLE_DEVICES=1 on cluster.
# Resumable: each run checks existing results and skips completed runs.
# ==============================================================================
set -e

mkdir -p logs
mkdir -p /data/pulakurthi.manohar/results

PYTHON="/home/pulakurthi.manohar/my_venv/bin/python"
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/data/pulakurthi.manohar/hf_home"

echo "======================================================================"
echo "Starting Scaling & Ablation Experiments at $(date)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

run_exp() {
    CFG=$1
    TAG=$(basename "$CFG" .yaml)
    echo ""
    echo ">>> Running config: $CFG ($(date)) <<<"
    $PYTHON scripts/run_experiments.py --config "$CFG" 2>&1 | tee -a "logs/${TAG}.log"
    echo ">>> Completed config: $CFG ($(date)) <<<"
}

# ------------------------------------------------------------------------------
# Priority 1: Experiment B (Domain Ablation, seeds 0-9)
# ------------------------------------------------------------------------------
echo "--- Starting Experiment B (Domain Ablation) ---"
run_exp "configs/ablate_banking_domain_4parts_np2_k8.yaml"
run_exp "configs/ablate_banking_domain_8parts_np3_k12.yaml"

# ------------------------------------------------------------------------------
# Priority 2: Experiment A1 (Priority Shards: n in {8, 32, 100}, np=2, k=8)
# ------------------------------------------------------------------------------
echo "--- Starting Experiment A1 (Priority Shards: n=8, 32, 100, np=2) ---"
run_exp "configs/scale_banking_shards_n8_np2_k8.yaml"
run_exp "configs/scale_banking_shards_n32_np2_k8.yaml"
run_exp "configs/scale_banking_shards_n100_np2_k8.yaml"

# ------------------------------------------------------------------------------
# Priority 3: Experiment A1 (Remaining Shards np=2: n in {16, 64})
# ------------------------------------------------------------------------------
echo "--- Starting Experiment A1 (Remaining Shards: n=16, 64, np=2) ---"
run_exp "configs/scale_banking_shards_n16_np2_k8.yaml"
run_exp "configs/scale_banking_shards_n64_np2_k8.yaml"

# ------------------------------------------------------------------------------
# Priority 4: Experiment A1 (Shards np=3, k=12: n in {8, 16, 32, 64, 100})
# ------------------------------------------------------------------------------
echo "--- Starting Experiment A1 (Shards np=3, k=12) ---"
run_exp "configs/scale_banking_shards_n8_np3_k12.yaml"
run_exp "configs/scale_banking_shards_n16_np3_k12.yaml"
run_exp "configs/scale_banking_shards_n32_np3_k12.yaml"
run_exp "configs/scale_banking_shards_n64_np3_k12.yaml"
run_exp "configs/scale_banking_shards_n100_np3_k12.yaml"

# ------------------------------------------------------------------------------
# Table and Plot generation
# ------------------------------------------------------------------------------
echo "--- Generating Tables and Scaling Plots ---"
$PYTHON scripts/generate_tables.py --results_dir /data/pulakurthi.manohar/results > tables_scaling.txt 2>&1 || true
$PYTHON scripts/plot_scaling.py --results_dir /data/pulakurthi.manohar/results --out_dir plots 2>&1 || true

echo "======================================================================"
echo "Phase A1 & Experiment B Finished at $(date)"
echo "======================================================================"
