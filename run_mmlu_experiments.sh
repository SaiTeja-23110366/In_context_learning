#!/bin/bash
source ~/my_venv/bin/activate
cd /data/pulakurthi.manohar/In_context_learning

echo "=========================================="
echo "STARTING MMLU CONTEXTUAL BANDIT EXPERIMENTS"
echo "Date: $(date)"
echo "=========================================="

# 1. Run 4-domain 7B experiment on GPU 1
echo "=========================================="
echo "[1/3] Running MMLU 4-domain (Qwen2.5-7B)..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_experiments.py --config configs/mmlu_domain_7b.yaml

# 2. Run 8-domain 7B experiment on GPU 1
echo "=========================================="
echo "[2/3] Running MMLU 8-domain (Qwen2.5-7B)..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_experiments.py --config configs/mmlu_domain_8parts_7b.yaml

# 3. Run 4-domain 14B experiment on GPU 1
echo "=========================================="
echo "[3/3] Running MMLU 4-domain (Qwen2.5-14B)..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_experiments.py --config configs/mmlu_domain_14b.yaml

echo "=========================================="
echo "GENERATING COMPARISON TABLES"
echo "=========================================="
python3 scripts/generate_tables.py --results_dir /data/pulakurthi.manohar/results

echo "=========================================="
echo "ALL MMLU EXPERIMENTS COMPLETED AT $(date)"
echo "=========================================="
