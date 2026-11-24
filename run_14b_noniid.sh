#!/bin/bash
source ~/my_venv/bin/activate

echo "========================================"
echo "STARTING 14B NON-IID RUNS ON GPU 1"
echo "========================================"

for config in configs/agnews_noniid_14b.yaml configs/ecom_noniid_14b.yaml configs/syn_noniid_14b.yaml; do
    echo "========================================"
    echo "STARTING: $config"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_experiments.py --config "$config"
done

echo "========================================"
echo "ALL 14B NON-IID EXPERIMENTS COMPLETED!"
echo "========================================"
