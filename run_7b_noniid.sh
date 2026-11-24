#!/bin/bash
source ~/my_venv/bin/activate

echo "========================================"
echo "STARTING 7B NON-IID RUNS ON GPU 0"
echo "========================================"

for config in configs/agnews_noniid_7b.yaml configs/ecom_noniid_7b.yaml configs/syn_noniid_7b.yaml; do
    echo "========================================"
    echo "STARTING: $config"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_experiments.py --config "$config"
done

echo "========================================"
echo "ALL 7B NON-IID EXPERIMENTS COMPLETED!"
echo "========================================"
