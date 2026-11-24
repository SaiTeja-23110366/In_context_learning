#!/bin/bash
source ~/my_venv/bin/activate

# Loop through all 7B NON-IID experiment configurations
for config in configs/*noniid_7b.yaml; do
    echo "========================================"
    echo "STARTING: $config"
    echo "========================================"
    # Use GPU 1 because GPU 0 is full
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_experiments.py --config "$config"
done

echo "========================================"
echo "ALL 7B NON-IID EXPERIMENTS FINISHED!"
echo "========================================"
