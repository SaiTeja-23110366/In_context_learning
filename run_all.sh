#!/bin/bash
source ~/my_venv/bin/activate

# Loop through all 3B experiment configurations (both IID and non-IID)
for config in configs/*llama3b.yaml; do
    echo "========================================"
    echo "STARTING: $config"
    echo "========================================"
    python3 scripts/run_experiments.py --config "$config"
done

echo "========================================"
echo "ALL 3B EXPERIMENTS FINISHED!"
echo "========================================"
