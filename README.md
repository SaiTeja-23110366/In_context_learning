# Robust In-Context Learning via Multi-Armed Bandit-Based Partition Selection

Reproduction of the paper + our independent label-noise injection module.

## Setup (run once on the cluster)

```bash
cd /data/pulakurthi.manohar/PROJECT_Manisha
pip install -r requirements.txt
```

## Run the unit tests (our independent noise module)

```bash
python -m pytest tests/test_noise.py -v
```

## Run a smoke test (fast check, ~5 min)

```bash
python scripts/run_experiments.py --config configs/agnews_iid_7b.yaml
```

## Run all experiments (IID — 3 models × 3 datasets)

```bash
for cfg in configs/agnews_iid_llama3b.yaml configs/agnews_iid_7b.yaml configs/agnews_iid_14b.yaml \
           configs/ecom_iid_llama3b.yaml   configs/ecom_iid_7b.yaml   configs/ecom_iid_14b.yaml \
           configs/syn_iid_llama3b.yaml    configs/syn_iid_7b.yaml    configs/syn_iid_14b.yaml; do
    echo "=== Running $cfg ==="
    python scripts/run_experiments.py --config $cfg
done
```

## Run all experiments (Non-IID — 3 models × 3 datasets)

```bash
for cfg in configs/agnews_noniid_llama3b.yaml configs/agnews_noniid_7b.yaml configs/agnews_noniid_14b.yaml \
           configs/ecom_noniid_llama3b.yaml   configs/ecom_noniid_7b.yaml   configs/ecom_noniid_14b.yaml \
           configs/syn_noniid_llama3b.yaml    configs/syn_noniid_7b.yaml    configs/syn_noniid_14b.yaml; do
    echo "=== Running $cfg ==="
    python scripts/run_experiments.py --config $cfg
done
```

## Generate results tables (matching paper Tables 1a and 1b)

```bash
python scripts/generate_tables.py --results_dir /data/pulakurthi.manohar/results
```

## What is ours vs what is reproduced

| Component | Source |
|---|---|
| `src/bandit.py` | Paper reproduction (UCB1 + kUCB) |
| `src/data_utils.py` | Paper reproduction (IID + Non-IID partitioning) |
| `src/icl.py` | Paper reproduction (prompt + inference) |
| `scripts/run_experiments.py` | Paper reproduction (experiment loop) |
| **`src/noise_module.py`** | **OUR ADDITION** — standalone label-noise injection |
| **`tests/test_noise.py`** | **OUR ADDITION** — unit tests for noise module |
