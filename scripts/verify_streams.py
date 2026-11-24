import subprocess
import json
import os
import glob

# Remote script to run on cluster
remote_script = """
import glob, json, os, re, statistics

results_dir = '/data/pulakurthi.manohar/results'

configs = [
    'scale_banking_shards_n8_np2_k8',
    'scale_banking_shards_n16_np2_k8',
    'scale_banking_shards_n32_np2_k8',
    'scale_banking_shards_n64_np2_k8',
    'scale_banking_shards_n100_np2_k8',
    'scale_banking_shards_n8_np3_k12',
    'scale_banking_shards_n16_np3_k12',
    'scale_banking_shards_n32_np3_k12',
    'scale_banking_shards_n64_np3_k12',
    'scale_banking_shards_n100_np3_k12',
    'ablate_banking_domain_4parts_np2_k8',
    'ablate_banking_domain_8parts_np3_k12',
]

methods = ['no_noise', 'worst_case', 'uniform', 'indep_ucb', 'kucb', 'linucb', 'neural_pg', 'knn_retrieval']

mismatches = []

for cfg in configs:
    for s in range(10):
        # 1. Summary file
        sf = f'{results_dir}/summary_{cfg}_seed{s}.json'
        if not os.path.exists(sf):
            mismatches.append({'config': cfg, 'seed': s, 'method': 'ALL', 'issue': 'summary_missing'})
            continue
        with open(sf) as f:
            sdata = json.load(f)
        s_noise = sdata.get('noise_rates')
        s_part_sizes = sdata.get('partition_sizes')
        
        # Check per-method files
        golds = {}
        for m in methods:
            mf = f'{results_dir}/{m}_{cfg}_seed{s}.json'
            if not os.path.exists(mf):
                mismatches.append({'config': cfg, 'seed': s, 'method': m, 'issue': 'file_missing'})
                continue
            with open(mf) as f:
                mdata = json.load(f)
            r = mdata.get('rows', [])
            golds[m] = [row.get('gold') for row in r]
            
            # Check partition sizes if in extra
            extra = mdata.get('extra', {})
            if isinstance(extra, dict) and 'partition_sizes' in extra:
                if extra['partition_sizes'] != s_part_sizes:
                    mismatches.append({'config': cfg, 'seed': s, 'method': m, 'issue': 'partition_size_mismatch'})
        
        # Compare test stream gold label sequences across methods
        present_methods = [m for m in methods if m in golds]
        if present_methods:
            base_m = present_methods[0]
            base_gold = golds[base_m]
            for m in present_methods[1:]:
                if golds[m] != base_gold:
                    mismatches.append({
                        'config': cfg,
                        'seed': s,
                        'method': m,
                        'issue': f'gold_stream_mismatch_vs_{base_m}',
                        'base_len': len(base_gold),
                        'm_len': len(golds[m])
                    })

print("=== SUMMARY OF MISMATCHES ===")
print(f"Total mismatch records across all configs, seeds, and methods: {len(mismatches)}")
for m in mismatches:
    print(m)

# Now specifically analyze uniform at n=100, np=3 vs n=8..64
print("\\n=== UNIFORM ACCURACY AND NOISE EXPOSURE ANALYSIS ===")
np3_configs = [
    'scale_banking_shards_n8_np3_k12',
    'scale_banking_shards_n16_np3_k12',
    'scale_banking_shards_n32_np3_k12',
    'scale_banking_shards_n64_np3_k12',
    'scale_banking_shards_n100_np3_k12',
]

for cfg in np3_configs:
    accs = []
    shards_pulled_noise = []
    per_seed_info = []
    for s in range(10):
        sf = f'{results_dir}/summary_{cfg}_seed{s}.json'
        mf = f'{results_dir}/uniform_{cfg}_seed{s}.json'
        if os.path.exists(sf) and os.path.exists(mf):
            with open(sf) as f:
                sdata = json.load(f)
            with open(mf) as f:
                mdata = json.load(f)
            acc = mdata.get('accuracy', 0.0)
            accs.append(acc)
            noise_rates = sdata.get('noise_rates', [])
            rows = mdata.get('rows', [])
            pulled_rates = []
            for r in rows:
                if 'superarm' in r:
                    pulled_rates.extend([noise_rates[i] for i in r['superarm']])
                elif 'arm' in r and r['arm'] >= 0:
                    pulled_rates.append(noise_rates[r['arm']])
            m_noise = sum(pulled_rates) / len(pulled_rates) if pulled_rates else 0.0
            shards_pulled_noise.append(m_noise)
            per_seed_info.append(f"s{s}:{acc*100:.1f}%(noise={m_noise:.3f})")
    
    mean_acc = statistics.mean(accs)
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
    mean_noise = statistics.mean(shards_pulled_noise) if shards_pulled_noise else 0.0
    print(f"{cfg}:")
    print(f"  Uniform Acc: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}% (seeds={len(accs)})")
    print(f"  Mean Noise Exposure of Pulled Shards: {mean_noise:.4f}")
    print(f"  Per seed: {', '.join(per_seed_info)}")

print("\\n=== COMPARISON OF N=64 VS N=100 UNIFORM BEHAVIOR ===")
# Let us check how Uniform selects shots in n=64 vs n=100 in run_experiments.py
# In run_experiments.py line 191:
# arm = int(rng.integers(0, len(noisy_partitions)))
# shots = sample_shots(noisy_partitions[arm], k, rng, use_noisy=True)
# NOTICE: Uniform selects ONLY 1 partition for ALL k shots, or how does it work?
"""

proc = subprocess.run(
    ["ssh", "pulakurthi.manohar@flair2.iitgn.ac.in", "python3"],
    input=remote_script,
    text=True,
    capture_output=True,
)
print("STDOUT:")
print(proc.stdout)
if proc.stderr:
    print("STDERR:")
    print(proc.stderr)
