import json
import glob
import os
import re

results_dir = "/data/pulakurthi.manohar/results"

# Check available runs
summary_files = glob.glob(f"{results_dir}/summary_*_seed*.json")
runs = {}
for sf in summary_files:
    fname = os.path.basename(sf)
    m = re.match(r"^summary_(.+)_seed(\d+)\.json$", fname)
    if m:
        rname, s = m.group(1), int(m.group(2))
        runs.setdefault(rname, []).append(s)

print("Available runs and seed counts:")
for r in sorted(runs.keys()):
    print(f"  {r}: {len(runs[r])} seeds {sorted(runs[r])}")


# Inspect one seed for each method in scale_banking_shards_n16_np2_k8
methods = ['no_noise', 'worst_case', 'uniform', 'indep_ucb', 'kucb', 'linucb', 'neural_pg', 'knn_retrieval', 'oracle_domain']
run_sample = "scale_banking_shards_n16_np2_k8"
print("\nInspecting methods for run:", run_sample)
for m in methods:
    fpath = f"{results_dir}/{m}_{run_sample}_seed0.json"
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        extra_keys = list(d.get('extra', {}).keys()) if isinstance(d.get('extra'), dict) else str(d.get('extra'))
        sample_row = d.get('rows', [])[0] if d.get('rows') else None
        print(f"Method {m}: acc={d.get('accuracy'):.4f}, rows={len(d.get('rows', []))}, extra={extra_keys}")
        if sample_row:
            print(f"   Sample row keys: {list(sample_row.keys())}")
            print(f"   Sample row: {sample_row}")
    else:
        print(f"Method {m}: file NOT found: {fpath}")

# Inspect uniform and oracle rows
for r_name in ["scale_banking_shards_n16_np2_k8", "ablate_banking_domain_4parts_np2_k8"]:
    print(f"\n--- Detailed check for {r_name} ---")
    for m in ['uniform', 'oracle_domain', 'worst_case', 'no_noise']:
        fpath = f"{results_dir}/{m}_{r_name}_seed0.json"
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            print(f"Method {m}:")
            for i in range(min(3, len(d.get('rows', [])))):
                print(f"  row {i}: {d['rows'][i]}")
            if 'extra' in d:
                for k in ['superarms', 'arm_counts', 'selected_partitions', 'partition_sizes']:
                    if k in d['extra']:
                        val = d['extra'][k]
                        if isinstance(val, list):
                            print(f"  extra[{k}]: len={len(val)}, first 5={val[:5]}")
                        else:
                            print(f"  extra[{k}]: {val}")

# Check noise rates in domain runs
for dr in ['ablate_banking_domain_4parts_np2_k8', 'ablate_banking_domain_8parts_np3_k12']:
    fpath = f"{results_dir}/summary_{dr}_seed0.json"
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        print(f"Domain run {dr} noise_rates:", d.get('noise_rates'))


