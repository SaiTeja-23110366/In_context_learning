import glob
import os

results_dir = "/data/pulakurthi.manohar/results"
methods = ["no_noise", "worst_case", "uniform", "indep_ucb", "kucb", "linucb", "neural_pg", "knn_retrieval"]
run = "scale_banking_shards_n100_np3_k12"

print(f"Checking files for {run}:")
missing = []
for s in range(10):
    sf = f"{results_dir}/summary_{run}_seed{s}.json"
    if not os.path.exists(sf):
        missing.append(f"summary seed {s}")
    for m in methods:
        mf = f"{results_dir}/{m}_{run}_seed{s}.json"
        if not os.path.exists(mf):
            missing.append(f"{m} seed {s}")

if missing:
    print(f"Missing {len(missing)} files: {missing}")
else:
    print("ALL 10 SEEDS (summary + all 8 methods = 90 files) ARE COMPLETE AND PRESENT!")
