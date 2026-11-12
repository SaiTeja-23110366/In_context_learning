import json

print('=== n_p=2, k=8 SCALING RESULTS ===')
for n in [8, 16, 32, 64, 100]:
    fn = f'/data/pulakurthi.manohar/results/summary_scale_banking_shards_n{n}_np2_k8.json'
    try:
        with open(fn) as f:
            d = json.load(f)
        c = n * (n - 1) // 2
        print(f"n={n:3d} (C={c:4d}): indep={d['indep_ucb']['mean']:5.2f}+-{d['indep_ucb']['std']:4.2f} | kucb={d['kucb']['mean']:5.2f}+-{d['kucb']['std']:4.2f} | lin={d['linucb']['mean']:5.2f}+-{d['linucb']['std']:4.2f} | knn={d['knn_retrieval']['mean']:5.2f}+-{d['knn_retrieval']['std']:4.2f} | unif={d['uniform']['mean']:5.2f}+-{d['uniform']['std']:4.2f}")
    except Exception as e:
        print(f"n={n}: {e}")

print('\n=== n_p=3, k=12 SCALING RESULTS ===')
for n in [8, 16, 32, 64, 100]:
    fn = f'/data/pulakurthi.manohar/results/summary_scale_banking_shards_n{n}_np3_k12.json'
    try:
        with open(fn) as f:
            d = json.load(f)
        c = n * (n - 1) * (n - 2) // 6
        print(f"n={n:3d} (C={c:6d}): indep={d['indep_ucb']['mean']:5.2f}+-{d['indep_ucb']['std']:4.2f} | kucb={d['kucb']['mean']:5.2f}+-{d['kucb']['std']:4.2f} | lin={d['linucb']['mean']:5.2f}+-{d['linucb']['std']:4.2f} | knn={d['knn_retrieval']['mean']:5.2f}+-{d['knn_retrieval']['std']:4.2f} | unif={d['uniform']['mean']:5.2f}+-{d['uniform']['std']:4.2f}")
    except Exception as e:
        print(f"n={n}: in progress ({e})")
