import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_task, make_non_iid_partitions

for seed in [0, 1, 2]:
    print(f"=== SEED {seed} ===")
    train, test, label_names = load_task("ecommerce", "/data/pulakurthi.manohar/hf_home", seed=seed)
    parts = make_non_iid_partitions(train, 4, seed, 0.3)
    for i, p in enumerate(parts):
        counts = {}
        for x in p:
            counts[label_names[x["label"]]] = counts.get(label_names[x["label"]], 0) + 1
        print(f"  Partition {i} (size {len(p)}): {counts}")
