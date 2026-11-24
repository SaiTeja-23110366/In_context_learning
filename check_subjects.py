from datasets import load_dataset
from collections import Counter

cache_dir = "/data/pulakurthi.manohar/hf_home"
ds = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
val_dev = list(ds["validation"]) + list(ds["dev"])
print("Total val+dev:", len(val_dev))
subjects = Counter(x["subject"] for x in val_dev)
print("Unique subjects:", len(subjects))
print("Sample subject counts:", list(subjects.items())[:5])
