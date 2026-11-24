from datasets import load_dataset

cache_dir = "/data/pulakurthi.manohar/hf_home"
ds = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
for split in ds:
    print(f"Split {split}: {len(ds[split])} examples, columns: {ds[split].column_names}")
    print(f"Sample from {split}: {ds[split][0]}")
