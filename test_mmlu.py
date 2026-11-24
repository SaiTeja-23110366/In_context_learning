import os
import sys
from datasets import load_dataset

cache_dir = "/data/pulakurthi.manohar/hf_home"
print("Testing MMLU dataset download...")
try:
    ds = load_dataset("cais/mmlu", "all", split="dev", cache_dir=cache_dir)
    print("Dev set loaded successfully! Length:", len(ds))
    print("Sample:", ds[0])
except Exception as e:
    print("Error loading cais/mmlu:", e)
