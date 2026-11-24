from datasets import load_dataset
from src.data_utils import MMLU_DOMAINS, MMLU_8_DOMAINS, SUBJECT_TO_DOMAIN, SUBJECT_TO_8_DOMAIN

cache_dir = "/data/pulakurthi.manohar/hf_home"
ds = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
all_subjects = sorted(list({x["subject"] for x in ds["validation"]}))
print(f"Total subjects in MMLU: {len(all_subjects)}")

unmapped4 = [s for s in all_subjects if s not in SUBJECT_TO_DOMAIN]
print("Unmapped in 4-domains:", unmapped4)

unmapped8 = [s for s in all_subjects if s not in SUBJECT_TO_8_DOMAIN]
print("Unmapped in 8-domains:", unmapped8)
