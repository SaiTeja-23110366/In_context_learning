import json
import glob

def check(fn):
    print("=== File:", fn.split('/')[-1], "===")
    try:
        with open(fn) as f:
            d = json.load(f)
        print("Keys:", list(d.keys()))
        if 'noise_rates' in d:
            print("noise_rates:", d['noise_rates'])
        if 'rows' in d and len(d['rows']) > 0:
            print("rows count:", len(d['rows']))
            print("sample row 0:", d['rows'][0])
            print("sample row 1:", d['rows'][1])
        if 'extra' in d:
            print("extra keys:", list(d['extra'].keys()) if isinstance(d['extra'], dict) else d['extra'])
            if 'superarms' in d['extra']:
                print("num superarms:", len(d['extra']['superarms']))
                print("first 10 superarms:", d['extra']['superarms'][:10])
            if 'arm_counts' in d['extra']:
                print("arm_counts sample:", d['extra']['arm_counts'][:10])
    except Exception as e:
        print("Error:", e)

check('/data/pulakurthi.manohar/results/summary_ablate_banking_domain_4parts_np2_k8_seed0.json')
check('/data/pulakurthi.manohar/results/summary_ablate_banking_domain_8parts_np3_k12_seed0.json')
check('/data/pulakurthi.manohar/results/kucb_scale_banking_shards_n100_np2_k8_seed0.json')
check('/data/pulakurthi.manohar/results/kucb_scale_banking_shards_n64_np3_k12_seed0.json')
