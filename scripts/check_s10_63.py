import subprocess

script = """
import json
with open('/data/pulakurthi.manohar/results/kucb_scale_banking_shards_n64_np3_k12_seed0.json') as f:
    d = json.load(f)
rows = d['rows']
from collections import Counter
counts = Counter()
for r in rows:
    for s in r['superarm']:
        counts[s] += 1
s10_63 = [counts[i] for i in range(10, 64)]
print(f'Shards 10..63 pulls: min={min(s10_63)}, max={max(s10_63)}, unique counts={set(s10_63)}')
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 63]:
    print(f'Shard {i}: {counts[i]}')
"""

p = subprocess.run(["ssh", "pulakurthi.manohar@flair2.iitgn.ac.in", "python3"], input=script, text=True, capture_output=True)
print(p.stdout)
if p.stderr:
    print("STDERR:", p.stderr)
