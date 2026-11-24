import subprocess
import json

script = '''
import json

results_dir = '/data/pulakurthi.manohar/results'
ns = [8, 16, 64, 100]
methods = ['uniform', 'indep_ucb', 'kucb', 'linucb', 'neural_pg']

data = {n: {m: {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []} for m in methods} for n in ns}
gaps = {n: {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q4_minus_Q1': []} for n in ns}

for n in ns:
    cfg = f'scale_banking_shards_n{n}_np2_k8'
    for s in range(10):
        seed_accs = {}
        for m in methods:
            fname = f'{results_dir}/{m}_{cfg}_seed{s}.json'
            with open(fname) as f:
                d = json.load(f)
            rows = d['rows']
            q1 = sum(r['reward'] for r in rows[0:80]) / 80.0
            q2 = sum(r['reward'] for r in rows[80:160]) / 80.0
            q3 = sum(r['reward'] for r in rows[160:240]) / 80.0
            q4 = sum(r['reward'] for r in rows[240:320]) / 80.0
            data[n][m]['Q1'].append(q1)
            data[n][m]['Q2'].append(q2)
            data[n][m]['Q3'].append(q3)
            data[n][m]['Q4'].append(q4)
            seed_accs[m] = (q1, q2, q3, q4)
        
        g_q1 = seed_accs['linucb'][0] - seed_accs['indep_ucb'][0]
        g_q2 = seed_accs['linucb'][1] - seed_accs['indep_ucb'][1]
        g_q3 = seed_accs['linucb'][2] - seed_accs['indep_ucb'][2]
        g_q4 = seed_accs['linucb'][3] - seed_accs['indep_ucb'][3]
        gaps[n]['Q1'].append(g_q1)
        gaps[n]['Q2'].append(g_q2)
        gaps[n]['Q3'].append(g_q3)
        gaps[n]['Q4'].append(g_q4)
        gaps[n]['Q4_minus_Q1'].append(g_q4 - g_q1)

out = {'data': data, 'gaps': gaps}
print(json.dumps(out))
'''

proc = subprocess.run(
    ['ssh', 'pulakurthi.manohar@flair2.iitgn.ac.in', 'python3'],
    input=script,
    text=True,
    capture_output=True
)
if proc.stderr:
    print("STDERR:", proc.stderr)

raw_json = proc.stdout.strip()
with open("In_context_learning/results_summary/learning_curves_data.json", "w") as f:
    f.write(raw_json)
print("Saved learning_curves_data.json locally!")

