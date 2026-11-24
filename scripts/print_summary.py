import glob
import json
import os
import sys

def summarize_file(path):
    print(f"\n=== {os.path.basename(path)} ===")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    for k, v in d.items():
        if isinstance(v, dict) and "accuracy" in v:
            acc = v["accuracy"] * 100
            sec = v.get("wall_clock_seconds", "—")
            cpu = v.get("peak_cpu_mb", "—")
            gpu = v.get("peak_gpu_mb", "—")
            arms = v.get("n_arms", "—")
            print(f"{k:<15}: {acc:.2f}% | time: {sec}s | cpu: {cpu}MB | gpu: {gpu}MB | arms: {arms}")
        elif isinstance(v, dict) and v.get("status") == "infeasible":
            print(f"{k:<15}: INFEASIBLE (superarms={v.get('superarms')})")

def main():
    if len(sys.argv) > 1:
        summarize_file(sys.argv[1])
    else:
        results_dir = "/data/pulakurthi.manohar/results"
        pattern = os.path.join(results_dir, "summary_*.json")
        files = sorted(glob.glob(pattern))
        target_files = [
            f for f in files 
            if any(p in os.path.basename(f) for p in ["ablate_", "scale_", "smoke_scaling"])
        ]
        if not target_files:
            print("No matching scaling/ablation summary files found.")
            return
        for f in target_files:
            summarize_file(f)

if __name__ == "__main__":
    main()

