import json
import sys

def main():
    path = sys.argv[1]
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

if __name__ == "__main__":
    main()
