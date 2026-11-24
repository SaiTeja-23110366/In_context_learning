import glob
import os
from collections import defaultdict

RESULTS_DIR = "/data/pulakurthi.manohar/results"

def main():
    files = glob.glob(f"{RESULTS_DIR}/*.json")
    print(f"Total json files in {RESULTS_DIR}: {len(files)}")
    
    summaries = glob.glob(f"{RESULTS_DIR}/summary_*_seed*.json")
    runs = defaultdict(list)
    for sf in summaries:
        base = os.path.basename(sf).replace("summary_", "").replace(".json", "")
        parts = base.rsplit("_seed", 1)
        if len(parts) == 2:
            try:
                runs[parts[0]].append(int(parts[1]))
            except ValueError:
                pass
            
    print("\n--- All Runs in Results Directory ---")
    for rname in sorted(runs.keys()):
        seeds = sorted(runs[rname])
        print(f"{rname:<42} : {len(seeds)} seeds -> {seeds}")
        
    print("\nOverall Status:", "ALL EXPERIMENTS COMPLETE!" if all_complete else "SOME RUNS INCOMPLETE!")

if __name__ == "__main__":
    main()
