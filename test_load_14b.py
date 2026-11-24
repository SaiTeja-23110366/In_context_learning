import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from src.icl import load_model

print("Checking available GPUs...")
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Initial allocated:", torch.cuda.memory_allocated(0) / 1024**3, "GB")

print("Loading Qwen2.5-14B-Instruct...")
tok, model = load_model("/data/pulakurthi.manohar/hf_home/Qwen2.5-14B-Instruct")
print("Successfully loaded 14B model! Device:", model.device)
print("Allocated after load:", torch.cuda.memory_allocated(0) / 1024**3, "GB")
