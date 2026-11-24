import scipy
import scipy.stats as stats
import matplotlib
import sentence_transformers
import torch
print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
print("torch device count:", torch.cuda.device_count())
