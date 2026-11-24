import sys
import os
sys.path.insert(0, '/data/pulakurthi.manohar/In_context_learning')
from src.data_utils import load_banking
from sentence_transformers import SentenceTransformer
import numpy as np

train, test, label_names = load_banking('/data/pulakurthi.manohar/hf_home')
print(f"Loaded Banking: train={len(train)}, test={len(test)}, labels={len(label_names)}")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/data/pulakurthi.manohar/hf_home', device='cpu')
train_texts = [x['text'] for x in train]
test_texts = [x['text'] for x in test]
print("Encoding train texts...")
train_emb = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=False)
print("Encoding test texts...")
test_emb = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

sims = test_emb @ train_emb.T
max_sims = np.max(sims, axis=1)
print(f"Max cosine stats: min={max_sims.min():.4f}, mean={max_sims.mean():.4f}, max={max_sims.max():.4f}")

exact_matches = sum(1 for q in test_texts if any(q.strip().lower() == t.strip().lower() for t in train_texts))
high_sims = sum(1 for s in max_sims if s > 0.95)
print(f"Exact string matches: {exact_matches} / {len(test)}")
print(f"Cosine > 0.95 matches: {high_sims} / {len(test)}")
