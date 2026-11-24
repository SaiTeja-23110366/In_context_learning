import sys
sys.path.insert(0, "/data/pulakurthi.manohar/In_context_learning")
from src.data_utils import load_banking, BANKING_INTENTS_8, BANKING_LABELS_8
from collections import Counter

train, test, labels = load_banking("/data/pulakurthi.manohar/hf_home", seed=0)

print(f"BANKING_LABELS_8: {labels}")
print(f"BANKING_INTENTS_8: {BANKING_INTENTS_8}")
print(f"Exact Train Size: {len(train)}")
print(f"Exact Test Size: {len(test)}")

train_counts = Counter(r["label"] for r in train)
test_counts = Counter(r["label"] for r in test)
train_doms = Counter(r["domain"] for r in train)
test_doms = Counter(r["domain"] for r in test)

print("\nTrain class distribution:")
for i in range(len(labels)):
    print(f"  {labels[i]} ({BANKING_INTENTS_8[i][0]}, {BANKING_INTENTS_8[i][1]}): {train_counts[i]}")

print("\nTest class distribution:")
for i in range(len(labels)):
    print(f"  {labels[i]} ({BANKING_INTENTS_8[i][0]}, {BANKING_INTENTS_8[i][1]}): {test_counts[i]}")

print("\nDomain counts (Train):", dict(train_doms))
print("Domain counts (Test):", dict(test_doms))
