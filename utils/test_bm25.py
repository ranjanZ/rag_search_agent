import pickle
import numpy as np
from rank_bm25 import BM25Okapi

# Load the NEW name_bm25 index
with open("data/indexes/name_bm25.pkl", 'rb') as f:
    name_bm25 = pickle.load(f)
# Test queries
queries = [
    "who is Prof Salam",
    "Salem",
    "Salem Lahlou",
    "Prof Salem"
]
print("=== Testing NEW Name BM25 Index ===\n")
for query in queries:
    tokenized_query = query.lower().split()
    scores = name_bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:5]
    print(f"Query: '{query}'")
    print(f"Top 5 indices: {top_indices}")
    print(f"Scores: {scores[top_indices]}")
    # Show what names are at those indices
    for idx in top_indices[:3]:
        doc_freq = name_bm25.doc_freqs[idx]
        # Find name tokens (usually boosted with count 2)
        name_tokens = [k for k, v in doc_freq.items() if v >= 2 and len(k.split()) <= 3]
        if name_tokens:
            print(f"  Index {idx}: {name_tokens[0]}")
    print()