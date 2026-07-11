import json
import numpy as np
from src.retrieval_service import engine

# ------------------------------
# Configuration
# ------------------------------
GT_PATH = "data/gt_data/gt_faculty.json"
K_VALUES = [1, 3, 5]

# ------------------------------
# Load ground truth
# ------------------------------
def load_gt(path):
    """Returns list of dicts with keys: 'question', 'expected_ids' (list of ints)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Handle both possible formats
    if isinstance(data, dict) and "answerable" in data and "answerable_ref" in data:
        queries = []
        for q, (doc_id, name) in zip(data["answerable"], data["answerable_ref"]):
            queries.append({"question": q, "expected_ids": [doc_id]})
        return queries
    else:
        # Assume list of dicts
        return data

gt_queries = load_gt(GT_PATH)
print(f"Loaded {len(gt_queries)} ground‑truth queries.")

# ------------------------------
# Helper to extract retrieved IDs from a retriever's output
# ------------------------------
def get_retrieved_ids(retriever_data):
    """
    Given the result from one retriever (semantic/lexical or fused),
    return a list of document IDs in rank order.
    Uses the 'document_id' stored in metadata during ingestion.
    """
    if isinstance(retriever_data, dict) and "ids" in retriever_data:
        # For semantic/lexical retriever - extract document_id from chunks
        chunks = retriever_data.get("chunks", [])
        return [chunk["metadata"].get("document_id") for chunk in chunks if chunk["metadata"].get("document_id") is not None]
    elif isinstance(retriever_data, list):
        # For fused (list of result dicts) - extract document_id from metadata
        return [item["metadata"].get("document_id") for item in retriever_data if item["metadata"].get("document_id") is not None]
    else:
        return []

# ------------------------------
# Evaluate a single query
# ------------------------------
def evaluate_query(query, engine, k_values):
    """Return dict: retriever_name -> {k: recall (0 or 1)}."""
    expected = set(query["expected_ids"])
    results = engine.hybrid_search(query["question"])
    recalls = {}
    for ret_name, ret_data in results.items():
        retrieved = get_retrieved_ids(ret_data)
        recalls[ret_name] = {}
        for k in k_values:
            top_k = retrieved[:k]
            recalls[ret_name][k] = 1.0 if any(e in top_k for e in expected) else 0.0
    return recalls

# ------------------------------
# Run evaluation over all queries
# ------------------------------
# We'll aggregate scores per retriever and k
retriever_names = set()
for gt in gt_queries:
    rec = evaluate_query(gt, engine, K_VALUES)
    retriever_names.update(rec.keys())

# Initialize accumulators
accum = {ret: {k: [] for k in K_VALUES} for ret in retriever_names}

for gt in gt_queries:
    rec = evaluate_query(gt, engine, K_VALUES)
    for ret in retriever_names:
        for k in K_VALUES:
            accum[ret][k].append(rec.get(ret, {}).get(k, 0.0))

# Compute averages
avg_recalls = {}
for ret in retriever_names:
    avg_recalls[ret] = {}
    for k in K_VALUES:
        scores = accum[ret][k]
        avg_recalls[ret][k] = np.mean(scores) if scores else 0.0

# ------------------------------
# Display matrix
# ------------------------------
print("\n" + "="*70)
print("📊 RECALL@K EVALUATION MATRIX")
print("="*70)
print(f"{'Retriever':<15} | " + " | ".join(f"Recall@{k}" for k in K_VALUES))
print("-"*70)
for ret in sorted(retriever_names):
    row = f"{ret:<15} | " + " | ".join(f"{avg_recalls[ret][k]:.4f}" for k in K_VALUES)
    print(row)
print("="*70)