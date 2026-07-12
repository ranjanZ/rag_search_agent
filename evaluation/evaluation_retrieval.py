
"""
Retrieval Evaluation Script for QA Datasets

This script evaluates retrieval performance (Recall@K, MRR, NDCG@K)
for any configured dataset (SQuAD v2, NewsQA, or local datasets).

Usage:
    python evaluation_retrieval.py --datasets squad_v2 --split validation --sample_size 100
    python evaluation_retrieval.py --datasets newsqa --split test --sample_size 100
    python evaluation_retrieval.py --datasets faculty --gt_path data/gt_data/gt_faculty.json

Metrics:
    - Recall@K
    - Precision@K
    - MRR (Mean Reciprocal Rank)
    - NDCG@K
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Import config first (before retrieval_service to avoid auto-initialization)
from src.config import DATASET_CONFIG, DEFAULT_DATASETS


def lazy_import_retrieval_engine():
    """Lazy import of retrieval engine to avoid initialization at import time."""
    from src.retrieval_service import engine
    return engine


def load_dataset_for_evaluation(dataset_name: str, split: str = "validation",
                                 sample_size: Optional[int] = None,
                                 random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Load dataset for evaluation based on config.

    Args:
        dataset_name: Name of dataset from DATASET_CONFIG
        split: Dataset split
        sample_size: Number of samples to use
        random_state: Random seed

    Returns:
        List of evaluation examples with question, context, answers, is_answerable
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 'datasets' library not installed. Run: pip install datasets")
        return []

    if dataset_name not in DATASET_CONFIG:
        print(f"❌ Dataset '{dataset_name}' not found in DATASET_CONFIG")
        return []

    config = DATASET_CONFIG[dataset_name]

    if config.get('type') != 'huggingface':
        print(f"⚠️ Dataset '{dataset_name}' is not a HuggingFace dataset. Skipping.")
        return []

    hf_name = config.get('name', dataset_name)
    config_split = config.get('split', split)
    config_sample = config.get('sample_size', sample_size)

    print(f"Loading {hf_name} (split={config_split})...")
    dataset = load_dataset(hf_name, split=config_split)

    if config_sample and len(dataset) > config_sample:
        dataset = dataset.shuffle(seed=random_state).select(range(config_sample))
        print(f"Sampled {config_sample} examples from {hf_name}")

    eval_data = []
    for i, example in enumerate(dataset):
        doc_id = example.get('id', f"{dataset_name}_{i}")
        question = example.get('question', '').strip()
        context = example.get('context', '').strip()

        # Extract answers
        answers = example.get('answers', {})
        if isinstance(answers, dict):
            answer_texts = answers.get('text', [])
        elif isinstance(answers, list):
            answer_texts = answers
        else:
            answer_texts = []

        is_answerable = len(answer_texts) > 0 and answer_texts[0].strip()

        eval_data.append({
            'id': doc_id,
            'question': question,
            'context': context,
            'answers': answer_texts if is_answerable else [],
            'is_answerable': is_answerable
        })

    print(f"✅ Loaded {len(eval_data)} examples from {dataset_name}")
    return eval_data


def load_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth data from JSON file.

    Supports multiple formats:
    1. List of dicts with 'question', 'expected_ids', 'answers', 'is_answerable'
    2. Dict with 'answerable' and 'answerable_ref' keys (legacy format)
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle legacy format
    if isinstance(data, dict) and "answerable" in data and "answerable_ref" in data:
        queries = []
        for q, (doc_id, name) in zip(data["answerable"], data["answerable_ref"]):
            queries.append({
                "question": q,
                "expected_ids": [doc_id],
                "answers": [name],  # Assume name is the answer
                "is_answerable": True
            })
        return queries

    # Handle list format
    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported ground truth format in {gt_path}")


def get_retrieved_ids(retriever_data) -> List[str]:
    """
    Extract document IDs from retriever output.
    """
    if isinstance(retriever_data, dict) and "ids" in retriever_data:
        chunks = retriever_data.get("chunks", [])
        return [
            str(chunk["metadata"].get("document_id"))
            for chunk in chunks
            if chunk["metadata"].get("document_id") is not None
        ]
    elif isinstance(retriever_data, list):
        return [
            str(item["metadata"].get("document_id"))
            for item in retriever_data
            if item["metadata"].get("document_id") is not None
        ]
    return []


# =====================
# RETRIEVAL METRICS
# =====================

def recall_at_k(retrieved: List[Any], expected: set, k: int) -> float:
    """Calculate Recall@K: fraction of relevant documents retrieved in top-k."""
    if not expected:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & expected) / len(expected)


def precision_at_k(retrieved: List[Any], expected: set, k: int) -> float:
    """Calculate Precision@K: fraction of retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & expected) / k


def mean_reciprocal_rank(retrieved: List[Any], expected: set) -> float:
    """Calculate MRR: reciprocal of rank of first relevant document."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in expected:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(scores: List[float], k: int) -> float:
    """Calculate DCG@K."""
    scores = scores[:k]
    return sum(score / np.log2(i + 2) for i, score in enumerate(scores))


def ndcg_at_k(retrieved: List[Any], expected: set, k: int) -> float:
    """Calculate NDCG@K with binary relevance."""
    relevance = [1.0 if doc_id in expected else 0.0 for doc_id in retrieved[:k]]
    dcg = dcg_at_k(relevance, k)

    # Ideal DCG
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_old(query: str, expected_ids: set, k_values: List[int]) -> Dict[str, Any]:
    """
    Evaluate retrieval for a single query.

    Returns metrics for all retrievers (semantic, lexical, name_lexical, fused).
    """
    engine = lazy_import_retrieval_engine()
    results = engine.hybrid_search(query)
    metrics = {}

    for retriever_name, retriever_data in results.items():
        retrieved_ids = get_retrieved_ids(retriever_data)

        retriever_metrics = {}

        # Recall@K
        for k in k_values:
            retriever_metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, expected_ids, k)

        # Precision@K
        for k in k_values:
            retriever_metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, expected_ids, k)

        # MRR
        retriever_metrics['mrr'] = mean_reciprocal_rank(retrieved_ids, expected_ids)

        # NDCG@K
        for k in k_values:
            retriever_metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, expected_ids, k)

        metrics[retriever_name] = retriever_metrics

    return metrics



def evaluate_retrieval(
    query: str, 
    expected_ids: set, 
    k_values: List[int],
    semantic_threshold=0.3,
    lexical_threshold=0.01    
) -> Dict[str, Any]:
    """
    Evaluate retrieval for a single query.

    Returns metrics for all retrievers (semantic, lexical, name_lexical, fused).
    
    Args:
        query: The search query.
        expected_ids: Set of expected chunk IDs.
        k_values: List of K values for K-based metrics (e.g., [1, 3, 5, 10]).
        semantic_threshold: Minimum Cosine Similarity for FAISS (default: 0.3).
        lexical_threshold: Minimum BM25 Score for Lexical Search (default: 0.01).
    """
    engine = lazy_import_retrieval_engine()
    
    # Pass the dynamic thresholds to the hybrid_search method
    results = engine.hybrid_search(
        query, 
        semantic_threshold=semantic_threshold, 
        lexical_threshold=lexical_threshold
    )
    metrics = {}

    for retriever_name, retriever_data in results.items():
        retrieved_ids = get_retrieved_ids(retriever_data)

        retriever_metrics = {}

        # Recall@K
        for k in k_values:
            retriever_metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, expected_ids, k)

        # Precision@K
        for k in k_values:
            retriever_metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, expected_ids, k)

        # MRR
        retriever_metrics['mrr'] = mean_reciprocal_rank(retrieved_ids, expected_ids)

        # NDCG@K
        for k in k_values:
            retriever_metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, expected_ids, k)

        metrics[retriever_name] = retriever_metrics

    return metrics


# Ensure SEMANTIC_THRESHOLD and LEXICAL_THRESHOLD are imported or defined in this file
# from src.config import SEMANTIC_THRESHOLD, LEXICAL_THRESHOLD

def run_retrieval_evaluation(
    eval_data: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
    semantic_threshold=0.3,  
    lexical_threshold=0.01     #
) -> Dict[str, Any]:
    """
    Run retrieval evaluation on dataset.

    For SQuAD/NewsQA: Use document ID as expected
    For local datasets: Use expected_ids from ground truth
    
    Args:
        eval_data: List of evaluation examples.
        k_values: List of K values for K-based metrics.
        semantic_threshold: Minimum Cosine Similarity for FAISS (default: 0.3).
        lexical_threshold: Minimum BM25 Score for Lexical Search (default: 0.01).
    """
    print("\n" + "="*70)
    print("🔍 RUNNING RETRIEVAL EVALUATION")
    # Print the active thresholds so you know what configuration is running
    print(f"📏 Active Thresholds -> Semantic: {semantic_threshold} | Lexical: {lexical_threshold}")
    print("="*70)

    accumulators = defaultdict(lambda: defaultdict(list))

    for i, example in enumerate(eval_data):
        question = example['question']

        # Determine expected IDs
        if 'expected_ids' in example:
            expected_ids = set(str(id) for id in example['expected_ids'])
        else:
            # Use document ID from the example
            expected_ids = {str(example.get('id', f'unknown_{i}'))}

        if not expected_ids or expected_ids == {'None'}:
            continue

        try:
            # Pass the thresholds down to the evaluate_retrieval function
            metrics = evaluate_retrieval(
                question, 
                expected_ids, 
                k_values,
                semantic_threshold=semantic_threshold,
                lexical_threshold=lexical_threshold
            )

            for retriever_name, retriever_metrics in metrics.items():
                for metric_name, value in retriever_metrics.items():
                    accumulators[retriever_name][metric_name].append(value)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(eval_data)} queries...")

        except Exception as e:
            print(f"⚠️ Error evaluating query {i}: {e}")
            continue

    # Compute averages
    results = {}
    for retriever_name, metrics in accumulators.items():
        results[retriever_name] = {}
        for metric_name, values in metrics.items():
            results[retriever_name][metric_name] = np.mean(values) if values else 0.0

    return results


def print_retrieval_results(results: Dict[str, Any], k_values: List[int]):
    """Print retrieval evaluation results in a formatted table."""
    print("\n" + "="*90)
    print("📈 RETRIEVAL EVALUATION RESULTS")
    print("="*90)

    # Header
    header = f"{'Retriever':<15} | "
    header += " | ".join([f"R@{k}" for k in k_values])
    header += " | "
    header += " | ".join([f"P@{k}" for k in k_values])
    header += " | MRR   | "
    header += " | ".join([f"N@{k}" for k in k_values])
    print(header)
    print("-" * 90)

    # Rows
    for retriever_name in sorted(results.keys()):
        metrics = results[retriever_name]
        row = f"{retriever_name:<15} | "

        # Recall@K
        row += " | ".join([f"{metrics.get(f'recall@{k}', 0):.4f}" for k in k_values])
        row += " | "

        # Precision@K
        row += " | ".join([f"{metrics.get(f'precision@{k}', 0):.4f}" for k in k_values])
        row += " | "

        # MRR
        row += f"{metrics.get('mrr', 0):.4f} | "

        # NDCG@K
        row += " | ".join([f"{metrics.get(f'ndcg@{k}', 0):.4f}" for k in k_values])

        print(row)

    print("="*90)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Evaluation Script")
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                       help='Dataset names to evaluate (from DATASET_CONFIG)')
    parser.add_argument('--split', type=str, default='validation',
                       help='Dataset split to use')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to evaluate')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--gt_path', type=str, default=None,
                       help='Path to ground truth file (for local datasets)')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 3, 5],
                       help='K values for Recall@K metrics')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for results (JSON)')

    args = parser.parse_args()

    # Determine datasets to evaluate
    datasets_to_eval = args.datasets if args.datasets else DEFAULT_DATASETS

    print("="*70)
    print("🚀 RETRIEVAL EVALUATION")
    print("="*70)
    print(f"Datasets: {datasets_to_eval}")
    print(f"Split: {args.split}")
    print(f"Sample Size: {args.sample_size}")
    print(f"K Values: {args.k_values}")

    all_retrieval_results = {}

    for dataset_name in datasets_to_eval:
        print(f"\n{'='*70}")
        print(f"Evaluating dataset: {dataset_name}")
        print('='*70)

        # Load evaluation data
        if dataset_name in DATASET_CONFIG and DATASET_CONFIG[dataset_name].get('type') == 'huggingface':
            eval_data = load_dataset_for_evaluation(
                dataset_name,
                args.split,
                args.sample_size,
                args.random_state
            )
        elif args.gt_path:
            eval_data = load_ground_truth(args.gt_path)
            print(f"✅ Loaded {len(eval_data)} examples from ground truth file")
        else:
            print(f"⚠️ Cannot evaluate dataset '{dataset_name}'. Provide --gt_path for local datasets.")
            continue

        if not eval_data:
            print(f"⚠️ No evaluation data for {dataset_name}. Skipping.")
            continue

        # Run retrieval evaluation
        retrieval_results = run_retrieval_evaluation(eval_data, args.k_values)
        all_retrieval_results[dataset_name] = retrieval_results
        print_retrieval_results(retrieval_results, args.k_values)

    # Save results to file
    if args.output:
        output_data = {
            'retrieval': all_retrieval_results,
            'config': {
                'datasets': datasets_to_eval,
                'split': args.split,
                'sample_size': args.sample_size,
                'k_values': args.k_values
            }
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to {args.output}")

    print("\n✅ Retrieval evaluation complete!")


if __name__ == "__main__":
    main()
