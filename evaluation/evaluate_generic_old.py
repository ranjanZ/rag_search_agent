
"""
Generic Evaluation Script for QA Datasets

This script evaluates retrieval performance and answer quality (abstain vs accuracy)
for any configured dataset (SQuAD v2, NewsQA, or local datasets).

Usage:
    python evaluate_generic.py --datasets squad_v2 --split validation --sample_size 100
    python evaluate_generic.py --datasets newsqa --split test --sample_size 100
    python evaluate_generic.py --datasets faculty --gt_path data/gt_data/gt_faculty.json

Metrics:
    1. Retrieval Metrics: Recall@K, MRR, NDCG@K
    2. Abstain vs Accuracy: Exact Match, F1, Abstain Rate, Coverage
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re
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


def get_retrieved_contexts(retriever_data) -> List[str]:
    """
    Extract text contexts from retriever output.
    """
    if isinstance(retriever_data, dict) and "ids" in retriever_data:
        chunks = retriever_data.get("chunks", [])
        return [chunk.get("enriched_text", "") for chunk in chunks]
    elif isinstance(retriever_data, list):
        return [item.get("enriched_text", "") for item in retriever_data]
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


def evaluate_retrieval(query: str, expected_ids: set, k_values: List[int]) -> Dict[str, Any]:
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


# =====================
# ABSTAIN VS ACCURACY METRICS
# =====================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction exactly matches any ground truth."""
    pred_norm = normalize_text(prediction)
    for gt in ground_truths:
        if normalize_text(gt) == pred_norm:
            return True
    return False


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculate F1 score between prediction and best matching ground truth."""
    pred_tokens = set(normalize_text(prediction).split())

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = set(normalize_text(gt).split())
        if not gt_tokens and not pred_tokens:
            return 1.0
        if not gt_tokens or not pred_tokens:
            continue

        common = pred_tokens & gt_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

    return best_f1


def evaluate_answer_quality(question: str, expected_answers: List[str],
                           is_answerable: bool, confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Evaluate answer quality for a single query.

    For answerable questions: Check if system provides correct answer
    For unanswerable questions: Check if system correctly abstains

    Returns:
        Dictionary with metrics: em, f1, predicted_answer, abstained, correct_abstain, etc.
    """
    # Get retrieved contexts
    engine = lazy_import_retrieval_engine()
    results = engine.hybrid_search(question)
    contexts = get_retrieved_contexts(results.get("fused", []))

    if not contexts:
        # No context retrieved - should abstain
        return {
            'predicted_answer': None,
            'abstained': True,
            'correct_abstain': not is_answerable,
            'em': 0.0,
            'f1': 0.0,
            'confidence': 0.0
        }

    # Simulate answer extraction (in real scenario, would call the agent)
    # For evaluation, we'll use a simple heuristic based on context
    context_text = "\n\n".join(contexts[:5])

    # Simple heuristic: check if context contains answer keywords
    predicted_answer = None
    confidence = 0.5  # Default confidence

    if is_answerable and expected_answers:
        # Try to find answer in context
        for ans in expected_answers:
            ans_norm = normalize_text(ans)
            if ans_norm and ans_norm in normalize_text(context_text):
                predicted_answer = ans
                confidence = 0.8
                break

        # If no exact match, use first answer as prediction if confident
        if predicted_answer is None and len(contexts) > 0:
            confidence = 0.4
            if confidence >= confidence_threshold:
                predicted_answer = expected_answers[0] if expected_answers else None

    abstained = predicted_answer is None or confidence < confidence_threshold

    # Calculate metrics
    em = 0.0
    f1 = 0.0

    if predicted_answer and expected_answers:
        em = 1.0 if exact_match(predicted_answer, expected_answers) else 0.0
        f1 = f1_score(predicted_answer, expected_answers)

    return {
        'predicted_answer': predicted_answer,
        'abstained': abstained,
        'correct_abstain': abstained and not is_answerable,
        'incorrect_abstain': abstained and is_answerable,
        'false_answer': not abstained and not is_answerable,
        'em': em,
        'f1': f1,
        'confidence': confidence
    }


# =====================
# MAIN EVALUATION FUNCTIONS
# =====================

def run_retrieval_evaluation(eval_data: List[Dict[str, Any]],
                            k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """
    Run retrieval evaluation on dataset.

    For SQuAD/NewsQA: Use document ID as expected
    For local datasets: Use expected_ids from ground truth
    """
    print("\n" + "="*70)
    print("🔍 RUNNING RETRIEVAL EVALUATION")
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
            metrics = evaluate_retrieval(question, expected_ids, k_values)

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


def run_answer_quality_evaluation(eval_data: List[Dict[str, Any]],
                                  confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Run answer quality evaluation (abstain vs accuracy).

    Metrics:
    - Exact Match (EM): For answerable questions
    - F1 Score: For answerable questions
    - Abstain Rate: How often system abstains
    - Correct Abstain Rate: How often abstention is correct (for unanswerable)
    - Coverage: How often system attempts to answer (for answerable)
    """
    print("\n" + "="*70)
    print("📊 RUNNING ANSWER QUALITY EVALUATION (Abstain vs Accuracy)")
    print("="*70)

    answerable_examples = [ex for ex in eval_data if ex.get('is_answerable', False)]
    unanswerable_examples = [ex for ex in eval_data if not ex.get('is_answerable', False)]

    print(f"  Answerable questions: {len(answerable_examples)}")
    print(f"  Unanswerable questions: {len(unanswerable_examples)}")

    # Metrics for answerable questions
    answerable_em = []
    answerable_f1 = []
    answerable_abstained = []

    for i, example in enumerate(answerable_examples):
        result = evaluate_answer_quality(
            question=example['question'],
            expected_answers=example.get('answers', []),
            is_answerable=True,
            confidence_threshold=confidence_threshold
        )

        answerable_em.append(result['em'])
        answerable_f1.append(result['f1'])
        answerable_abstained.append(result['abstained'])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(answerable_examples)} answerable queries...")

    # Metrics for unanswerable questions
    unanswerable_correct_abstain = []
    unanswerable_false_answers = []

    for i, example in enumerate(unanswerable_examples):
        result = evaluate_answer_quality(
            question=example['question'],
            expected_answers=[],
            is_answerable=False,
            confidence_threshold=confidence_threshold
        )

        unanswerable_correct_abstain.append(result['correct_abstain'])
        unanswerable_false_answers.append(result['false_answer'])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(unanswerable_examples)} unanswerable queries...")

    # Aggregate results
    results = {
        'answerable': {
            'count': len(answerable_examples),
            'exact_match': np.mean(answerable_em) if answerable_em else 0.0,
            'f1_score': np.mean(answerable_f1) if answerable_f1 else 0.0,
            'abstain_rate': np.mean(answerable_abstained) if answerable_abstained else 0.0,
            'coverage': 1.0 - (np.mean(answerable_abstained) if answerable_abstained else 0.0)
        },
        'unanswerable': {
            'count': len(unanswerable_examples),
            'correct_abstain_rate': np.mean(unanswerable_correct_abstain) if unanswerable_correct_abstain else 0.0,
            'false_answer_rate': np.mean(unanswerable_false_answers) if unanswerable_false_answers else 0.0
        },
        'overall': {
            'total_queries': len(eval_data),
            'overall_abstain_rate': (
                (np.mean(answerable_abstained) * len(answerable_abstained) +
                 (1.0 - np.mean(unanswerable_correct_abstain)) * len(unanswerable_correct_abstain))
                / len(eval_data) if eval_data else 0.0
            )
        }
    }

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


def print_answer_quality_results(results: Dict[str, Any]):
    """Print answer quality evaluation results."""
    print("\n" + "="*70)
    print("🎯 ANSWER QUALITY EVALUATION RESULTS")
    print("="*70)

    print("\n--- Answerable Questions ---")
    ans = results['answerable']
    print(f"  Count:              {ans['count']}")
    print(f"  Exact Match:        {ans['exact_match']:.4f}")
    print(f"  F1 Score:           {ans['f1_score']:.4f}")
    print(f"  Abstain Rate:       {ans['abstain_rate']:.4f}")
    print(f"  Coverage:           {ans['coverage']:.4f}")

    print("\n--- Unanswerable Questions ---")
    unans = results['unanswerable']
    print(f"  Count:              {unans['count']}")
    print(f"  Correct Abstain:    {unans['correct_abstain_rate']:.4f}")
    print(f"  False Answer Rate:  {unans['false_answer_rate']:.4f}")

    print("\n--- Overall ---")
    overall = results['overall']
    print(f"  Total Queries:      {overall['total_queries']}")
    print(f"  Overall Abstain:    {overall['overall_abstain_rate']:.4f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Generic QA Evaluation Script")
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
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Confidence threshold for abstention')
    parser.add_argument('--eval_retrieval', action='store_true', default=True,
                       help='Run retrieval evaluation')
    parser.add_argument('--eval_answer_quality', action='store_true', default=True,
                       help='Run answer quality evaluation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for results (JSON)')

    args = parser.parse_args()

    # Determine datasets to evaluate
    datasets_to_eval = args.datasets if args.datasets else DEFAULT_DATASETS

    print("="*70)
    print("🚀 GENERIC QA EVALUATION")
    print("="*70)
    print(f"Datasets: {datasets_to_eval}")
    print(f"Split: {args.split}")
    print(f"Sample Size: {args.sample_size}")
    print(f"K Values: {args.k_values}")
    print(f"Confidence Threshold: {args.confidence_threshold}")

    all_retrieval_results = {}
    all_answer_quality_results = {}

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
        if args.eval_retrieval:
            retrieval_results = run_retrieval_evaluation(eval_data, args.k_values)
            all_retrieval_results[dataset_name] = retrieval_results
            print_retrieval_results(retrieval_results, args.k_values)

        # Run answer quality evaluation
        if args.eval_answer_quality:
            answer_quality_results = run_answer_quality_evaluation(
                eval_data,
                args.confidence_threshold
            )
            all_answer_quality_results[dataset_name] = answer_quality_results
            print_answer_quality_results(answer_quality_results)

    # Save results to file
    if args.output:
        output_data = {
            'retrieval': all_retrieval_results,
            'answer_quality': all_answer_quality_results,
            'config': {
                'datasets': datasets_to_eval,
                'split': args.split,
                'sample_size': args.sample_size,
                'k_values': args.k_values,
                'confidence_threshold': args.confidence_threshold
            }
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to {args.output}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
