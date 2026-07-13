import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
import random



# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.config import DATASET_CONFIG, DEFAULT_DATASETS

def lazy_import_chat_agent():
    """Lazy import of chat agent to avoid initialization at import time."""
    from src.chat_agent_service.chat_agent import run_agent
    #from src.chat_agent_service.chat_multi_agent import run_agent

    return run_agent

def load_dataset_for_evaluation(dataset_name: str, split: str = "validation",
                                 sample_size: Optional[int] = None,
                                 random_state: int = 42) -> List[Dict[str, Any]]:
    """Load dataset for evaluation based on config."""
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
        return []

    hf_name = config.get('name', dataset_name)
    config_split = config.get('split', split)
    config_sample = config.get('sample_size', sample_size)

    print(f"Loading {hf_name} (split={config_split})...")
    dataset = load_dataset(hf_name, split=config_split)

    if config_sample and len(dataset) > config_sample:
        dataset = dataset.shuffle(seed=random_state).select(range(config_sample))

    eval_data = []
    for i, example in enumerate(dataset):
        answers = example.get('answers', {})
        answer_texts = answers.get('text', []) if isinstance(answers, dict) else answers
        is_answerable = len(answer_texts) > 0 and answer_texts[0].strip()

        eval_data.append({
            'id': example.get('id', f"{dataset_name}_{i}"),
            'question': example.get('question', '').strip(),
            'answers': answer_texts if is_answerable else [],
            'is_answerable': is_answerable
        })
    return eval_data

def load_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    """Load ground truth data from JSON file."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and "answerable" in data and "answerable_ref" in data:
        queries = []
        for q, (doc_id, name) in zip(data["answerable"], data["answerable_ref"]):
            queries.append({"question": q, "answers": [name], "is_answerable": True})
        for q in data["unanswerable"]:
            queries.append({"question": q, "answers": [], "is_answerable": False})
        return queries
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported ground truth format in {gt_path}")

# ==========================================
# 1. DATA COLLECTION (Runs LLM exactly ONCE)
# ==========================================

def collect_agent_confidences(eval_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Runs the agent on all queries exactly ONCE to collect raw confidence scores.
    This is a massive optimization that allows us to test 100s of thresholds instantly.
    """
    run_agent_fn = lazy_import_chat_agent()
    results = []
    total_start = time.time()
    
    print(f"\n🚀 Running agent on {len(eval_data)} queries to collect confidence scores...")
    
    for i, example in enumerate(eval_data):
        start_time = time.time()
        
        try:
            # We set threshold to 0.0 so the agent NEVER abstains during data collection.
            # This ensures we get the raw, unfiltered confidence score for every query.
            final_state = run_agent_fn(
                query=example['question'],
                chat_history=[],
                use_historical_context=False,
                confidence_threshold=0.0 
            )
            
            confidence_score = final_state.get('confidence_score', 0.0)
            final_answer = final_state.get('final_answer', '')
            
            # Check if the LLM textually abstained (e.g., "I don't know")
            abstain_phrases = ["i'm sorry", "cannot answer", "don't know", "do not know", 
                               "unable to answer", "not enough information", "insufficient"]
            text_abstained = any(phrase in final_answer.lower() for phrase in abstain_phrases)
            
        except Exception as e:
            print(f"⚠️ Error evaluating query {i}: {e}")
            confidence_score = 0.0
            text_abstained = True
            
        end_time = time.time()
        
        results.append({
            'is_answerable': example.get('is_answerable'),
            'confidence': confidence_score,
            'text_abstained': text_abstained,
            'time_taken': end_time - start_time,
            'query':example['question'],
            'agent_ans':final_state['final_answer']
        })
        
        # 🕒 Progress and Time Estimation
        if (i + 1) % 5 == 0 or (i + 1) == len(eval_data):
            avg_time = np.mean([r['time_taken'] for r in results])
            remaining_queries = len(eval_data) - (i + 1)
            est_time_remaining = avg_time * remaining_queries
            
            print(f"  Progress: {i+1}/{len(eval_data)} | "
                  f"Avg time/query: {avg_time:.2f}s | "
                  f"Est. time remaining: {est_time_remaining:.1f}s")
                  
    total_time = time.time() - total_start
    print(f"\n✅ Data collection complete. Total time: {total_time:.2f}s")
    return results

# ==========================================
# 2. THRESHOLD SWEEP & PLOTTING
# ==========================================

def evaluate_threshold_sweep(agent_results: List[Dict[str, Any]], threshold_range: List[float]) -> pd.DataFrame:
    """
    Calculates abstention metrics for a range of confidence thresholds 
    using the pre-collected confidence scores (Takes milliseconds).
    """
    sweep_results = []
    
    for threshold in threshold_range:
        answerable = [r for r in agent_results if r['is_answerable']]
        unanswerable = [r for r in agent_results if not r['is_answerable']]
        
        # Answerable metrics
        abstained_ans = sum(1 for r in answerable if (r['confidence'] < threshold) or r['text_abstained'])
        abstain_rate_ans = abstained_ans / len(answerable) if answerable else 0
        coverage = 1.0 - abstain_rate_ans
        
        # Unanswerable metrics
        correct_abstain = sum(1 for r in unanswerable if (r['confidence'] < threshold) or r['text_abstained'])
        correct_abstain_rate = correct_abstain / len(unanswerable) if unanswerable else 0
        false_answer_rate = 1.0 - correct_abstain_rate
        
        sweep_results.append({
            'threshold': threshold,
            'coverage': coverage,
            'correct_abstain_rate': correct_abstain_rate,
            'false_answer_rate': false_answer_rate
        })
        
    return pd.DataFrame(sweep_results)

def plot_threshold_sweep(sweep_df: pd.DataFrame):
    """Plots the abstention metrics across different confidence thresholds."""
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=sweep_df, x='threshold', y='coverage', marker='o', 
                 label='Coverage (Answerable)', color='blue', linewidth=2.5)
    sns.lineplot(data=sweep_df, x='threshold', y='correct_abstain_rate', marker='s', 
                 label='Correct Abstain (Unanswerable)', color='green', linewidth=2.5)
    sns.lineplot(data=sweep_df, x='threshold', y='false_answer_rate', marker='^', 
                 label='False Answer / Hallucination (Unanswerable)', color='red', linewidth=2.5)
    
    plt.title('Abstention Metrics vs Confidence Threshold', fontsize=14)
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Rate (0.0 to 1.0)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. SINGLE THRESHOLD EVALUATION
# ==========================================

def run_answer_quality_evaluation(eval_data: List[Dict[str, Any]], confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """Run answer quality evaluation for a SINGLE threshold."""
    print("\n" + "="*70)
    print(f"📊 RUNNING EVALUATION FOR THRESHOLD: {confidence_threshold}")
    print("="*70)

    # We reuse the optimized collection and sweep for a single threshold
    agent_results = collect_agent_confidences(eval_data)
    sweep_df = evaluate_threshold_sweep(agent_results, [confidence_threshold])
    row = sweep_df.iloc[0]
    
    answerable_count = sum(1 for r in agent_results if r['is_answerable'])
    unanswerable_count = sum(1 for r in agent_results if not r['is_answerable'])
    
    # Calculate true overall abstain rate (Fixed bug from previous version)
    total_abstains = ((1.0 - row['coverage']) * answerable_count) + (row['correct_abstain_rate'] * unanswerable_count)
    
    return {
        'answerable': {
            'count': answerable_count,
            'abstain_rate': 1.0 - row['coverage'],
            'coverage': row['coverage']
        },
        'unanswerable': {
            'count': unanswerable_count,
            'correct_abstain_rate': row['correct_abstain_rate'],
            'false_answer_rate': row['false_answer_rate']
        },
        'overall': {
            'total_queries': len(eval_data),
            'overall_abstain_rate': total_abstains / len(eval_data) if eval_data else 0.0
        }
    }



def compute_result(agent_results, confidence_threshold: float = 0.6) -> Dict[str, Any]:
    sweep_df = evaluate_threshold_sweep(agent_results, [confidence_threshold])
    row = sweep_df.iloc[0]
    
    answerable_count = sum(1 for r in agent_results if r['is_answerable'])
    unanswerable_count = sum(1 for r in agent_results if not r['is_answerable'])
    
    # Calculate true overall abstain rate (Fixed bug from previous version)
    total_abstains = ((1.0 - row['coverage']) * answerable_count) + (row['correct_abstain_rate'] * unanswerable_count)
    
    return {
        'answerable': {
            'count': answerable_count,
            'abstain_rate': 1.0 - row['coverage'],
            'coverage': row['coverage']
        },
        'unanswerable': {
            'count': unanswerable_count,
            'correct_abstain_rate': row['correct_abstain_rate'],
            'false_answer_rate': row['false_answer_rate']
        },
        'overall': {
            'total_queries': len(eval_data),
            'overall_abstain_rate': total_abstains / len(eval_data) if eval_data else 0.0
        }
    }



def print_answer_quality_results(results: Dict[str, Any]):
    """Print answer quality evaluation results."""
    print("\n" + "="*70)
    print("🎯 ANSWER QUALITY EVALUATION RESULTS")
    print("="*70)

    print("\n--- Answerable Questions ---")
    ans = results['answerable']
    print(f"  Count:              {ans['count']}")
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

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer Quality Evaluation Script")
    parser.add_argument('--datasets', nargs='+', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--gt_path', type=str, default="data/gt_data/gt_faculty.json")
    parser.add_argument('--confidence_threshold', type=float, default=0.6)
    
    # NEW: Add a flag to run the threshold sweep and plot
    parser.add_argument('--sweep', action='store_true', default=True, 
                       help='Run a threshold sweep and plot the results')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()
    datasets_to_eval = args.datasets if args.datasets else DEFAULT_DATASETS

    print("="*70)
    print("🚀 ANSWER QUALITY EVALUATION")
    print("="*70)
    print(f"Mode: {'📈 THRESHOLD SWEEP & PLOT' if args.sweep else f'🎯 Single Threshold ({args.confidence_threshold})'}")

    for dataset_name in datasets_to_eval:
        print(f"\n{'='*70}")
        print(f"Evaluating dataset: {dataset_name}")
        print('='*70)

        # Load data
        if dataset_name in DATASET_CONFIG and DATASET_CONFIG[dataset_name].get('type') == 'huggingface':
            eval_data = load_dataset_for_evaluation(dataset_name, args.split, args.sample_size)
        elif args.gt_path:
            eval_data = load_ground_truth(args.gt_path)
        else:
            continue

        if not eval_data:
            continue

        eval_data= random.sample(eval_data, 100)

        if args.sweep:
            # --- SWEEP MODE ---
            print("\n🔄 RUNNING THRESHOLD SWEEP...")
            threshold_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            # 1. Collect confidences ONCE (Measures time and estimates completion)
            agent_results = collect_agent_confidences(eval_data)
        
            with open(f'data/output/{dataset_name}.json', 'w') as fout:
                json.dump(agent_results, fout)


            # 2. Evaluate all thresholds instantly
            sweep_df = evaluate_threshold_sweep(agent_results, threshold_range)
            
            # 3. Plot the results
            plot_threshold_sweep(sweep_df)
            
            # 4. Recommend the best threshold
            valid_sweep = sweep_df[sweep_df['coverage'] >= 0.80] # Keep at least 80% coverage
            if not valid_sweep.empty:
                best_idx = valid_sweep['correct_abstain_rate'].idxmax()
                best_threshold = valid_sweep.loc[best_idx, 'threshold']
                print(f"\n🏆 RECOMMENDED THRESHOLD: {best_threshold} "
                      f"(Maximizes Correct Abstain while keeping Coverage >= 80%)")
                
                answer_quality_results=compute_result(agent_results, confidence_threshold=best_threshold)
                print_answer_quality_results(answer_quality_results)


            else:
                print("\n⚠️ No threshold found that maintains >= 80% Coverage.")
                
        else:
            # --- SINGLE THRESHOLD MODE ---
            answer_quality_results = run_answer_quality_evaluation(eval_data, args.confidence_threshold)
            print_answer_quality_results(answer_quality_results)

    print("\n✅ Evaluation complete!")

#if __name__ == "__main__":
#    main()
