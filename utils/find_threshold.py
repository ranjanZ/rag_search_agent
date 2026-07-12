import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evaluation.evaluation_retrieval import run_retrieval_evaluation, load_ground_truth

def find_optimal_thresholds_and_plot(eval_data, threshold_ranges, k=5):
    """
    Grid search to find optimal thresholds independently for each retriever 
    based strictly on RECALL (Hit Rate), and plot the respective curves.
    """
    results_data = []
    
    # Track best RECALL and thresholds for EACH retriever independently
    best_recall = {'semantic': -1.0, 'lexical': -1.0, 'fused': -1.0}
    best_thresholds = {
        'semantic': {'semantic': 0.3}, 
        'lexical': {'lexical': 0.01}, 
        'fused': {'semantic': 0.3, 'lexical': 0.01}
    }
    best_metrics = {'semantic': {}, 'lexical': {}, 'fused': {}}
    
    total_iters = len(threshold_ranges['semantic']) * len(threshold_ranges['lexical'])
    current_iter = 0

    print(f"Starting grid search over {total_iters} combinations...")

    # 1. Run the Grid Search
    for sem_thresh in threshold_ranges['semantic']:
        for lex_thresh in threshold_ranges['lexical']:
            current_iter += 1
            
            # Run evaluation
            metrics = run_retrieval_evaluation(
                eval_data, 
                k_values=[k], 
                semantic_threshold=sem_thresh, 
                lexical_threshold=lex_thresh
            )
            
            # Store metrics and evaluate each retriever independently
            for retriever_name in ['semantic', 'lexical', 'fused']:
                if retriever_name in metrics:
                    recall_val = metrics[retriever_name].get(f'recall@{k}', 0.0)
                    
                    results_data.append({
                        'semantic_threshold': sem_thresh,
                        'lexical_threshold': lex_thresh,
                        'retriever': retriever_name,
                        f'Recall@{k}': recall_val
                    })
                    
                    # Update best based STRICTLY ON RECALL
                    if recall_val > best_recall[retriever_name]:
                        best_recall[retriever_name] = recall_val
                        best_metrics[retriever_name] = metrics[retriever_name]
                        
                        # Record the specific thresholds that achieved this best recall
                        if retriever_name == 'semantic':
                            best_thresholds['semantic'] = {'semantic': sem_thresh}
                        elif retriever_name == 'lexical':
                            best_thresholds['lexical'] = {'lexical': lex_thresh}
                        elif retriever_name == 'fused':
                            best_thresholds['fused'] = {'semantic': sem_thresh, 'lexical': lex_thresh}
                    
            if current_iter % 5 == 0 or current_iter == total_iters:
                print(f"  Progress: {current_iter}/{total_iters} combinations evaluated.")

    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    
    # ==========================================
    # CHART 1: Semantic Retriever Hit Rate vs Semantic Threshold
    # ==========================================
    # The semantic retriever's performance is independent of the lexical threshold.
    # We group by semantic_threshold and take the mean recall to get a clean 1D line.
    df_sem = df[df['retriever'] == 'semantic']
    df_sem_agg = df_sem.groupby('semantic_threshold')[f'Recall@{k}'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_sem_agg, 
        x='semantic_threshold', 
        y=f'Recall@{k}', 
        marker='o',
        color='#1f77b4', # Blue
        linewidth=2.5,
        markersize=8
    )
    plt.title(f'Semantic Retriever Hit Rate (Recall@{k}) vs Semantic Threshold', fontsize=14)
    plt.xlabel('Semantic Threshold (Cosine Similarity)', fontsize=12)
    plt.ylabel(f'Recall@{k} (Hit Rate)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # ==========================================
    # CHART 2: Lexical Retriever Hit Rate vs Lexical Threshold
    # ==========================================
    # The lexical retriever's performance is independent of the semantic threshold.
    # We group by lexical_threshold and take the mean recall to get a clean 1D line.
    df_lex = df[df['retriever'] == 'lexical']
    df_lex_agg = df_lex.groupby('lexical_threshold')[f'Recall@{k}'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_lex_agg, 
        x='lexical_threshold', 
        y=f'Recall@{k}', 
        marker='s', # Square markers
        color='#2ca02c', # Green
        linewidth=2.5,
        markersize=8
    )
    plt.title(f'Lexical Retriever Hit Rate (Recall@{k}) vs Lexical Threshold', fontsize=14)
    plt.xlabel('Lexical Threshold (BM25 Score)', fontsize=12)
    plt.ylabel(f'Recall@{k} (Hit Rate)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # ==========================================
    # CHART 3: Fused Retriever Hit Rate vs Both Thresholds (Heatmap)
    # ==========================================
    # This shows the interpolation of both thresholds and the final fused hit rate.
    df_fused = df[df['retriever'] == 'fused']
    if not df_fused.empty:
        pivot_df = df_fused.pivot_table(
            index='lexical_threshold', 
            columns='semantic_threshold', 
            values=f'Recall@{k}'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt=".3f", 
            cmap="YlGnBu", 
            cbar_kws={'label': f'Recall@{k}'},
            linewidths=.5
        )
        plt.title(f'Fused Retriever Hit Rate (Recall@{k}) Heatmap', fontsize=14)
        plt.xlabel('Semantic Threshold', fontsize=12)
        plt.ylabel('Lexical Threshold', fontsize=12)
        plt.tight_layout()
        plt.show()

    return best_thresholds, best_metrics, df


if __name__ == "__main__":
    # Load your ground truth data
    eval_data = load_ground_truth("data/gt_data/gt_faculty.json")  

    eval_data = load_dataset_for_evaluation(
        "squad_v2",
        DATASET_CONFIG['squad_v2']['split'],
        DATASET_CONFIG['squad_v2']['sample_size'],
        DATASET_CONFIG['squad_v2']['random_state']
    )



    # Decided Threshold Ranges based on typical FAISS/BM25 score distributions
    threshold_ranges = {
        'semantic': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],       # Cosine similarity
        'lexical': [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0] 
    }
    
    # Run the optimization and plotting (Optimizing for Recall@5)
    optimal_thresholds, best_metrics, results_df = find_optimal_thresholds_and_plot(
        eval_data, 
        threshold_ranges, 
        k=5
    )
    
    print("\n" + "="*50)
    print("🏆 OPTIMAL THRESHOLDS FOUND")
    print("="*50)
    print(f"Best Semantic Threshold: {optimal_thresholds['semantic']}")
    print(f"Best Lexical Threshold:  {optimal_thresholds['lexical']}")
    print("\nBest Metrics (Fused Retriever):")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")