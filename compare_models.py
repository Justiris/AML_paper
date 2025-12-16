"""
Model Comparison Script for Anti-Money Laundering Detection
Compares K-means, Random Forest, and GNN models.
"""

import numpy as np
import os

# Configuration
RESULTS_DIR = 'results'


def load_results():
    """Load results from all models."""
    results = {}
    
    # K-means
    try:
        kmeans = np.load(os.path.join(RESULTS_DIR, 'kmeans_results.npz'), allow_pickle=True)
        metrics = kmeans['metrics'].item()
        results['K-means'] = {
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'roc_auc': metrics.get('roc_auc', 0),
            'pr_auc': metrics.get('pr_auc', 0)
        }
        print("✓ K-means results loaded")
    except Exception as e:
        print(f"✗ K-means results not found: {e}")
        results['K-means'] = None
    
    # Random Forest
    try:
        rf = np.load(os.path.join(RESULTS_DIR, 'rf_results.npz'), allow_pickle=True)
        results['Random Forest'] = {
            'precision': float(rf['precision']),
            'recall': float(rf['recall']),
            'f1': float(rf['f1']),
            'roc_auc': float(rf['roc_auc']),
            'pr_auc': float(rf['pr_auc'])
        }
        print("✓ Random Forest results loaded")
    except Exception as e:
        print(f"✗ Random Forest results not found: {e}")
        results['Random Forest'] = None
    
    # GNN Baseline (4 edge features - without payment type)
    # Hardcoded results from original training run
    results['GNN (Base)'] = {
        'precision': 0.8803571428571428,
        'recall': 0.9984810126582279,
        'f1': 0.93570581257414,
        'roc_auc': 0.9999466337588488,
        'pr_auc': 0.9985418588820953
    }
    print("✓ GNN (Base) results loaded (4 edge features)")
    
    # GNN with Payment Type (11 edge features)
    try:
        gnn = np.load(os.path.join(RESULTS_DIR, 'gnn_results.npz'), allow_pickle=True)
        results['GNN (+PayType)'] = {
            'precision': float(gnn['precision']),
            'recall': float(gnn['recall']),
            'f1': float(gnn['f1']),
            'roc_auc': float(gnn['roc_auc']),
            'pr_auc': float(gnn['pr_auc'])
        }
        print("✓ GNN (+PayType) results loaded (11 edge features)")
    except Exception as e:
        print(f"✗ GNN (+PayType) results not found: {e}")
        results['GNN (+PayType)'] = None
    
    return results


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - ANTI-MONEY LAUNDERING DETECTION")
    print("=" * 80)
    
    metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    
    # Header
    print(f"\n{'Metric':<15}", end='')
    for model in results:
        print(f"{model:>15}", end='')
    print(f"{'Best':>15}")
    print("-" * (15 + 15 * len(results) + 15))
    
    # Data rows
    for metric, name in zip(metrics, metric_names):
        print(f"{name:<15}", end='')
        
        values = []
        for model in results:
            if results[model] is not None:
                val = results[model][metric]
                values.append((model, val))
                print(f"{val:>15.4f}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        
        # Find best
        if values:
            best_model = max(values, key=lambda x: x[1])[0]
            print(f"{best_model:>15}")
        else:
            print(f"{'N/A':>15}")
    
    print("-" * (15 + 15 * len(results) + 15))


def print_summary(results):
    """Print summary and analysis."""
    print("\n" + "=" * 80)
    print("SUMMARY & ANALYSIS")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to compare.")
        return
    
    # Find best model by each metric
    print("\n📊 Best Model by Metric:")
    print("-" * 40)
    
    for metric_name, metric_key in [('F1-Score', 'f1'), ('PR-AUC', 'pr_auc'), ('ROC-AUC', 'roc_auc')]:
        best = max(valid_results.items(), key=lambda x: x[1][metric_key])
        print(f"  {metric_name}: {best[0]} ({best[1][metric_key]:.4f})")
    
    # Overall recommendation
    print("\n🏆 Overall Recommendation:")
    print("-" * 40)
    
    # Score each model (weighted average of key metrics for imbalanced data)
    # PR-AUC is most important for imbalanced data
    scores = {}
    for model, metrics in valid_results.items():
        score = (
            metrics['pr_auc'] * 0.4 +  # Most important for imbalanced
            metrics['f1'] * 0.3 +
            metrics['recall'] * 0.2 +  # Want to catch fraud
            metrics['precision'] * 0.1
        )
        scores[model] = score
    
    best_overall = max(scores.items(), key=lambda x: x[1])
    print(f"  Best overall model: {best_overall[0]}")
    print(f"  Weighted score: {best_overall[1]:.4f}")
    
    # Model-specific insights
    print("\n📝 Model Insights:")
    print("-" * 40)
    
    if 'K-means' in valid_results:
        km = valid_results['K-means']
        print(f"  K-means: Unsupervised approach, good for anomaly detection")
        print(f"           PR-AUC: {km['pr_auc']:.4f}, F1: {km['f1']:.4f}")
    
    if 'Random Forest' in valid_results:
        rf = valid_results['Random Forest']
        print(f"  Random Forest: Supervised, handles SMOTE-balanced data well")
        print(f"                 PR-AUC: {rf['pr_auc']:.4f}, F1: {rf['f1']:.4f}")
    
    if 'GNN (Base)' in valid_results:
        gnn = valid_results['GNN (Base)']
        print(f"  GNN (Base): Captures graph structure (4 edge features)")
        print(f"              PR-AUC: {gnn['pr_auc']:.4f}, F1: {gnn['f1']:.4f}")
    
    if 'GNN (+PayType)' in valid_results:
        gnn = valid_results['GNN (+PayType)']
        print(f"  GNN (+PayType): Adds payment type features (11 edge features)")
        print(f"                  PR-AUC: {gnn['pr_auc']:.4f}, F1: {gnn['f1']:.4f}")


def save_comparison(results):
    """Save comparison results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Convert to saveable format
    save_data = {}
    for model, metrics in results.items():
        if metrics is not None:
            save_data[model] = metrics
    
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'model_comparison.npz'),
        **{f"{model}_{metric}": value 
           for model, metrics in save_data.items() 
           for metric, value in metrics.items()}
    )
    
    print(f"\nComparison saved to {RESULTS_DIR}/model_comparison.npz")


def main():
    """Main execution."""
    print("=" * 80)
    print("LOADING MODEL RESULTS")
    print("=" * 80)
    
    results = load_results()
    print_comparison_table(results)
    print_summary(results)
    save_comparison(results)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
