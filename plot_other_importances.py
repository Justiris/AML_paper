import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import sys

# Import functions from existing script
sys.path.append('.')
try:
    from feature_importance import get_rf_feature_importance, get_kmeans_feature_importance
except ImportError:
    print("Could not import from feature_importance.py")
    sys.exit(1)

RESULTS_DIR = 'results'
ARTIFACTS_DIR = '/home/qinkai/.gemini/antigravity/brain/9dd07dbb-2275-4977-99f1-d6ceeaa63813'

def plot_importance(features_importances, model_name, top_n=20):
    # Take top N features if there are many
    if len(features_importances) > top_n:
        features_importances = features_importances[:top_n]
        print(f"Truncated to top {top_n} features for {model_name}")

    # Plot 1: Horizontal Bar Chart
    features_importances_asc = sorted(features_importances, key=lambda x: x[1], reverse=False)
    sorted_names = [f[0] for f in features_importances_asc]
    raw_sorted_importances = [f[1] for f in features_importances_asc]
    
    total_importance = sum(raw_sorted_importances)
    sorted_importances = [(imp / total_importance) * 100 for imp in raw_sorted_importances]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(sorted_names))
    height = 0.6
    
    # Generic blue color for these models
    colors = ['#3498db' for _ in sorted_names]
            
    bars = ax.barh(y, sorted_importances, height, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Normalized Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} Feature Importance', fontsize=14, fontweight='bold')
    
    max_val = max(sorted_importances) if sorted_importances else 1
    ax.set_xlim(0, max_val * 1.20)
    
    for i, v in enumerate(sorted_importances):
        ax.text(v + max_val * 0.02, i, f'{v:.1f}%', va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    bar_filename = f'{model_name.lower().replace(" ", "_")}_feature_importance_bar.png'
    bar_save_path = os.path.join(RESULTS_DIR, bar_filename)
    plt.savefig(bar_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Copy to artifacts
    if os.path.exists(ARTIFACTS_DIR):
        shutil.copy(bar_save_path, os.path.join(ARTIFACTS_DIR, bar_filename))

    # Plot 2: Cumulative Line Chart
    features_importances_desc = sorted(features_importances, key=lambda x: x[1], reverse=True)
    sorted_names_desc = [f[0] for f in features_importances_desc]
    raw_sorted_importances_desc = [f[1] for f in features_importances_desc]
    
    normalized_importances = [(imp / total_importance) for imp in raw_sorted_importances_desc]
    cumulative_importance = np.cumsum(normalized_importances)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_names_desc))
    
    ax.plot(x, cumulative_importance * 100, color='#9b59b6', marker='o', linestyle='-', linewidth=2.5, markersize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names_desc, rotation=45, ha='right')
    ax.set_ylabel('Cumulative %', fontsize=12, fontweight='bold', color='#8e44ad')
    ax.set_ylim(0, 105)
    ax.set_title(f'Cumulative Percentage of {model_name} Features', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(cumulative_importance * 100):
        ax.text(i, v - 5, f'{v:.1f}%', ha='center', va='top', fontsize=10, color='#8e44ad', fontweight='bold')
        
    plt.tight_layout()
    line_filename = f'{model_name.lower().replace(" ", "_")}_cumulative_importance.png'
    line_save_path = os.path.join(RESULTS_DIR, line_filename)
    plt.savefig(line_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Copy to artifacts
    if os.path.exists(ARTIFACTS_DIR):
        shutil.copy(line_save_path, os.path.join(ARTIFACTS_DIR, line_filename))

def main():
    print("Extracting Random Forest Importance...")
    try:
        rf_imp = get_rf_feature_importance()
        plot_importance(rf_imp, "Random Forest")
    except Exception as e:
        print(f"Error processing RF: {e}")

    print("\nExtracting K-Means Importance...")
    try:
        kmeans_imp = get_kmeans_feature_importance()
        plot_importance(kmeans_imp, "K-Means")
    except Exception as e:
        print(f"Error processing K-Means: {e}")

if __name__ == '__main__':
    main()
