import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil

RESULTS_DIR = 'results'
ARTIFACTS_DIR = '/home/qinkai/.gemini/antigravity/brain/9dd07dbb-2275-4977-99f1-d6ceeaa63813'

def main():
    # Load explanations data
    data_path = os.path.join(RESULTS_DIR, 'gnn_explanations.npz')
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    data = np.load(data_path, allow_pickle=True)
    
    if 'feature_names' in data and 'importance_means' in data:
        feature_names = data['feature_names']
        importance_means = data['importance_means']
    else:
        print("Required arrays not found in npz file.")
        return

    # Pair and sort by importance ascending for horizontal bar chart
    features_importances = list(zip(feature_names, importance_means))
    features_importances.sort(key=lambda x: x[1], reverse=False)
    
    sorted_names = [f[0] for f in features_importances]
    raw_sorted_importances = [f[1] for f in features_importances]
    
    # Normalize to percentages
    total_importance = sum(raw_sorted_importances)
    sorted_importances = [(imp / total_importance) * 100 for imp in raw_sorted_importances]
    
    # -------------------------------------------------------------
    # Plot 1: Horizontal Bar Chart of Importance Scores
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y = np.arange(len(sorted_names))
    height = 0.6
    
    colors = []
    for name in sorted_names:
        if name.startswith('payment_'):
            colors.append('#2ecc71')  # Green
        elif name in ['amount', 'hour']:
            colors.append('#3498db')  # Blue
        else:
            colors.append('#e74c3c')  # Red
            
    bars = ax.barh(y, sorted_importances, height, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Normalized Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('GNN Edge Feature Importance', fontsize=14, fontweight='bold')
    
    # Extend x-axis limit to prevent text from being cut off
    max_val = max(sorted_importances) if sorted_importances else 1
    ax.set_xlim(0, max_val * 1.20)
    
    # Add value labels horizontally
    for i, v in enumerate(sorted_importances):
        ax.text(v + max_val * 0.02, i, f'{v:.1f}%', va='center', ha='left', fontsize=10)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, edgecolor='black', label='Transaction Features'),
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Cross-border Features'),
        Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black', label='Payment Type')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    bar_save_path = os.path.join(RESULTS_DIR, 'gnn_feature_importance_bar.png')
    plt.savefig(bar_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    shutil.copy(bar_save_path, os.path.join(ARTIFACTS_DIR, 'gnn_feature_importance_bar.png'))
    print(f"Saved bar chart to {bar_save_path} and Artifacts Dir")

    # -------------------------------------------------------------
    # Plot 2: Line Chart of Cumulative Importance
    # -------------------------------------------------------------
    # Re-sort descending to calculate meaningful cumulative importance
    features_importances.sort(key=lambda x: x[1], reverse=True)
    sorted_names_desc = [f[0] for f in features_importances]
    sorted_importances_desc = [f[1] for f in features_importances]
    
    # Normalize to sum to 100% for the cumulative chart
    total_importance = sum(sorted_importances_desc)
    normalized_importances = [imp / total_importance for imp in sorted_importances_desc]
    
    cumulative_importance = np.cumsum(normalized_importances)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_names_desc))
    
    # Plot as percentages (multiply by 100)
    ax.plot(x, cumulative_importance * 100, color='#9b59b6', marker='o', linestyle='-', linewidth=2.5, markersize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names_desc, rotation=45, ha='right')
    ax.set_ylabel('Cumulative %', fontsize=12, fontweight='bold', color='#8e44ad')
    ax.set_ylim(0, 105)
    ax.set_title('Cumulative Percentage of GNN Edge Features', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(cumulative_importance * 100):
        ax.text(i, v - 5, f'{v:.1f}%', ha='center', va='top', fontsize=10, color='#8e44ad', fontweight='bold')
        
    plt.tight_layout()
    line_save_path = os.path.join(RESULTS_DIR, 'gnn_cumulative_importance.png')
    plt.savefig(line_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    shutil.copy(line_save_path, os.path.join(ARTIFACTS_DIR, 'gnn_cumulative_importance.png'))
    print(f"Saved cumulative line chart to {line_save_path} and Artifacts Dir")

if __name__ == '__main__':
    main()
