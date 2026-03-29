"""
Feature Importance Analysis for All Models
Compares feature importance across K-means, Random Forest, and GNN models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random

# Configuration
RESULTS_DIR = 'results'
DATA_DIR = 'processed_data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rf_feature_importance():
    """Get Random Forest feature importance."""
    print("\n=== Random Forest Feature Importance ===")
    
    rf = np.load(os.path.join(RESULTS_DIR, 'rf_results.npz'), allow_pickle=True)
    importances = rf['feature_importances']
    names = [str(n) for n in rf['feature_names']]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    result = []
    for idx in indices:
        result.append((names[idx], importances[idx]))
        
    return result


def get_kmeans_feature_importance():
    """Get K-means pseudo feature importance via correlation with anomaly scores."""
    print("\n=== K-means Feature Importance (Anomaly Score Correlation) ===")
    
    # Load K-means data
    kmeans_data = np.load(os.path.join(DATA_DIR, 'kmeans_features.npz'), allow_pickle=True)
    X = kmeans_data['X']
    feature_names = [str(n) for n in kmeans_data['feature_names']]
    
    # Load anomaly scores
    kmeans_results = np.load(os.path.join(RESULTS_DIR, 'kmeans_results.npz'), allow_pickle=True)
    anomaly_scores = kmeans_results['anomaly_scores']
    
    # Compute correlation between each feature and anomaly scores
    correlations = []
    for i in range(X.shape[1]):
        corr = np.abs(np.corrcoef(X[:, i], anomaly_scores)[0, 1])
        correlations.append((feature_names[i], corr if not np.isnan(corr) else 0))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    return correlations


def get_gnn_feature_importance(model_type='paytype'):
    """Get GNN feature importance using gradient-based attribution."""
    print(f"\n=== GNN ({'with PayType' if model_type == 'paytype' else 'Base'}) Feature Importance ===")
    
    # Import GNN model
    try:
        from torch_geometric.nn import SAGEConv
        HAS_PYG = True
    except ImportError:
        HAS_PYG = False
        print("PyTorch Geometric not available")
        return []
    
    # Load data
    gnn_data = np.load(os.path.join(DATA_DIR, 'gnn_graph.npz'), allow_pickle=True)
    node_features = torch.FloatTensor(gnn_data['node_features']).to(DEVICE)
    edge_index = torch.LongTensor(gnn_data['edge_index']).to(DEVICE)
    edge_features_full = gnn_data['edge_features']
    edge_labels = torch.FloatTensor(gnn_data['edge_labels']).to(DEVICE)
    edge_feature_names = list(gnn_data['edge_feature_names'])
    
    # Select features based on model type
    if model_type == 'base':
        edge_features = torch.FloatTensor(edge_features_full[:, :4]).to(DEVICE)
        model_path = os.path.join(RESULTS_DIR, 'gnn_base_model.pt')
        feature_names = [str(n) for n in edge_feature_names[:4]]
    else:
        edge_features = torch.FloatTensor(edge_features_full).to(DEVICE)
        model_path = os.path.join(RESULTS_DIR, 'gnn_model.pt')
        feature_names = [str(n) for n in edge_feature_names]
    
    # Clean up feature names
    feature_names = [n.replace('payment_type_', 'payment_') for n in feature_names]
    
    # Load model architecture (import from model files)
    import torch.nn as nn
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    class EdgeClassifierMLP(nn.Module):
        def __init__(self, node_dim, edge_dim, hidden_dim):
            super().__init__()
            input_dim = node_dim * 2 + edge_dim
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        def forward(self, src_emb, dst_emb, edge_feat):
            x = torch.cat([src_emb, dst_emb, edge_feat], dim=1)
            return self.layers(x).squeeze(-1)
    
    class GraphSAGEEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            if num_layers > 1:
                self.convs.append(SAGEConv(hidden_dim, out_dim))
        
        def forward(self, x, edge_index):
            import torch.nn.functional as F
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=DROPOUT, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x
    
    class GNNModel(nn.Module):
        def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
            super().__init__()
            self.encoder = GraphSAGEEncoder(node_in_dim, hidden_dim, hidden_dim, NUM_LAYERS)
            self.edge_classifier = EdgeClassifierMLP(hidden_dim, edge_in_dim, hidden_dim)
        
        def forward(self, node_features, edge_index, edge_features, batch_edge_idx=None):
            node_emb = self.encoder(node_features, edge_index)
            if batch_edge_idx is not None:
                src_emb = node_emb[edge_index[0, batch_edge_idx]]
                dst_emb = node_emb[edge_index[1, batch_edge_idx]]
            else:
                src_emb = node_emb[edge_index[0]]
                dst_emb = node_emb[edge_index[1]]
            logits = self.edge_classifier(src_emb, dst_emb, edge_features)
            return logits
    
    # Initialize and load model
    model = GNNModel(
        node_in_dim=node_features.shape[1],
        edge_in_dim=edge_features.shape[1],
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Get fraud edges for importance analysis
    fraud_mask = edge_labels > 0.5
    fraud_indices = torch.where(fraud_mask)[0]
    
    # Sample fraud edges for efficiency
    n_samples = min(1000, len(fraud_indices))
    sample_idx = fraud_indices[torch.randperm(len(fraud_indices))[:n_samples]]
    
    # Compute gradient-based feature importance
    feature_importance = torch.zeros(edge_features.shape[1]).to(DEVICE)
    
    edge_features.requires_grad_(True)
    
    for i, idx in enumerate(sample_idx):
        if i % 200 == 0:
            print(f"  Processing {i}/{n_samples}...")
        
        edge_features.grad = None
        
        # Forward pass for single edge
        idx_tensor = torch.tensor([idx.item()]).to(DEVICE)
        batch_edge_feat = edge_features[idx:idx+1]
        batch_edge_feat.requires_grad_(True)
        
        # Get node embeddings
        node_emb = model.encoder(node_features, edge_index)
        src_emb = node_emb[edge_index[0, idx]]
        dst_emb = node_emb[edge_index[1, idx]]
        
        logit = model.edge_classifier(src_emb.unsqueeze(0), dst_emb.unsqueeze(0), batch_edge_feat)
        
        # Backward pass
        logit.backward()
        
        if batch_edge_feat.grad is not None:
            feature_importance += batch_edge_feat.grad.abs().squeeze()
    
    # Average importance
    feature_importance = feature_importance / n_samples
    feature_importance = feature_importance.cpu().numpy()
    
    # Create result
    result = list(zip(feature_names, feature_importance))
    result.sort(key=lambda x: x[1], reverse=True)
    
    return result


def print_feature_importance(name, importance_list, top_n=10):
    """Print feature importance table."""
    print(f"\n{name}:")
    print("-" * 50)
    for i, (feature, imp) in enumerate(importance_list[:top_n]):
        print(f"  {i+1:2d}. {feature:<30} {imp:.4f}")


def plot_feature_importance_comparison(all_importances, save_path=None):
    """Create comparison plot of feature importance across models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
    
    titles = ['K-means (Anomaly Correlation)', 'Random Forest', 'GNN (Base)', 'GNN (+PayType)']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for ax, (model_name, importances), title, color in zip(
        axes.flatten(), all_importances.items(), titles, colors
    ):
        if not importances:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            continue
        
        # Take top 10
        top_n = importances[:10]
        features = [f[0][:20] for f in top_n]  # Truncate long names
        values = [f[1] for f in top_n]
        
        bars = ax.barh(range(len(features)), values, color=color, alpha=0.8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title, fontweight='bold')
        
        # Extend x-axis limit to prevent text from being cut off
        max_val = max(values) if values else 1
        ax.set_xlim(0, max_val * 1.25)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + (max_val * 0.02), bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.close()


def main():
    """Main execution."""
    set_seed(RANDOM_STATE)
    
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS - ALL MODELS")
    print("=" * 60)
    
    all_importances = {}
    
    # K-means
    try:
        kmeans_imp = get_kmeans_feature_importance()
        all_importances['kmeans'] = kmeans_imp
        print_feature_importance("K-means (Top 10)", kmeans_imp)
    except Exception as e:
        print(f"K-means error: {e}")
        all_importances['kmeans'] = []
    
    # Random Forest
    try:
        rf_imp = get_rf_feature_importance()
        all_importances['rf'] = rf_imp
        print_feature_importance("Random Forest (Top 10)", rf_imp)
    except Exception as e:
        print(f"Random Forest error: {e}")
        all_importances['rf'] = []
    
    # GNN Base
    try:
        gnn_base_imp = get_gnn_feature_importance('base')
        all_importances['gnn_base'] = gnn_base_imp
        print_feature_importance("GNN Base (All 4 features)", gnn_base_imp, top_n=4)
    except Exception as e:
        print(f"GNN Base error: {e}")
        all_importances['gnn_base'] = []
    
    # GNN +PayType
    try:
        gnn_paytype_imp = get_gnn_feature_importance('paytype')
        all_importances['gnn_paytype'] = gnn_paytype_imp
        print_feature_importance("GNN +PayType (Top 10)", gnn_paytype_imp)
    except Exception as e:
        print(f"GNN +PayType error: {e}")
        all_importances['gnn_paytype'] = []
    
    # Create comparison plot
    plot_path = os.path.join(RESULTS_DIR, 'feature_importance_comparison.png')
    plot_feature_importance_comparison(all_importances, plot_path)
    
    # Save results
    np.savez(
        os.path.join(RESULTS_DIR, 'feature_importance_all.npz'),
        kmeans=all_importances.get('kmeans', []),
        rf=all_importances.get('rf', []),
        gnn_base=all_importances.get('gnn_base', []),
        gnn_paytype=all_importances.get('gnn_paytype', [])
    )
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to {RESULTS_DIR}/feature_importance_all.npz")
    print(f"Plot saved to {plot_path}")
    
    return all_importances


if __name__ == '__main__':
    results = main()
