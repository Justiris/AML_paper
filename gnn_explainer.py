"""
GNN Explainer for Anti-Money Laundering Detection
Uses PyTorch Geometric's GNNExplainer to explain edge predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import os
import warnings

warnings.filterwarnings('ignore')

# Check for PyTorch Geometric
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig
    HAS_PYG_EXPLAIN = True
except ImportError:
    HAS_PYG_EXPLAIN = False
    print("PyTorch Geometric Explain module not available.")

# Configuration
DATA_DIR = 'processed_data'
RESULTS_DIR = 'results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (must match training)
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# Explainer settings
NUM_EDGES_TO_EXPLAIN = 100  # Number of fraud edges to explain
EXPLAINER_EPOCHS = 100


class EdgeClassifierMLP(nn.Module):
    """MLP for edge classification given node embeddings and edge features."""
    
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
    """GraphSAGE encoder for node embeddings."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, out_dim))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GNNModel(nn.Module):
    """Full GNN model for edge classification."""
    
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


class ExplainableEdgeModel(nn.Module):
    """
    Wrapper model for edge classification that works with PyG's Explainer.
    
    The Explainer expects a model that takes (x, edge_index, ...) and returns
    predictions. For edge classification, we need to specify which edge to explain.
    """
    
    def __init__(self, base_model, edge_features, target_edge_idx):
        super().__init__()
        self.base_model = base_model
        self.edge_features = edge_features
        self.target_edge_idx = target_edge_idx
    
    def forward(self, x, edge_index, edge_mask=None):
        """
        Forward pass for explainer.
        Returns the prediction for the target edge.
        """
        # Get node embeddings
        node_emb = self.base_model.encoder(x, edge_index)
        
        # Get source and target node embeddings for the target edge
        src_idx = edge_index[0, self.target_edge_idx]
        dst_idx = edge_index[1, self.target_edge_idx]
        
        src_emb = node_emb[src_idx].unsqueeze(0)
        dst_emb = node_emb[dst_idx].unsqueeze(0)
        edge_feat = self.edge_features[self.target_edge_idx].unsqueeze(0)
        
        # Classify the edge
        logit = self.base_model.edge_classifier(src_emb, dst_emb, edge_feat)
        
        # Return as 2D tensor for classification (fraud, non-fraud scores)
        prob = torch.sigmoid(logit)
        return torch.stack([1 - prob, prob], dim=-1)


def load_data_and_model():
    """Load preprocessed graph data and trained model."""
    print("Loading data and model...")
    
    # Load graph data
    gnn_data = np.load(os.path.join(DATA_DIR, 'gnn_graph.npz'), allow_pickle=True)
    
    node_features = torch.FloatTensor(gnn_data['node_features']).to(DEVICE)
    edge_index = torch.LongTensor(gnn_data['edge_index']).to(DEVICE)
    edge_features = torch.FloatTensor(gnn_data['edge_features']).to(DEVICE)
    edge_labels = torch.FloatTensor(gnn_data['edge_labels']).to(DEVICE)
    
    print(f"  Nodes: {node_features.shape[0]:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Node features: {node_features.shape[1]}")
    print(f"  Edge features: {edge_features.shape[1]}")
    
    # Load trained model
    model = GNNModel(
        node_in_dim=node_features.shape[1],
        edge_in_dim=edge_features.shape[1],
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    
    model_path = os.path.join(RESULTS_DIR, 'gnn_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    print(f"  Model loaded from {model_path}")
    
    return node_features, edge_index, edge_features, edge_labels, model


def get_edge_feature_names():
    """Return the names of edge features."""
    return [
        'amount',
        'hour',
        'is_cross_border',
        'is_currency_exchange',
        'payment_ACH',
        'payment_Cash_Deposit',
        'payment_Cash_Withdrawal',
        'payment_Cheque',
        'payment_Credit_card',
        'payment_Cross_border',
        'payment_Debit_card'
    ]


def get_node_feature_names():
    """Return the names of node features."""
    return [
        'out_count',
        'out_amount',
        'in_count',
        'in_amount',
        'avg_out',
        'avg_in'
    ]


def explain_edge_gradient(model, node_features, edge_index, edge_features, 
                           edge_idx, edge_labels):
    """
    Explain a single edge prediction using gradient-based attribution.
    
    Returns:
        dict: Explanation with feature importance scores
    """
    model.eval()
    
    # Clone features for gradient computation
    node_feat = node_features.clone().detach().requires_grad_(True)
    edge_feat = edge_features.clone().detach()
    edge_feat_single = edge_feat[edge_idx].clone().detach().requires_grad_(True)
    
    # Forward pass for the target edge
    node_emb = model.encoder(node_feat, edge_index)
    
    src_idx = edge_index[0, edge_idx]
    dst_idx = edge_index[1, edge_idx]
    
    src_emb = node_emb[src_idx].unsqueeze(0)
    dst_emb = node_emb[dst_idx].unsqueeze(0)
    
    logit = model.edge_classifier(src_emb, dst_emb, edge_feat_single.unsqueeze(0))
    prob = torch.sigmoid(logit)
    
    # Compute gradients w.r.t. edge features
    prob.backward()
    
    # Edge feature importance (gradient * input)
    edge_importance = (edge_feat_single.grad * edge_feat_single).abs().detach().cpu().numpy()
    
    predicted_fraud = prob.item() > 0.5
    actual_fraud = edge_labels[edge_idx].item() > 0.5
    
    return {
        'edge_idx': edge_idx,
        'predicted_prob': prob.item(),
        'predicted_fraud': predicted_fraud,
        'actual_fraud': actual_fraud,
        'edge_importance': edge_importance,
        'src_node': src_idx.item(),
        'dst_node': dst_idx.item()
    }


def explain_edges_batch(model, node_features, edge_index, edge_features, 
                         edge_labels, edge_indices):
    """
    Explain multiple edges using gradient-based attribution.
    
    Args:
        edge_indices: List of edge indices to explain
        
    Returns:
        list: List of explanation dictionaries
    """
    explanations = []
    
    print(f"\nExplaining {len(edge_indices):,} edges...")
    
    for i, edge_idx in enumerate(edge_indices):
        if (i + 1) % 500 == 0 or (i + 1) == len(edge_indices):
            print(f"  Progress: {i + 1:,}/{len(edge_indices):,}")
        
        explanation = explain_edge_gradient(
            model, node_features, edge_index, edge_features, 
            edge_idx, edge_labels
        )
        explanations.append(explanation)
    
    return explanations


def aggregate_feature_importance(explanations, feature_names):
    """
    Aggregate feature importance across all explanations.
    
    Returns:
        dict: Feature name -> average importance score
    """
    if not explanations:
        return {}
    
    all_importance = np.array([exp['edge_importance'] for exp in explanations])
    
    # Normalize each explanation to sum to 1
    row_sums = all_importance.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = all_importance / row_sums
    
    # Average across all explanations
    mean_importance = normalized.mean(axis=0)
    std_importance = normalized.std(axis=0)
    
    # Create ranked list
    feature_importance = {}
    for i, name in enumerate(feature_names):
        feature_importance[name] = {
            'mean': mean_importance[i],
            'std': std_importance[i],
            'rank': 0  # Will be set below
        }
    
    # Assign ranks
    sorted_features = sorted(feature_importance.keys(), 
                            key=lambda x: feature_importance[x]['mean'], 
                            reverse=True)
    for rank, name in enumerate(sorted_features, 1):
        feature_importance[name]['rank'] = rank
    
    return feature_importance


def visualize_feature_importance(feature_importance, save_path=None):
    """
    Create a bar chart of feature importance.
    """
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), 
                            key=lambda x: x[1]['mean'], 
                            reverse=True)
    
    names = [f[0] for f in sorted_features]
    means = [f[1]['mean'] for f in sorted_features]
    stds = [f[1]['std'] for f in sorted_features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color bars by category
    colors = []
    for name in names:
        if name.startswith('payment_'):
            colors.append('#2ecc71')  # Green for payment type
        elif name in ['amount', 'hour']:
            colors.append('#3498db')  # Blue for transaction features
        else:
            colors.append('#e74c3c')  # Red for cross-border features
    
    bars = ax.barh(range(len(names)), means, xerr=stds, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance (mean ± std)')
    ax.set_title('Edge Feature Importance for Fraud Detection\n(Gradient-based Attribution)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, label='Transaction Features'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Cross-border Features'),
        Patch(facecolor='#2ecc71', alpha=0.8, label='Payment Type')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Feature importance plot saved to {save_path}")
    
    plt.close()
    
    return fig


def visualize_explanation_subgraph(explanation, edge_index, node_features, 
                                    max_neighbors=10, save_path=None):
    """
    Visualize a transaction and its neighborhood.
    """
    src_node = explanation['src_node']
    dst_node = explanation['dst_node']
    
    # Find neighbors of source and destination
    edge_index_np = edge_index.cpu().numpy()
    
    # Get 1-hop neighbors
    src_neighbors = set(edge_index_np[1, edge_index_np[0] == src_node])
    dst_neighbors = set(edge_index_np[1, edge_index_np[0] == dst_node])
    
    # Also get incoming edges
    src_incoming = set(edge_index_np[0, edge_index_np[1] == src_node])
    dst_incoming = set(edge_index_np[0, edge_index_np[1] == dst_node])
    
    all_neighbors = src_neighbors | dst_neighbors | src_incoming | dst_incoming
    all_neighbors.discard(src_node)
    all_neighbors.discard(dst_node)
    
    # Limit neighbors
    if len(all_neighbors) > max_neighbors:
        all_neighbors = set(list(all_neighbors)[:max_neighbors])
    
    # Build subgraph
    G = nx.DiGraph()
    
    # Add main nodes
    G.add_node(src_node, label='Sender', color='#e74c3c')
    G.add_node(dst_node, label='Receiver', color='#3498db')
    
    # Add neighbors
    for n in all_neighbors:
        G.add_node(n, label='Neighbor', color='#95a5a6')
    
    # Add edges
    nodes_in_graph = {src_node, dst_node} | all_neighbors
    for i in range(edge_index_np.shape[1]):
        s, d = edge_index_np[0, i], edge_index_np[1, i]
        if s in nodes_in_graph and d in nodes_in_graph:
            is_main_edge = (s == src_node and d == dst_node)
            G.add_edge(s, d, weight=3 if is_main_edge else 1, 
                      color='#e74c3c' if is_main_edge else '#bdc3c7')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    node_colors = [G.nodes[n].get('color', '#95a5a6') for n in G.nodes()]
    node_sizes = [800 if n in [src_node, dst_node] else 400 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                           alpha=0.9, ax=ax)
    
    # Draw edges
    edge_colors = [G.edges[e].get('color', '#bdc3c7') for e in G.edges()]
    edge_widths = [G.edges[e].get('weight', 1) for e in G.edges()]
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.6, arrows=True, arrowsize=15, ax=ax)
    
    # Labels for main nodes only
    labels = {src_node: f'Sender\n({src_node})', dst_node: f'Receiver\n({dst_node})'}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)
    
    # Title
    fraud_str = "FRAUD" if explanation['actual_fraud'] else "NORMAL"
    pred_str = f"Predicted: {explanation['predicted_prob']:.2%}"
    ax.set_title(f"Transaction Subgraph - {fraud_str}\n{pred_str}", fontsize=12)
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def save_explanations(explanations, feature_importance, save_dir=RESULTS_DIR):
    """Save explanation results to file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for saving
    edge_indices = np.array([e['edge_idx'] for e in explanations])
    predicted_probs = np.array([e['predicted_prob'] for e in explanations])
    predicted_fraud = np.array([e['predicted_fraud'] for e in explanations])
    actual_fraud = np.array([e['actual_fraud'] for e in explanations])
    edge_importance = np.array([e['edge_importance'] for e in explanations])
    
    # Feature importance summary
    feature_names = get_edge_feature_names()
    importance_means = np.array([feature_importance[n]['mean'] for n in feature_names])
    importance_stds = np.array([feature_importance[n]['std'] for n in feature_names])
    
    save_path = os.path.join(save_dir, 'gnn_explanations.npz')
    np.savez_compressed(
        save_path,
        edge_indices=edge_indices,
        predicted_probs=predicted_probs,
        predicted_fraud=predicted_fraud,
        actual_fraud=actual_fraud,
        edge_importance=edge_importance,
        feature_names=np.array(feature_names),
        importance_means=importance_means,
        importance_stds=importance_stds
    )
    
    print(f"  Explanations saved to {save_path}")
    
    return save_path


def print_feature_importance_summary(feature_importance):
    """Print a summary of feature importance."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE RANKING")
    print("=" * 60)
    
    sorted_features = sorted(feature_importance.items(), 
                            key=lambda x: x[1]['mean'], 
                            reverse=True)
    
    print(f"\n{'Rank':<6}{'Feature':<25}{'Importance':<15}{'Std':<10}")
    print("-" * 56)
    
    for name, info in sorted_features:
        print(f"{info['rank']:<6}{name:<25}{info['mean']:.4f}{'':<5}{info['std']:.4f}")
    
    print("\n" + "=" * 60)


def main():
    """Main execution."""
    print("=" * 60)
    print("GNN EXPLAINER FOR AML DETECTION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    if not HAS_PYG_EXPLAIN:
        print("ERROR: PyTorch Geometric Explain module not available.")
        return
    
    # Load data and model
    node_features, edge_index, edge_features, edge_labels, model = load_data_and_model()
    
    # Find fraud edges to explain
    fraud_mask = edge_labels > 0.5
    fraud_indices = torch.where(fraud_mask)[0].cpu().numpy()
    
    print(f"\nTotal fraud edges: {len(fraud_indices):,}")
    print(f"Explaining ALL fraud edges...")
    
    # Generate explanations for ALL fraud edges
    explanations = explain_edges_batch(
        model, node_features, edge_index, edge_features, 
        edge_labels, fraud_indices
    )
    
    # Aggregate feature importance
    feature_names = get_edge_feature_names()
    feature_importance = aggregate_feature_importance(explanations, feature_names)
    
    # Print summary
    print_feature_importance_summary(feature_importance)
    
    # Visualize feature importance
    vis_path = os.path.join(RESULTS_DIR, 'gnn_feature_importance.png')
    visualize_feature_importance(feature_importance, save_path=vis_path)
    
    # Visualize a few example subgraphs
    print("\nGenerating example subgraph visualizations...")
    for i, exp in enumerate(explanations[:3]):
        subgraph_path = os.path.join(RESULTS_DIR, f'gnn_subgraph_example_{i+1}.png')
        visualize_explanation_subgraph(exp, edge_index, node_features, save_path=subgraph_path)
        print(f"  Saved subgraph visualization: {subgraph_path}")
    
    # Save all explanations
    save_explanations(explanations, feature_importance)
    
    # Summary statistics
    correct_preds = sum(1 for e in explanations if e['predicted_fraud'] == e['actual_fraud'])
    accuracy = correct_preds / len(explanations)
    
    print("\n" + "=" * 60)
    print("EXPLANATION SUMMARY")
    print("=" * 60)
    print(f"  Edges explained: {len(explanations)}")
    print(f"  Prediction accuracy on explained edges: {accuracy:.1%}")
    print(f"  Feature importance plot: {vis_path}")
    print("=" * 60)
    
    return explanations, feature_importance


if __name__ == '__main__':
    explanations, feature_importance = main()
