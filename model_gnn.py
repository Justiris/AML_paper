"""
Graph Neural Network for Anti-Money Laundering Detection
Uses edge classification on transaction graph with GraphSAGE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, 
    average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Check for PyTorch Geometric
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not installed. Using custom implementation.")

# Configuration
DATA_DIR = 'processed_data'
RESULTS_DIR = 'results'
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 65536  # For edge batching
PATIENCE = 10


class EdgeClassifierMLP(nn.Module):
    """MLP for edge classification given node embeddings and edge features."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # Input: 2 node embeddings + edge features
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


class SimpleGNN(nn.Module):
    """Simple GNN without PyTorch Geometric dependency."""
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x, edge_index):
        # Simple message passing (mean aggregation)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if HAS_PYG:
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
else:
    GraphSAGEEncoder = SimpleGNN


class GNNModel(nn.Module):
    """Full GNN model for edge classification."""
    
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.encoder = GraphSAGEEncoder(node_in_dim, hidden_dim, hidden_dim, NUM_LAYERS)
        self.edge_classifier = EdgeClassifierMLP(hidden_dim, edge_in_dim, hidden_dim)
    
    def forward(self, node_features, edge_index, edge_features, batch_edge_idx=None):
        # Get node embeddings using full edge_index for message passing
        node_emb = self.encoder(node_features, edge_index)
        
        # Get source and destination embeddings for edges we're classifying
        if batch_edge_idx is not None:
            src_emb = node_emb[edge_index[0, batch_edge_idx]]
            dst_emb = node_emb[edge_index[1, batch_edge_idx]]
        else:
            src_emb = node_emb[edge_index[0]]
            dst_emb = node_emb[edge_index[1]]
        
        # Classify edges
        logits = self.edge_classifier(src_emb, dst_emb, edge_features)
        return logits


def load_data():
    """Load preprocessed graph data."""
    print("Loading GNN data...")
    
    gnn_data = np.load(os.path.join(DATA_DIR, 'gnn_graph.npz'), allow_pickle=True)
    
    node_features = torch.FloatTensor(gnn_data['node_features'])
    edge_index = torch.LongTensor(gnn_data['edge_index'])
    edge_features = torch.FloatTensor(gnn_data['edge_features'])
    edge_labels = torch.FloatTensor(gnn_data['edge_labels'])
    
    print(f"Nodes: {node_features.shape[0]:,}")
    print(f"Node features: {node_features.shape[1]}")
    print(f"Edges: {edge_index.shape[1]:,}")
    print(f"Edge features: {edge_features.shape[1]}")
    print(f"Fraud edges: {edge_labels.sum().int().item():,} ({edge_labels.mean()*100:.4f}%)")
    
    return node_features, edge_index, edge_features, edge_labels


def split_edges(edge_labels, test_size=0.2, val_size=0.1):
    """Split edges into train/val/test sets."""
    n_edges = len(edge_labels)
    indices = np.arange(n_edges)
    
    # Stratified split
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=RANDOM_STATE,
        stratify=edge_labels.numpy()
    )
    
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size/(1-test_size), random_state=RANDOM_STATE,
        stratify=edge_labels[train_idx].numpy()
    )
    
    print(f"\nEdge splits:")
    print(f"  Train: {len(train_idx):,} ({edge_labels[train_idx].mean()*100:.4f}% fraud)")
    print(f"  Val:   {len(val_idx):,} ({edge_labels[val_idx].mean()*100:.4f}% fraud)")
    print(f"  Test:  {len(test_idx):,} ({edge_labels[test_idx].mean()*100:.4f}% fraud)")
    
    return train_idx, val_idx, test_idx


def train_epoch(model, optimizer, node_features, edge_index, edge_features, 
                edge_labels, train_idx, pos_weight):
    """Train for one epoch."""
    model.train()
    
    # Shuffle training indices
    perm = torch.randperm(len(train_idx))
    train_idx_shuffled = train_idx[perm]
    
    total_loss = 0
    n_batches = 0
    
    # Mini-batch training on edges
    for i in range(0, len(train_idx_shuffled), BATCH_SIZE):
        batch_idx = train_idx_shuffled[i:i+BATCH_SIZE]
        
        optimizer.zero_grad()
        
        logits = model(node_features, edge_index, edge_features[batch_idx], batch_idx)
        
        loss = F.binary_cross_entropy_with_logits(
            logits, edge_labels[batch_idx], pos_weight=pos_weight
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, node_features, edge_index, edge_features, edge_labels, eval_idx):
    """Evaluate model on given edge indices."""
    model.eval()
    
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, len(eval_idx), BATCH_SIZE):
            batch_idx = eval_idx[i:i+BATCH_SIZE]
            logits = model(node_features, edge_index, edge_features[batch_idx], batch_idx)
            all_logits.append(logits)
    
    logits = torch.cat(all_logits)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    y_true = edge_labels[eval_idx].cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        roc_auc = 0
        pr_auc = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'y_true': y_true
    }


def train_model(node_features, edge_index, edge_features, edge_labels):
    """Train GNN model."""
    print("\n" + "=" * 60)
    print("TRAINING GNN MODEL")
    print("=" * 60)
    
    # Move to device
    node_features = node_features.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_features = edge_features.to(DEVICE)
    edge_labels = edge_labels.to(DEVICE)
    
    # Split edges
    train_idx, val_idx, test_idx = split_edges(edge_labels.cpu())
    train_idx = torch.LongTensor(train_idx).to(DEVICE)
    val_idx = torch.LongTensor(val_idx).to(DEVICE)
    test_idx = torch.LongTensor(test_idx).to(DEVICE)
    
    # Calculate positive weight for class imbalance
    n_pos = edge_labels.sum()
    n_neg = len(edge_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(DEVICE)
    print(f"\nPositive weight: {pos_weight.item():.2f}")
    
    # Initialize model
    model = GNNModel(
        node_in_dim=node_features.shape[1],
        edge_in_dim=edge_features.shape[1],
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    
    print(f"\nModel architecture:")
    print(f"  Node encoder: GraphSAGE ({node_features.shape[1]} -> {HIDDEN_DIM})")
    print(f"  Edge classifier: MLP ({HIDDEN_DIM}*2 + {edge_features.shape[1]} -> 1)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print(f"\nTraining for up to {EPOCHS} epochs...")
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, optimizer, node_features, edge_index, 
            edge_features, edge_labels, train_idx, pos_weight
        )
        
        val_metrics = evaluate(model, node_features, edge_index, edge_features, edge_labels, val_idx)
        
        scheduler.step(val_metrics['pr_auc'])
        
        if val_metrics['pr_auc'] > best_val_auc:
            best_val_auc = val_metrics['pr_auc']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Val PR-AUC={val_metrics['pr_auc']:.4f}, "
                  f"Val F1={val_metrics['f1']:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, test_idx, node_features, edge_index, edge_features, edge_labels


def evaluate_final(model, node_features, edge_index, edge_features, edge_labels, test_idx):
    """Final evaluation on test set."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = evaluate(model, node_features, edge_index, edge_features, edge_labels, test_idx)
    
    print("\nTest Set Metrics:")
    print("-" * 50)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"\n  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:>10,}  FP: {cm[0,1]:>10,}")
    print(f"  FN: {cm[1,0]:>10,}  TP: {cm[1,1]:>10,}")
    
    metrics['confusion_matrix'] = cm
    
    return metrics


def save_results(model, metrics):
    """Save model and results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, 'gnn_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save results
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'gnn_results.npz'),
        y_prob=metrics['y_prob'],
        y_pred=metrics['y_pred'],
        y_true=metrics['y_true'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        roc_auc=metrics['roc_auc'],
        pr_auc=metrics['pr_auc'],
        confusion_matrix=metrics['confusion_matrix']
    )
    
    print(f"Results saved to {RESULTS_DIR}/gnn_results.npz")


def main():
    """Main execution."""
    print("=" * 60)
    print("GNN FOR AML DETECTION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"PyTorch Geometric: {'Available' if HAS_PYG else 'Not available (using simple GNN)'}")
    
    # Load data
    node_features, edge_index, edge_features, edge_labels = load_data()
    
    # Train model
    model, test_idx, node_features, edge_index, edge_features, edge_labels = train_model(
        node_features, edge_index, edge_features, edge_labels
    )
    
    # Evaluate
    metrics = evaluate_final(model, node_features, edge_index, edge_features, edge_labels, test_idx)
    
    # Save results
    save_results(model, metrics)
    
    print("\n" + "=" * 60)
    print("GNN MODEL COMPLETE")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    metrics = main()
