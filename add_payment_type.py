"""
Add Payment Type to GNN Edge Features
Quick update to existing gnn_graph.npz without full re-processing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

DATA_PATH = 'SAML-D.csv'
OUTPUT_DIR = 'processed_data'
CHUNK_SIZE = 500000

def main():
    print("=" * 60)
    print("ADDING PAYMENT TYPE TO GNN EDGE FEATURES")
    print("=" * 60)
    
    # Load existing GNN data
    print("\nLoading existing GNN data...")
    gnn_data = np.load(os.path.join(OUTPUT_DIR, 'gnn_graph.npz'), allow_pickle=True)
    
    existing_edge_features = gnn_data['edge_features']
    node_features = gnn_data['node_features']
    edge_index = gnn_data['edge_index']
    edge_labels = gnn_data['edge_labels']
    account_to_idx = gnn_data['account_to_idx']
    node_feature_names = list(gnn_data['node_feature_names'])
    old_edge_feature_names = list(gnn_data['edge_feature_names'])
    
    print(f"Existing edge features shape: {existing_edge_features.shape}")
    print(f"Existing edge feature names: {old_edge_feature_names}")
    
    # First pass: gather unique payment types
    print("\nGathering unique payment types...")
    all_payment_types = set()
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, usecols=['Payment_type']):
        all_payment_types.update(chunk['Payment_type'].unique())
    
    payment_types_list = sorted(list(all_payment_types))
    print(f"Payment types found: {payment_types_list}")
    
    # Create one-hot encoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(np.array(payment_types_list).reshape(-1, 1))
    
    # Second pass: extract payment types in order
    print("\nExtracting payment types for all edges...")
    all_payment_type_ohe = []
    chunk_idx = 0
    
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, usecols=['Payment_type']):
        chunk_idx += 1
        payment_type_ohe = ohe.transform(chunk['Payment_type'].values.reshape(-1, 1))
        all_payment_type_ohe.append(payment_type_ohe)
        print(f"  Processed chunk {chunk_idx}...", end='\r')
    
    print(f"\nCombining payment type features...")
    payment_type_features = np.vstack(all_payment_type_ohe)
    print(f"Payment type features shape: {payment_type_features.shape}")
    
    # Combine with existing edge features (which are already scaled)
    # Note: payment type is one-hot encoded (0/1), no need to scale
    print("\nCombining edge features...")
    new_edge_features = np.hstack([existing_edge_features, payment_type_features])
    print(f"New edge features shape: {new_edge_features.shape}")
    
    # New feature names
    payment_type_feature_names = [f'payment_type_{pt}' for pt in ohe.categories_[0]]
    new_edge_feature_names = old_edge_feature_names + payment_type_feature_names
    print(f"New edge feature names: {new_edge_feature_names}")
    
    # Save updated data
    print("\nSaving updated GNN data...")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'gnn_graph.npz'),
        node_features=node_features,
        edge_index=edge_index,
        edge_features=new_edge_features,
        edge_labels=edge_labels,
        account_to_idx=account_to_idx,
        node_feature_names=node_feature_names,
        edge_feature_names=new_edge_feature_names
    )
    
    print(f"\n✓ Updated GNN data saved!")
    print(f"  Edge features: {existing_edge_features.shape[1]} → {new_edge_features.shape[1]}")
    print(f"  New features: {payment_type_feature_names}")

if __name__ == '__main__':
    main()
