"""
Data Preparation Pipeline for Anti-Money Laundering Models
Prepares SAML-D dataset for K-means, Random Forest, and GNN models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
import os
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'SAML-D.csv'
OUTPUT_DIR = 'processed_data'
CHUNK_SIZE = 500000  # Process in chunks of 500K rows
RANDOM_STATE = 42
TEST_SIZE = 0.2

def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}/")

def get_data_info():
    """Get basic info about the dataset without loading it fully."""
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    # Count total rows
    total_rows = sum(1 for _ in open(DATA_PATH)) - 1  # -1 for header
    print(f"Total rows: {total_rows:,}")
    
    # Get column names
    sample = pd.read_csv(DATA_PATH, nrows=5)
    print(f"Columns: {list(sample.columns)}")
    print(f"Number of columns: {len(sample.columns)}")
    
    return total_rows

def extract_features_from_chunk(df):
    """Extract features from a chunk of data."""
    
    # Time-based features
    df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = pd.to_datetime(df['Date']).dt.month
    
    # Amount-based features
    df['amount_log'] = np.log1p(df['Amount'])
    
    # Transaction pattern features
    df['is_cross_border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
    df['is_currency_exchange'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
    
    return df

def process_data_in_chunks():
    """Process the full dataset in chunks."""
    print("\n" + "=" * 60)
    print("PROCESSING DATA IN CHUNKS")
    print("=" * 60)
    
    # First pass: gather unique values for encoders
    print("\nPass 1: Gathering unique categorical values...")
    
    all_payment_currencies = set()
    all_received_currencies = set()
    all_sender_locations = set()
    all_receiver_locations = set()
    all_payment_types = set()
    all_laundering_types = set()
    
    chunk_count = 0
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
        all_payment_currencies.update(chunk['Payment_currency'].unique())
        all_received_currencies.update(chunk['Received_currency'].unique())
        all_sender_locations.update(chunk['Sender_bank_location'].unique())
        all_receiver_locations.update(chunk['Receiver_bank_location'].unique())
        all_payment_types.update(chunk['Payment_type'].unique())
        all_laundering_types.update(chunk['Laundering_type'].unique())
        chunk_count += 1
        print(f"  Processed chunk {chunk_count}...", end='\r')
    
    print(f"\nTotal chunks: {chunk_count}")
    print(f"Unique payment currencies: {len(all_payment_currencies)}")
    print(f"Unique received currencies: {len(all_received_currencies)}")
    print(f"Unique sender locations: {len(all_sender_locations)}")
    print(f"Unique receiver locations: {len(all_receiver_locations)}")
    print(f"Unique payment types: {len(all_payment_types)}")
    print(f"Unique laundering types: {len(all_laundering_types)}")
    
    # Create label encoders
    le_payment_curr = LabelEncoder().fit(list(all_payment_currencies))
    le_received_curr = LabelEncoder().fit(list(all_received_currencies))
    le_sender_loc = LabelEncoder().fit(list(all_sender_locations))
    le_receiver_loc = LabelEncoder().fit(list(all_receiver_locations))
    le_payment_type = LabelEncoder().fit(list(all_payment_types))
    le_laundering_type = LabelEncoder().fit(list(all_laundering_types))
    
    # One-hot encoder for payment types
    ohe_payment_type = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_payment_type.fit(np.array(list(all_payment_types)).reshape(-1, 1))
    
    # Second pass: process and transform data
    print("\nPass 2: Processing and transforming data...")
    
    all_features = []
    all_labels = []
    all_accounts = {'sender': set(), 'receiver': set()}
    all_graph_data = []
    
    chunk_idx = 0
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
        chunk_idx += 1
        print(f"  Processing chunk {chunk_idx}/{chunk_count}...")
        
        # Extract features
        chunk = extract_features_from_chunk(chunk)
        
        # Collect unique accounts for GNN
        all_accounts['sender'].update(chunk['Sender_account'].unique())
        all_accounts['receiver'].update(chunk['Receiver_account'].unique())
        
        # Store graph edge data (for GNN)
        graph_chunk = chunk[['Sender_account', 'Receiver_account', 'Amount', 
                             'Is_laundering', 'Payment_type', 'hour', 
                             'is_cross_border', 'is_currency_exchange']].copy()
        all_graph_data.append(graph_chunk)
        
        # Encode categorical variables
        chunk['payment_currency_enc'] = le_payment_curr.transform(chunk['Payment_currency'])
        chunk['received_currency_enc'] = le_received_curr.transform(chunk['Received_currency'])
        chunk['sender_location_enc'] = le_sender_loc.transform(chunk['Sender_bank_location'])
        chunk['receiver_location_enc'] = le_receiver_loc.transform(chunk['Receiver_bank_location'])
        chunk['laundering_type_enc'] = le_laundering_type.transform(chunk['Laundering_type'])
        
        # One-hot encode payment type
        payment_type_ohe = ohe_payment_type.transform(chunk['Payment_type'].values.reshape(-1, 1))
        payment_type_cols = [f'payment_type_{pt}' for pt in ohe_payment_type.categories_[0]]
        
        # Select numerical features for RF/K-means
        numerical_features = [
            'Amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'month',
            'is_cross_border', 'is_currency_exchange',
            'payment_currency_enc', 'received_currency_enc',
            'sender_location_enc', 'receiver_location_enc'
        ]
        
        features = chunk[numerical_features].values
        features = np.hstack([features, payment_type_ohe])
        
        all_features.append(features)
        all_labels.append(chunk['Is_laundering'].values)
        
        # Clean up
        del chunk
        gc.collect()
    
    # Combine all chunks
    print("\nCombining all chunks...")
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"Total samples: {X.shape[0]:,}")
    print(f"Total features: {X.shape[1]}")
    print(f"Class distribution: 0={np.sum(y==0):,}, 1={np.sum(y==1):,}")
    print(f"Fraud rate: {np.mean(y)*100:.4f}%")
    
    # Create feature names
    feature_names = numerical_features + payment_type_cols
    
    return X, y, feature_names, all_accounts, all_graph_data, {
        'le_payment_curr': le_payment_curr,
        'le_received_curr': le_received_curr,
        'le_sender_loc': le_sender_loc,
        'le_receiver_loc': le_receiver_loc,
        'le_payment_type': le_payment_type,
        'le_laundering_type': le_laundering_type,
        'ohe_payment_type': ohe_payment_type
    }

def prepare_kmeans_data(X, feature_names):
    """Prepare normalized data for K-means clustering."""
    print("\n" + "=" * 60)
    print("PREPARING K-MEANS DATA")
    print("=" * 60)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'kmeans_features.npz'),
        X=X_scaled,
        feature_names=feature_names
    )
    
    print(f"K-means data shape: {X_scaled.shape}")
    print(f"Saved to: {OUTPUT_DIR}/kmeans_features.npz")
    
    return scaler

def prepare_rf_data(X, y, feature_names):
    """Prepare data for Random Forest with SMOTE oversampling."""
    print("\n" + "=" * 60)
    print("PREPARING RANDOM FOREST DATA")
    print("=" * 60)
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train size: {X_train.shape[0]:,}")
    print(f"Test size: {X_test.shape[0]:,}")
    print(f"Train fraud rate: {np.mean(y_train)*100:.4f}%")
    print(f"Test fraud rate: {np.mean(y_test)*100:.4f}%")
    
    # Apply SMOTE to training data only
    print("\nApplying SMOTE oversampling to training data...")
    smote = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Train size: {X_train_resampled.shape[0]:,}")
    print(f"After SMOTE - Class distribution: 0={np.sum(y_train_resampled==0):,}, 1={np.sum(y_train_resampled==1):,}")
    
    # Calculate class weights for additional balancing
    n_samples = len(y_train)
    n_classes = 2
    class_weights = {
        0: n_samples / (n_classes * np.sum(y_train == 0)),
        1: n_samples / (n_classes * np.sum(y_train == 1))
    }
    print(f"Class weights: {class_weights}")
    
    # Save
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'rf_train.npz'),
        X=X_train_resampled,
        y=y_train_resampled,
        feature_names=feature_names,
        class_weights=np.array([class_weights[0], class_weights[1]])
    )
    
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'rf_test.npz'),
        X=X_test,
        y=y_test,
        feature_names=feature_names
    )
    
    print(f"Saved to: {OUTPUT_DIR}/rf_train.npz, {OUTPUT_DIR}/rf_test.npz")
    
    return class_weights

def prepare_gnn_data(all_accounts, all_graph_data):
    """Prepare graph data for GNN."""
    print("\n" + "=" * 60)
    print("PREPARING GNN DATA")
    print("=" * 60)
    
    # Combine all graph data
    print("Combining graph edge data...")
    graph_df = pd.concat(all_graph_data, ignore_index=True)
    
    # Create unified account list (node IDs)
    all_unique_accounts = list(all_accounts['sender'].union(all_accounts['receiver']))
    account_to_idx = {acc: idx for idx, acc in enumerate(all_unique_accounts)}
    
    print(f"Total unique accounts (nodes): {len(all_unique_accounts):,}")
    print(f"Total transactions (edges): {len(graph_df):,}")
    
    # Create edge index
    print("Creating edge index...")
    src_nodes = graph_df['Sender_account'].map(account_to_idx).values
    dst_nodes = graph_df['Receiver_account'].map(account_to_idx).values
    edge_index = np.vstack([src_nodes, dst_nodes])
    
    # Edge features
    print("Creating edge features...")
    edge_features = graph_df[['Amount', 'hour', 'is_cross_border', 'is_currency_exchange']].values
    edge_labels = graph_df['Is_laundering'].values
    
    # Normalize edge features
    scaler = StandardScaler()
    edge_features_scaled = scaler.fit_transform(edge_features)
    
    # Calculate node features (aggregated from transactions)
    print("Calculating node features...")
    node_features = np.zeros((len(all_unique_accounts), 6))  # 6 features per node
    
    # Count outgoing/incoming transactions and sum amounts
    for i, (src, dst, amt) in enumerate(zip(src_nodes, dst_nodes, graph_df['Amount'].values)):
        node_features[src, 0] += 1  # outgoing count
        node_features[src, 1] += amt  # outgoing amount
        node_features[dst, 2] += 1  # incoming count
        node_features[dst, 3] += amt  # incoming amount
        if i % 1000000 == 0:
            print(f"  Processed {i:,} edges...", end='\r')
    
    # Calculate averages
    node_features[:, 4] = np.where(node_features[:, 0] > 0, 
                                    node_features[:, 1] / node_features[:, 0], 0)  # avg outgoing
    node_features[:, 5] = np.where(node_features[:, 2] > 0, 
                                    node_features[:, 3] / node_features[:, 2], 0)  # avg incoming
    
    # Normalize node features
    node_features_scaled = StandardScaler().fit_transform(node_features)
    
    print(f"\nNode features shape: {node_features_scaled.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge features shape: {edge_features_scaled.shape}")
    print(f"Edge labels shape: {edge_labels.shape}")
    print(f"Fraud edges: {np.sum(edge_labels):,} ({np.mean(edge_labels)*100:.4f}%)")
    
    # Save as numpy arrays (can be loaded into PyTorch Geometric later)
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'gnn_graph.npz'),
        node_features=node_features_scaled,
        edge_index=edge_index,
        edge_features=edge_features_scaled,
        edge_labels=edge_labels,
        account_to_idx=np.array(list(account_to_idx.items()), dtype=object),
        node_feature_names=['out_count', 'out_amount', 'in_count', 'in_amount', 'avg_out', 'avg_in'],
        edge_feature_names=['amount', 'hour', 'is_cross_border', 'is_currency_exchange']
    )
    
    print(f"Saved to: {OUTPUT_DIR}/gnn_graph.npz")

def verify_data():
    """Verify processed data integrity."""
    print("\n" + "=" * 60)
    print("VERIFYING PROCESSED DATA")
    print("=" * 60)
    
    # Check K-means data
    print("\n[K-means Data]")
    kmeans_data = np.load(os.path.join(OUTPUT_DIR, 'kmeans_features.npz'), allow_pickle=True)
    print(f"  Shape: {kmeans_data['X'].shape}")
    print(f"  Features: {list(kmeans_data['feature_names'])}")
    print(f"  NaN values: {np.isnan(kmeans_data['X']).sum()}")
    
    # Check RF data
    print("\n[Random Forest Data]")
    rf_train = np.load(os.path.join(OUTPUT_DIR, 'rf_train.npz'), allow_pickle=True)
    rf_test = np.load(os.path.join(OUTPUT_DIR, 'rf_test.npz'), allow_pickle=True)
    print(f"  Train shape: {rf_train['X'].shape}, Labels: {rf_train['y'].shape}")
    print(f"  Test shape: {rf_test['X'].shape}, Labels: {rf_test['y'].shape}")
    print(f"  Train class dist: 0={np.sum(rf_train['y']==0):,}, 1={np.sum(rf_train['y']==1):,}")
    print(f"  Test class dist: 0={np.sum(rf_test['y']==0):,}, 1={np.sum(rf_test['y']==1):,}")
    print(f"  Class weights: {rf_train['class_weights']}")
    print(f"  NaN in train: {np.isnan(rf_train['X']).sum()}")
    print(f"  NaN in test: {np.isnan(rf_test['X']).sum()}")
    
    # Check GNN data
    print("\n[GNN Data]")
    gnn_data = np.load(os.path.join(OUTPUT_DIR, 'gnn_graph.npz'), allow_pickle=True)
    print(f"  Nodes: {gnn_data['node_features'].shape[0]:,}")
    print(f"  Node features: {gnn_data['node_features'].shape[1]}")
    print(f"  Edges: {gnn_data['edge_index'].shape[1]:,}")
    print(f"  Edge features: {gnn_data['edge_features'].shape[1]}")
    print(f"  Fraud edges: {np.sum(gnn_data['edge_labels']):,}")
    print(f"  NaN in nodes: {np.isnan(gnn_data['node_features']).sum()}")
    print(f"  NaN in edges: {np.isnan(gnn_data['edge_features']).sum()}")
    
    print("\n✓ All data verified successfully!")

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"Started at: {start_time}")
    
    create_output_dir()
    total_rows = get_data_info()
    
    # Process data
    X, y, feature_names, all_accounts, all_graph_data, encoders = process_data_in_chunks()
    
    # Prepare model-specific data
    prepare_kmeans_data(X, feature_names)
    prepare_rf_data(X, y, feature_names)
    prepare_gnn_data(all_accounts, all_graph_data)
    
    # Verify
    verify_data()
    
    end_time = datetime.now()
    print(f"\n{'=' * 60}")
    print(f"COMPLETED!")
    print(f"Started: {start_time}")
    print(f"Ended: {end_time}")
    print(f"Duration: {end_time - start_time}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare SAML-D data for AML models')
    parser.add_argument('--verify', action='store_true', help='Only verify existing data')
    args = parser.parse_args()
    
    if args.verify:
        verify_data()
    else:
        main()
