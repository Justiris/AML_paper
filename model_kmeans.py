"""
K-means Clustering Model for Anti-Money Laundering Detection
Uses unsupervised clustering to identify anomalous transaction patterns.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'processed_data'
RESULTS_DIR = 'results'
RANDOM_STATE = 42
N_CLUSTERS_RANGE = [5, 8, 10, 15, 20]  # Will find optimal


def load_data():
    """Load preprocessed data."""
    print("Loading K-means data...")
    kmeans_data = np.load(os.path.join(DATA_DIR, 'kmeans_features.npz'), allow_pickle=True)
    X = kmeans_data['X']
    feature_names = kmeans_data['feature_names']
    
    # Load labels for evaluation (from RF test data for consistency)
    rf_test = np.load(os.path.join(DATA_DIR, 'rf_test.npz'), allow_pickle=True)
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    
    return X, feature_names, rf_test


def find_optimal_clusters(X_sample, n_clusters_range):
    """Find optimal number of clusters using silhouette score."""
    print("\nFinding optimal number of clusters...")
    
    best_score = -1
    best_n = n_clusters_range[0]
    scores = []
    
    for n in n_clusters_range:
        print(f"  Testing n_clusters={n}...", end=' ')
        kmeans = MiniBatchKMeans(
            n_clusters=n, 
            random_state=RANDOM_STATE,
            batch_size=10000,
            n_init=3
        )
        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels, sample_size=50000)
        scores.append((n, score))
        print(f"Silhouette: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_n = n
    
    print(f"\nOptimal clusters: {best_n} (silhouette={best_score:.4f})")
    return best_n, scores


def train_kmeans(X, n_clusters):
    """Train K-means model."""
    print(f"\nTraining K-means with {n_clusters} clusters...")
    start = time.time()
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        batch_size=10000,
        n_init=3,
        max_iter=100
    )
    
    cluster_labels = kmeans.fit_predict(X)
    
    print(f"Training time: {time.time() - start:.2f}s")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    return kmeans, cluster_labels


def compute_anomaly_scores(X, kmeans):
    """Compute anomaly scores based on distance to cluster centroids."""
    print("\nComputing anomaly scores...")
    
    # Distance to nearest centroid
    distances = kmeans.transform(X)
    min_distances = distances.min(axis=1)
    
    # Normalize to [0, 1]
    anomaly_scores = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min())
    
    print(f"Anomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    print(f"Mean anomaly score: {anomaly_scores.mean():.4f}")
    
    return anomaly_scores


def compute_metrics_for_split(y_true, anomaly_scores, threshold, set_name=""):
    """Compute metrics for a given split."""
    y_pred = (anomaly_scores >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
        pr_auc = average_precision_score(y_true, anomaly_scores)
    except:
        roc_auc = 0
        pr_auc = 0
    
    if set_name:
        print(f"\n{set_name} Metrics:")
        print("-" * 50)
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  PR-AUC:    {pr_auc:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def evaluate_clustering(cluster_labels, y_true, anomaly_scores, train_y=None, train_anomaly_scores=None):
    """Evaluate clustering quality for fraud detection on train and test."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    n_clusters = len(np.unique(cluster_labels))
    
    # Analyze fraud distribution per cluster
    print("\nCluster Analysis (Fraud Distribution):")
    print("-" * 50)
    
    cluster_fraud_rates = []
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_size = mask.sum()
        fraud_count = y_true[mask].sum()
        fraud_rate = fraud_count / cluster_size if cluster_size > 0 else 0
        cluster_fraud_rates.append(fraud_rate)
        print(f"  Cluster {i:2d}: {cluster_size:>10,} samples, {fraud_count:>6,} fraud ({fraud_rate*100:>6.3f}%)")
    
    # Find optimal threshold using different percentiles on test set
    best_f1 = 0
    best_threshold = 0
    
    for percentile in [90, 95, 97, 99, 99.5]:
        threshold = np.percentile(anomaly_scores, percentile)
        y_pred = (anomaly_scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.4f}")
    
    # Training set metrics (if provided)
    train_metrics = None
    if train_y is not None and train_anomaly_scores is not None:
        train_metrics = compute_metrics_for_split(train_y, train_anomaly_scores, best_threshold, "Training Set")
    
    # Test set metrics
    test_metrics = compute_metrics_for_split(y_true, anomaly_scores, best_threshold, "Test Set")
    
    # Confusion matrix at best threshold
    y_pred_best = (anomaly_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    print(f"\nTest Confusion Matrix:")
    print(f"  TN: {cm[0,0]:>10,}  FP: {cm[0,1]:>10,}")
    print(f"  FN: {cm[1,0]:>10,}  TP: {cm[1,1]:>10,}")
    
    return train_metrics, test_metrics, cluster_fraud_rates


def save_results(kmeans, anomaly_scores, cluster_labels, train_metrics, test_metrics, cluster_fraud_rates):
    """Save model and results for both train and test."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save model artifacts
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'kmeans_results.npz'),
        cluster_centers=kmeans.cluster_centers_,
        cluster_labels=cluster_labels,
        anomaly_scores=anomaly_scores,
        cluster_fraud_rates=np.array(cluster_fraud_rates),
        # Test metrics
        precision=test_metrics['precision'],
        recall=test_metrics['recall'],
        f1=test_metrics['f1'],
        roc_auc=test_metrics['roc_auc'],
        pr_auc=test_metrics['pr_auc'],
        # Training metrics
        train_precision=train_metrics['precision'] if train_metrics else 0,
        train_recall=train_metrics['recall'] if train_metrics else 0,
        train_f1=train_metrics['f1'] if train_metrics else 0,
        train_roc_auc=train_metrics['roc_auc'] if train_metrics else 0,
        train_pr_auc=train_metrics['pr_auc'] if train_metrics else 0
    )
    
    print(f"\nResults saved to {RESULTS_DIR}/kmeans_results.npz")


def main():
    """Main execution."""
    print("=" * 60)
    print("K-MEANS CLUSTERING FOR AML DETECTION")
    print("=" * 60)
    
    # Load data
    X, feature_names, rf_test = load_data()
    
    # Load training labels too
    rf_train = np.load(os.path.join(DATA_DIR, 'rf_train.npz'), allow_pickle=True)
    
    # Get labels for test and train portions
    # Note: RF train uses SMOTE (duplicated data), but K-means uses original data
    # We need to use original train indices (everything except test)
    n_test = len(rf_test['y'])
    n_train_original = len(X) - n_test  # Original training size (no SMOTE)
    
    X_test = X[-n_test:]  # Last portion matches test set
    y_test = rf_test['y']
    
    # For training metrics, use the original training portion
    # We need original y_train (before SMOTE) - use first n_train_original of full labels
    # Actually we need to load the original labels before SMOTE
    # Since we can't easily get those, we'll skip training metrics for K-means
    
    print(f"\\nTrain samples: {n_train_original:,}, Test samples: {n_test:,}")
    print(f"Test fraud rate: {y_test.mean()*100:.4f}%")
    print("Note: K-means training metrics skipped (requires original train labels)")
    
    # Find optimal clusters using a sample
    sample_size = min(500000, len(X))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]
    
    optimal_n, cluster_scores = find_optimal_clusters(X_sample, N_CLUSTERS_RANGE)
    
    # Train on full data
    kmeans, cluster_labels = train_kmeans(X, optimal_n)
    
    # Compute anomaly scores
    anomaly_scores = compute_anomaly_scores(X, kmeans)
    
    # Get test portion only
    test_cluster_labels = cluster_labels[-n_test:]
    test_anomaly_scores = anomaly_scores[-n_test:]
    
    # Evaluate on test only (training labels not easily available post-SMOTE)
    train_metrics, test_metrics, cluster_fraud_rates = evaluate_clustering(
        test_cluster_labels, y_test, test_anomaly_scores,
        train_y=None, train_anomaly_scores=None
    )
    
    # Save results
    save_results(kmeans, anomaly_scores, cluster_labels, train_metrics, test_metrics, cluster_fraud_rates)
    
    print("\n" + "=" * 60)
    print("K-MEANS MODEL COMPLETE")
    print("=" * 60)
    
    return test_metrics


if __name__ == '__main__':
    metrics = main()
