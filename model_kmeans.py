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


def evaluate_clustering(cluster_labels, y_true, anomaly_scores):
    """Evaluate clustering quality for fraud detection."""
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
    
    # Identify high-risk clusters (above average fraud rate)
    avg_fraud_rate = y_true.mean()
    high_risk_clusters = [i for i, rate in enumerate(cluster_fraud_rates) if rate > avg_fraud_rate * 2]
    print(f"\nHigh-risk clusters (>2x avg fraud rate): {high_risk_clusters}")
    
    # Use anomaly scores for classification
    print("\nAnomaly Score Based Classification:")
    print("-" * 50)
    
    # Find optimal threshold using different percentiles
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for percentile in [90, 95, 97, 99, 99.5]:
        threshold = np.percentile(anomaly_scores, percentile)
        y_pred = (anomaly_scores >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'percentile': percentile,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    print(f"Best threshold at percentile {best_metrics['percentile']}")
    print(f"  Threshold: {best_metrics['threshold']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1-Score: {best_metrics['f1']:.4f}")
    
    # ROC-AUC and PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
        pr_auc = average_precision_score(y_true, anomaly_scores)
        print(f"\n  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        best_metrics['roc_auc'] = roc_auc
        best_metrics['pr_auc'] = pr_auc
    except Exception as e:
        print(f"Could not compute AUC scores: {e}")
        best_metrics['roc_auc'] = 0
        best_metrics['pr_auc'] = 0
    
    # Confusion matrix at best threshold
    y_pred_best = (anomaly_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:>10,}  FP: {cm[0,1]:>10,}")
    print(f"  FN: {cm[1,0]:>10,}  TP: {cm[1,1]:>10,}")
    
    return best_metrics, cluster_fraud_rates


def save_results(kmeans, anomaly_scores, cluster_labels, metrics, cluster_fraud_rates):
    """Save model and results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save model artifacts
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'kmeans_results.npz'),
        cluster_centers=kmeans.cluster_centers_,
        cluster_labels=cluster_labels,
        anomaly_scores=anomaly_scores,
        cluster_fraud_rates=np.array(cluster_fraud_rates),
        metrics=metrics
    )
    
    print(f"\nResults saved to {RESULTS_DIR}/kmeans_results.npz")


def main():
    """Main execution."""
    print("=" * 60)
    print("K-MEANS CLUSTERING FOR AML DETECTION")
    print("=" * 60)
    
    # Load data
    X, feature_names, rf_test = load_data()
    
    # Get labels for the test portion (use consistent test set)
    n_test = len(rf_test['y'])
    X_test = X[-n_test:]  # Last portion matches test set
    y_test = rf_test['y']
    
    print(f"\nUsing last {n_test:,} samples for evaluation")
    print(f"Test fraud rate: {y_test.mean()*100:.4f}%")
    
    # Find optimal clusters using a sample
    sample_size = min(500000, len(X))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]
    
    optimal_n, cluster_scores = find_optimal_clusters(X_sample, N_CLUSTERS_RANGE)
    
    # Train on full data
    kmeans, cluster_labels = train_kmeans(X, optimal_n)
    
    # Compute anomaly scores
    anomaly_scores = compute_anomaly_scores(X, kmeans)
    
    # Evaluate on test set
    test_cluster_labels = cluster_labels[-n_test:]
    test_anomaly_scores = anomaly_scores[-n_test:]
    
    metrics, cluster_fraud_rates = evaluate_clustering(
        test_cluster_labels, y_test, test_anomaly_scores
    )
    
    # Save results
    save_results(kmeans, anomaly_scores, cluster_labels, metrics, cluster_fraud_rates)
    
    print("\n" + "=" * 60)
    print("K-MEANS MODEL COMPLETE")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    metrics = main()
