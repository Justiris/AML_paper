"""
Random Forest Classifier for Anti-Money Laundering Detection
Uses SMOTE-balanced training data with class weights.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score, 
    average_precision_score, roc_curve, precision_recall_curve
)
import warnings
import time
import os
import joblib

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'processed_data'
RESULTS_DIR = 'results'
RANDOM_STATE = 42


def load_data():
    """Load preprocessed training and test data."""
    print("Loading Random Forest data...")
    
    train_data = np.load(os.path.join(DATA_DIR, 'rf_train.npz'), allow_pickle=True)
    test_data = np.load(os.path.join(DATA_DIR, 'rf_test.npz'), allow_pickle=True)
    
    X_train = train_data['X']
    y_train = train_data['y']
    class_weights = train_data['class_weights']
    feature_names = train_data['feature_names']
    
    X_test = test_data['X']
    y_test = test_data['y']
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Train class distribution: 0={np.sum(y_train==0):,}, 1={np.sum(y_train==1):,}")
    print(f"Test class distribution: 0={np.sum(y_test==0):,}, 1={np.sum(y_test==1):,}")
    print(f"Class weights: {class_weights}")
    
    return X_train, y_train, X_test, y_test, class_weights, feature_names


def train_model(X_train, y_train, class_weights):
    """Train Random Forest model."""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)
    
    # Convert class weights to dict format
    weights_dict = {0: class_weights[0], 1: class_weights[1]}
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=weights_dict,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\nModel parameters:")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  class_weight: {weights_dict}")
    
    print("\nTraining...")
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    
    print(f"\nTraining completed in {training_time:.2f}s")
    
    return model, training_time


def compute_metrics(model, X, y, set_name=""):
    """Compute metrics for a given dataset."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    
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
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'y_prob': y_prob,
        'y_pred': y_pred
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """Evaluate model performance on both train and test sets."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Training metrics
    train_metrics = compute_metrics(model, X_train, y_train, "Training Set")
    
    # Test metrics
    test_metrics = compute_metrics(model, X_test, y_test, "Test Set")
    
    # Test confusion matrix details
    cm = test_metrics['confusion_matrix']
    print(f"\nTest Confusion Matrix:")
    print(f"  TN: {cm[0,0]:>10,}  FP: {cm[0,1]:>10,}")
    print(f"  FN: {cm[1,0]:>10,}  TP: {cm[1,1]:>10,}")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    print("-" * 50)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Compute curves for saving
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_metrics['y_prob'])
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, test_metrics['y_prob'])
    
    curves = {
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }
    
    return train_metrics, test_metrics, curves, importances


def save_results(model, train_metrics, test_metrics, curves, feature_importances, feature_names):
    """Save model and results for both train and test."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, 'rf_model.joblib')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save results (both train and test)
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'rf_results.npz'),
        # Test metrics
        y_prob=test_metrics['y_prob'],
        precision=test_metrics['precision'],
        recall=test_metrics['recall'],
        f1=test_metrics['f1'],
        roc_auc=test_metrics['roc_auc'],
        pr_auc=test_metrics['pr_auc'],
        confusion_matrix=test_metrics['confusion_matrix'],
        # Training metrics
        train_precision=train_metrics['precision'],
        train_recall=train_metrics['recall'],
        train_f1=train_metrics['f1'],
        train_roc_auc=train_metrics['roc_auc'],
        train_pr_auc=train_metrics['pr_auc'],
        train_confusion_matrix=train_metrics['confusion_matrix'],
        # Other
        feature_importances=feature_importances,
        feature_names=feature_names,
        fpr=curves['fpr'],
        tpr=curves['tpr'],
        precision_curve=curves['precision_curve'],
        recall_curve=curves['recall_curve']
    )
    
    print(f"Results saved to {RESULTS_DIR}/rf_results.npz")


def main():
    """Main execution."""
    print("=" * 60)
    print("RANDOM FOREST FOR AML DETECTION")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test, class_weights, feature_names = load_data()
    
    # Train model
    model, training_time = train_model(X_train, y_train, class_weights)
    
    # Evaluate on both train and test
    train_metrics, test_metrics, curves, feature_importances = evaluate_model(
        model, X_train, y_train, X_test, y_test, feature_names
    )
    
    # Save results
    save_results(model, train_metrics, test_metrics, curves, feature_importances, feature_names)
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL COMPLETE")
    print("=" * 60)
    
    return test_metrics


if __name__ == '__main__':
    metrics = main()
