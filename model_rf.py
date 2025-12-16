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


def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Predictions
    print("\nGenerating predictions...")
    start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")
    
    # Classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print("\nClassification Metrics:")
    print("-" * 50)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # AUC scores
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    print(f"\n  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:>10,}  FP: {cm[0,1]:>10,}")
    print(f"  FN: {cm[1,0]:>10,}  TP: {cm[1,1]:>10,}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    print("-" * 50)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Compute curves for saving
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'feature_importances': importances,
        'training_time': inference_time
    }
    
    curves = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'pr_thresholds': pr_thresholds
    }
    
    return metrics, curves, y_prob


def save_results(model, metrics, curves, y_prob, feature_names):
    """Save model and results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, 'rf_model.joblib')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save results
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'rf_results.npz'),
        y_prob=y_prob,
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        roc_auc=metrics['roc_auc'],
        pr_auc=metrics['pr_auc'],
        confusion_matrix=metrics['confusion_matrix'],
        feature_importances=metrics['feature_importances'],
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
    
    # Evaluate
    metrics, curves, y_prob = evaluate_model(model, X_test, y_test, feature_names)
    
    # Save results
    save_results(model, metrics, curves, y_prob, feature_names)
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL COMPLETE")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    metrics = main()
