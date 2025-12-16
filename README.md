# Anti-Money Laundering Detection with Machine Learning

Comparison of three machine learning approaches for fraud detection on transaction data.

## Models

1. **K-means Clustering** - Unsupervised anomaly detection
2. **Random Forest** - Supervised classification with SMOTE balancing
3. **Graph Neural Network (GNN)** - GraphSAGE-based edge classification

## Results Summary

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|-----------|--------|----------|---------|--------|
| K-means | 0.11% | 10.48% | 0.22% | 0.506 | 0.001 |
| Random Forest | 0.15% | 74.68% | 0.30% | 0.685 | 0.017 |
| GNN (Base) | 88.04% | 99.85% | 93.57% | 0.9999 | 0.9985 |
| **GNN (+PayType)** | **90.38%** | **99.85%** | **94.88%** | **0.9998** | **0.9986** |

### Key Findings

- **GNN dramatically outperforms** traditional methods by leveraging transaction graph structure
- GNN (+PayType) achieves **94.88% F1-Score** vs <1% for baseline models
- Only **3 fraudulent transactions missed** out of 1,975 (99.85% recall)
- Low false positive rate: 210 false alarms out of ~1.9M legitimate transactions
- Adding **payment type features improved precision by 2.3%** (88.04%→90.38%)

#### GNN Model Comparison
| Variant | Edge Features | Precision | F1-Score | False Positives |
|---------|---------------|-----------|----------|----------------|
| GNN (Base) | 4 | 88.04% | 93.57% | 268 |
| GNN (+PayType) | 11 (+7 payment type) | **90.38%** | **94.88%** | **210** |

### Metrics Explained

All metrics are derived from the confusion matrix:

|  | Predicted Normal | Predicted Fraud |
|--|------------------|-----------------|
| **Actual Normal** | TN (True Negative) | FP (False Positive) |
| **Actual Fraud** | FN (False Negative) | TP (True Positive) |

**GNN Confusion Matrix**: TN=1,898,786 | FP=210 | FN=3 | TP=1,972

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision** | TP / (TP + FP) | Of all fraud predictions, how many were correct? |
| **Recall** | TP / (TP + FN) | Of all actual fraud, how many did we catch? |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean balancing both metrics |
| **ROC-AUC** | Area under TPR vs FPR curve | Overall discrimination ability (0.5=random, 1.0=perfect) |
| **PR-AUC** | Area under Precision-Recall curve | Better for imbalanced data like fraud detection |

### Why GNN Excels

| Model | Limitation |
|-------|------------|
| K-means | Unsupervised; can't learn fraud patterns from labels |
| Random Forest | Treats each transaction independently |
| **GNN** | Learns from graph structure (who sends money to whom) |

## GNN Features

The GNN uses two types of features:

### Node Features (per account)
| Feature | Description |
|---------|-------------|
| `out_count` | Number of outgoing transactions |
| `out_amount` | Total amount sent |
| `in_count` | Number of incoming transactions |
| `in_amount` | Total amount received |
| `avg_out` | Average outgoing transaction amount |
| `avg_in` | Average incoming transaction amount |

### Edge Features (per transaction)
| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount |
| `hour` | Time of day (0-23) |
| `is_cross_border` | Sender/receiver in different bank locations |
| `is_currency_exchange` | Payment currency ≠ Received currency |
| `payment_type_*` | One-hot encoded payment type (ACH, Cash Deposit, Cash Withdrawal, Cheque, Credit card, Cross-border, Debit card) |

## Future Work

### Feature Importance Analysis
GNNs lack built-in feature importance like Random Forest. Potential approaches:

| Method | Description |
|--------|-------------|
| Permutation Importance | Shuffle each feature, measure performance drop |
| Gradient-based Attribution | Compute gradients w.r.t. input features |
| Integrated Gradients | More robust attribution method |
| GNNExplainer | PyG's built-in explainability tool |

### Other Improvements
- Hyperparameter tuning (hidden dimensions, layers, dropout)
- Attention-based GNN architectures (GAT)
- Temporal modeling for transaction sequences
- Ensemble with traditional ML models

## Dataset

- **SAML-D**: ~9.5M transactions, ~0.1% fraud rate
- 855K unique accounts (graph nodes)
- 19 engineered features

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

# For PyTorch with AMD GPU (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install torch-geometric

# For PyTorch with NVIDIA GPU (CUDA)
pip install torch torchvision torchaudio
pip install torch-geometric
```

## Usage

```bash
# 1. Prepare data (requires SAML-D.csv)
python data_preparation.py

# 2. Train models
python model_kmeans.py
python model_rf.py
python model_gnn.py

# 3. Compare results
python compare_models.py
```

## Project Structure

```
├── data_preparation.py   # Data preprocessing pipeline
├── model_kmeans.py       # K-means clustering model
├── model_rf.py           # Random Forest classifier
├── model_gnn.py          # GNN with GraphSAGE
├── compare_models.py     # Model comparison and visualization
├── processed_data/       # Preprocessed data (not tracked)
└── results/              # Model outputs and metrics
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with ROCm/CUDA for GPU acceleration)
- PyTorch Geometric
- scikit-learn
- pandas, numpy, matplotlib, seaborn

## Notes

- GNN training: ~6.7 minutes on AMD Radeon RX 7900 XTX (ROCm 6.2)
- GNN early stopped at epoch 39/50 with patience=10
- The SAML-D dataset is not included due to size (~1GB)
- All model weights and results saved in `results/`
