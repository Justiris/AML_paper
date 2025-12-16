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
| **GNN** | **88.04%** | **99.85%** | **93.57%** | **0.9999** | **0.9985** |

### Key Findings

- **GNN dramatically outperforms** traditional methods by leveraging transaction graph structure
- GNN achieves **93.57% F1-Score** vs <1% for baseline models
- Only **3 fraudulent transactions missed** out of 1,975 (99.85% recall)
- Low false positive rate: 268 false alarms out of ~1.9M legitimate transactions

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
