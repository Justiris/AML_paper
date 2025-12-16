# Anti-Money Laundering Detection with Machine Learning

Comparison of three machine learning approaches for fraud detection on transaction data.

## Models

1. **K-means Clustering** - Unsupervised anomaly detection
2. **Random Forest** - Supervised classification with SMOTE balancing
3. **Graph Neural Network (GNN)** - GraphSAGE-based edge classification

## Results Summary

| Model | ROC-AUC | PR-AUC | Recall | F1-Score |
|-------|---------|--------|--------|----------|
| K-means | 0.506 | 0.001 | 0.10 | 0.002 |
| Random Forest | 0.685 | 0.017 | 0.75 | 0.003 |
| GNN (Epoch 15) | N/A | **0.998** | N/A | **0.71** |

**Key Finding**: GNN significantly outperforms traditional methods by leveraging transaction graph structure.

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

- GNN training benefits significantly from GPU acceleration
- The SAML-D dataset is not included due to size (~1GB)
