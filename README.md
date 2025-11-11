# Graph Neural Networks: Generalization and Graph Parameters

## Overview

This repository analyzes the relationship between graph parameters and generalization performance across different GNN architectures. Research examines 49 TUDataset benchmarks using four GNN models (GCN, SGC, GATv2, MPNN), computing 9 structural parameters and their correlations with generalization error.

**Key Finding**: Graph density (0.41) and clustering coefficient (0.49) show moderate correlation with generalization error, while basic properties like degree and diameter show weak correlation.

---

## Quick Start

```bash
# Setup environment
conda env create -f environment.yml && conda activate bt

# Generate reproducibility seeds
cd src && python generate_seeds.py

# Compute graph parameters
python main.py --function parameter --dataset test_dataset --verbose

# Run hyperparameter optimization
python setup_sweep.py --project_name bt_GCN
python run.py --project_name bt_GCN
```

---

## Installation

**Prerequisites**: Python 3.12, Conda, CUDA GPU (optional)

```bash
conda env create -f environment.yml
conda activate bt
```

**Key Dependencies**: PyTorch 2.2.0, PyTorch Geometric 2.5.2, NetworkX 3.3, Weights & Biases

---

## Project Structure

```
src/
├── main.py           # Analysis pipeline (parameters, correlation, errors)
├── train.py          # Training with k-fold CV and early stopping
├── net.py            # GNN models: GCN, GATv2, MPNN, SGC
├── utils.py          # Dataset loading, graph metrics, W&B config
├── run.py            # Execute W&B hyperparameter sweeps
└── ...               # Additional utility scripts

data/
├── TUDataset/        # Downloaded graph datasets
├── seeds/            # Shuffling seeds for reproducibility
└── *.txt             # Dataset lists by category

results/
├── parameters_*.csv           # Computed graph parameters
├── generalization_error.csv   # Model performance metrics
└── correlation_*.png          # Visualization plots
```

---

## Usage

### 1. Compute Graph Parameters

```bash
# Interactive mode
python main.py  # Select option 2

# Command-line
python main.py --function parameter --dataset bioinformatics --verbose

# Using Makefile
make p1  # bioinformatics
make p2  # computer vision
make p3  # social networks
```

**Computed metrics**: degree, shortest path, diameter, density, clustering, closeness/betweenness/eigenvector centrality, 1-WL color count

**Output**: `results/parameters_<category>.csv`

### 2. Train Models

```python
from train import train_procedure
from utils import TrainParams

params = TrainParams(
    hidden_size=64,
    num_hidden_layers=2,
    batch_size=128,
    learning_rate=0.001,
    patience_earlystopping=100,
    patience_plateau=30,
    normlization='batch'
)

# 10-fold cross-validation
train_procedure('PROTEINS', 'GCN', params, num_folds=10)
```

### 3. Hyperparameter Optimization

```bash
# Setup W&B sweeps
python setup_sweep.py --project_name bt_GCN

# Run optimization (Bayesian, 50 trials)
python run.py --project_name bt_GCN

# Extract best configurations
python -c "from main import get_best_hyperparameters; get_best_hyperparameters('bt_GCN')"
```

### 4. Analyze Correlations

```python
from main import get_correlation
get_correlation(model='GCN')  # Generates correlation plots
```

**Output**: `results/correlation_GCN.png` (color-coded by dataset size)

---

## Models & Parameters

### GNN Architectures

| Model | Description | Features |
|-------|-------------|----------|
| **GCN** | Graph Convolutional Network | Spectral convolution, efficient |
| **GATv2** | Graph Attention Network v2 | Dynamic attention, multi-head, residual |
| **MPNN** | Message Passing Neural Network | Edge features, configurable MLPs |
| **SGC** | Simplified Graph Convolution | Linear, no intermediate activations |

### Graph Parameters & Correlations

| Parameter | Correlation | Insight |
|-----------|------------|---------|
| **Density** | **0.41** | **Moderate positive** |
| **Clustering** | **0.49** | **Moderate positive** |
| Degree | 0.10 | Weak |
| Shortest path | -0.07 | Weak |
| Diameter | 0.03 | Weak |
| Centrality (all) | 0.16-0.21 | Weak |
| 1-WL colors | -0.17 | Weak negative |

---

## Datasets

**Source**: [TUDataset](https://chrsmrrs.github.io/datasets/) - 49 benchmark datasets

**Categories**: Bioinformatics, Computer Vision, Social Networks, Small Molecules, Synthetic

**Criteria**: >100 graphs, node attributes, classification tasks, max 4000 graphs

---

## Configuration

### Training Hyperparameters

```python
TrainParams(
    hidden_size=64,              # [32, 64, 128, 256]
    num_hidden_layers=2,         # [2, 4, 6]
    batch_size=128,              # [32, 64, 128]
    learning_rate=0.001,         # [0.0001, 0.01]
    patience_earlystopping=100,  # [100, 200]
    patience_plateau=30,         # [10, 20, 30]
    normlization='batch'         # ['batch', 'graph']
)
```

**Model-specific**:
- **GATv2**: `heads`, `dropout`, `residual`
- **MPNN**: `mlp_hidden_dim`

---

## Workflow

```
Dataset Selection → Seed Generation → Parameter Computation → 
Hyperparameter Optimization → Model Training → Error Calculation → 
Correlation Analysis → Visualization
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` or `hidden_size` |
| Dataset error | Run `python run_test.py` |
| W&B login | Run `wandb login` |
| Import error | `conda env update -f environment.yml` |

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Parameter computation, correlation analysis |
| `train.py` | Model training with k-fold CV |
| `run.py` | Execute W&B sweeps |
| `setup_sweep.py` | Configure hyperparameter optimization |
| `generate_seeds.py` | Create reproducible shuffles |
| `run_test.py` | Test dataset compatibility |

---

## Results Interpretation

**Generalization Error** = Training Accuracy - Test Accuracy (lower is better)

**Correlation Coefficients**:
- `> 0.5`: Strong
- `0.3-0.5`: Moderate
- `0.1-0.3`: Weak
- `< 0.1`: Negligible

**Files**:
- Parameters: `results/parameters_*.csv`
- Errors: `results/generalization_error_*.csv`
- Plots: `results/correlation_*.png`

---

## Additional Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Comprehensive guide (25+ pages)
- **[QUICK_START.md](QUICK_START.md)** - Fast-track commands
- **[CODE_REFERENCE.md](CODE_REFERENCE.md)** - Complete API reference
- **[INDEX.md](INDEX.md)** - Navigation guide
