# Quick Start Guide

## Setup (One-time)

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate bt

# 2. Generate dataset seeds
cd src
python generate_seeds.py
```

## Common Tasks

### 1. Compute Graph Parameters

```bash
# Interactive mode
cd src
python main.py
# Choose option 2

# Command line
python main.py --function parameter --dataset bioinformatics --verbose

# Using Make
make p1  # bioinformatics
make p2  # computer vision
make p3  # social networks
```

### 2. Run Hyperparameter Optimization

```bash
cd src

# Setup sweeps (one-time per model)
python setup_sweep.py --project_name bt_GCN

# Run optimization
python run.py --project_name bt_GCN
```

### 3. Train a Single Model

```python
cd src
python
>>> from train import train_test
>>> train_test('PROTEINS_full', 'GCN')
```

### 4. Get Best Hyperparameters

```python
cd src
python
>>> from main import get_best_hyperparameters
>>> get_best_hyperparameters('bt_GCN')
```

### 5. Compute Generalization Error

```python
cd src
python
>>> from main import calculate_generalation_error
>>> calculate_generalation_error('results/best_hyperparameters_bt_GCN.csv')
```

### 6. Correlation Analysis

```python
cd src
python
>>> from main import get_correlation
>>> get_correlation(model='GCN')
```

## File Locations

### Input
- **Dataset lists**: `data/*.txt` (e.g., `test_dataset.txt`)
- **Best hyperparameters**: `results/best_hyperparameters_*.csv`
- **Sweep IDs**: `data/dataset.csv`

### Output
- **Graph parameters**: `results/parameters_*.csv`
- **Generalization errors**: `results/generalization_error_*.csv`
- **Correlation plots**: `results/correlation_*.png`
- **Training plots**: `results/result_*.pdf`

## Model Names

Use these names in function calls:
- `'GCN'` - Graph Convolutional Network
- `'SGC'` - Simplified Graph Convolution
- `'GATv2'` - Graph Attention Network v2
- `'MPNN'` - Message Passing Neural Network

## Project Names for W&B

- `'bt_GCN'` or `'bt'` - GCN experiments
- `'bt_SGC'` - SGC experiments
- `'bt_GATv2'` - GATv2 experiments
- `'bt_MPNN'` - MPNN experiments

## Quick Debugging

```bash
# Test if datasets load correctly
cd src
python run_test.py

# Check single dataset
python
>>> from utils import load_dataset
>>> dataset = load_dataset('MUTAG')
>>> print(dataset)
```

## Common Parameters

```python
from utils import TrainParams

trainParams = TrainParams(
    hidden_size=64,              # Try: 32, 64, 128
    num_hidden_layers=2,         # Try: 2, 4, 6
    batch_size=128,              # Try: 32, 64, 128
    learning_rate=0.001,         # Range: 0.0001-0.01
    patience_earlystopping=100,  # Stop after N epochs without improvement
    patience_plateau=30,         # Reduce LR after N epochs without improvement
    normlization='batch'         # Options: 'batch', 'graph'
)
```

## Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` or `hidden_size` |
| Dataset not found | Check spelling in dataset name |
| W&B login error | Run `wandb login` |
| Import error | Run `conda env update -f environment.yml` |
| Slow training | Use GPU or reduce dataset size |

## Typical Workflow

1. **Setup**: Install environment, generate seeds
2. **Compute parameters**: For your dataset category
3. **Setup sweeps**: For your chosen GNN model
4. **Run optimization**: Let W&B find best hyperparameters
5. **Get best configs**: Extract from W&B
6. **Compute errors**: Calculate generalization error
7. **Analyze**: Generate correlation plots

## Need More Help?

See `DOCUMENTATION.md` for detailed explanations of:
- Code structure and module details
- Advanced configuration options
- Detailed troubleshooting
- Research background and findings
