# Code Reference Guide

## Module Overview

This document provides a detailed reference for all modules, classes, and functions in the codebase.

---

## `src/net.py` - Neural Network Models

### GCN (Graph Convolutional Network)

```python
class GCN(nn.Module):
    """
    Graph Convolutional Network based on Kipf & Welling (2016).
    
    Architecture:
        Input -> GCNConv layers -> Normalization -> ReLU -> 
        Global pooling -> FC layers -> Output
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_hidden_layers, norm='batch')
    def forward(self, x, edge_index, batch, edge_weight=None)
    def reset_parameters(self)
```

**Parameters:**
- `in_channels`: Number of input node features
- `hidden_channels`: Hidden layer size
- `out_channels`: Number of output classes
- `num_hidden_layers`: Number of hidden GCN layers
- `norm`: Normalization type ('batch' or 'graph')

### GATv2 (Graph Attention Network v2)

```python
class GATv2(nn.Module):
    """
    Improved Graph Attention Network with dynamic attention.
    
    Key features:
    - Multi-head attention
    - Dropout for regularization
    - Optional residual connections
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_hidden_layers, norm='batch', heads=1, 
                 dropout=0, residual=False)
```

**Additional parameters:**
- `heads`: Number of attention heads
- `dropout`: Dropout probability (0 = no dropout)
- `residual`: Whether to use residual connections

### MPNN (Message Passing Neural Network)

```python
class MPNN(nn.Module):
    """
    Message Passing Neural Network using NNConv.
    
    Features edge weights through learned MLPs.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_hidden_layers, norm='batch', mlp_hidden_dim=32)
```

**Additional parameters:**
- `mlp_hidden_dim`: Hidden dimension for edge MLPs

### SGC (Simplified Graph Convolution)

```python
class SGC(nn.Module):
    """
    Simplified GCN without non-linearities between layers.
    Faster and more efficient than GCN.
    """
```

### Helper Function

```python
def get_model(model_name, in_channels, hidden_channels, out_channels, 
              num_hidden_layers=4, norm='batch', **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_name: 'GCN', 'GAT', 'GATv2', 'SGC', or 'MPNN'
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized model
    
    Raises:
        ValueError: If model_name not supported
    """
```

---

## `src/train.py` - Training & Evaluation

### Main Training Function

```python
def train_procedure(dataset_name: str, model_name: str, trainParams, 
                   is_wandb=False, num_folds: int = 5):
    """
    Train a GNN model with k-fold cross-validation.
    
    Args:
        dataset_name: Name of TUDataset dataset
        model_name: 'GCN', 'GATv2', 'SGC', or 'MPNN'
        trainParams: TrainParams object with hyperparameters
        is_wandb: Whether to log to Weights & Biases
        num_folds: Number of cross-validation folds (default: 5)
    
    Returns:
        best_test_acc: Average best test accuracy across folds
    
    Training features:
        - k-fold cross-validation with train/val/test splits
        - Early stopping based on validation loss
        - ReduceLROnPlateau learning rate scheduling
        - Adam optimizer with weight decay (1e-4)
        - Cross-entropy loss
        - Automatic GPU/CPU device selection
    
    Split strategy:
        - Test set: fold_i * n to (fold_i+1) * n
        - Val set: next fold (wraps around)
        - Train set: remaining data
    """
```

### Testing & Evaluation

```python
def train_test(dataset_name: str, model_name: str = 'GATv2'):
    """
    Quick testing function with predefined hyperparameters.
    
    Useful for:
        - Debugging model/dataset compatibility
        - Quick experiments
        - Verifying setup
    """

@torch.no_grad()
def test(loader):
    """
    Evaluate model accuracy on a data loader.
    
    Args:
        loader: PyTorch Geometric DataLoader
    
    Returns:
        accuracy: Fraction of correctly classified graphs
    """

@torch.no_grad()
def val(loader):
    """
    Compute validation loss on a data loader.
    
    Args:
        loader: PyTorch Geometric DataLoader
    
    Returns:
        loss: Average cross-entropy loss per graph
    """
```

### Hyperparameter Optimization

```python
def hyperparameter_tuning(config=None):
    """
    W&B sweep training function.
    
    Called by W&B agent during hyperparameter search.
    
    Args:
        config: W&B configuration dict (auto-provided)
    
    Logs to W&B:
        - Training/validation loss curves
        - Training/validation accuracy curves
        - Best test accuracy
    """
```

### Generalization Error

```python
def get_generalization_error_from_a_dataset(dataset_name: str, 
                                           model_name: str, 
                                           trainParams: TrainParams):
    """
    Compute generalization error with 10-fold cross-validation.
    
    Args:
        dataset_name: Dataset name
        model_name: Model architecture
        trainParams: Training hyperparameters
    
    Returns:
        mean: Mean generalization error across folds
        std: Standard deviation of generalization error
    
    Generalization error = Train accuracy - Test accuracy
    
    Process:
        1. Train with 10-fold CV
        2. Record best train/test accuracy per fold
        3. Compute train-test gap (generalization error)
        4. Return mean and std across folds
    """
```

---

## `src/utils.py` - Utility Functions

### Dataset Loading

```python
def load_dataset(dataset_name: str, use_shuffle_seed=False):
    """
    Load a TUDataset with optional reproducible shuffling.
    
    Args:
        dataset_name: Name from TUDataset collection
        use_shuffle_seed: If True, applies saved permutation from data/seeds/
    
    Returns:
        dataset: PyG InMemoryDataset
    
    Special handling:
        - IMDB-BINARY: Adds degree-based node features
        - Datasets >4000 graphs: Truncated to 4000
        - Applies shuffle seed if available and requested
    
    Raises:
        FileNotFoundError: If dataset not available in TUDataset
    """

class IMDBPreTransform(object):
    """
    Preprocessing for IMDB-BINARY dataset.
    Converts node degrees to one-hot features (max degree: 135).
    """
    def __call__(self, data)
```

### Training Configuration

```python
@dataclass
class TrainParams:
    """
    Training hyperparameter configuration.
    
    Required attributes:
        hidden_size: Hidden layer dimensions (e.g., 64)
        num_hidden_layers: Number of GNN layers (e.g., 2)
        batch_size: Training batch size (e.g., 128)
        patience_earlystopping: Early stopping patience (e.g., 100)
        patience_plateau: LR scheduler patience (e.g., 30)
        normlization: 'batch' or 'graph'
        learning_rate: Adam learning rate (e.g., 0.001)
    
    Optional attributes (model-specific):
        heads: Number of attention heads (GATv2)
        dropout: Dropout probability (GATv2)
        residual: Use residual connections (GATv2)
        mlp_hidden_dim: MLP hidden size (MPNN)
    
    Usage:
        trainParams = TrainParams(
            hidden_size=64,
            num_hidden_layers=2,
            batch_size=128,
            patience_earlystopping=100,
            patience_plateau=30,
            normlization='batch',
            learning_rate=0.001
        )
    """
```

### W&B Integration

```python
def setup_wandb_sweep(project_name: str = 'bt', 
                     dataset_name: str = 'DD', 
                     temp=False):
    """
    Create a Weights & Biases hyperparameter sweep.
    
    Args:
        project_name: W&B project name
        dataset_name: Dataset to optimize for
        temp: If True, adds '_temp' suffix (for testing)
    
    Returns:
        sweep_id: W&B sweep identifier
    
    Sweep configuration:
        Method: Bayesian optimization
        Metric: Maximize 'best_test_acc'
        
        Hyperparameters:
            - batch_size: [32, 64, 128]
            - hidden_size: [32, 64, 128, 256]
            - num_hidden_layers: [2, 4, 6]
            - learning_rate: uniform(0.0001, 0.01)
            - normlization: ['batch', 'graph']
            - default_patience: [100, 200]
            - patience_plateau: [10, 20, 30]
        
        Model-specific (GATv2):
            - heads: [1, 2, 4]
            - dropout: [0, 0.1, 0.2]
            - residual: [True, False]
        
        Model-specific (MPNN):
            - mlp_hidden_dim: [16, 32, 64]
        
        Early termination: Hyperband (s=0, eta=3, max_iter=81)
        Run cap: 50 trials
    """

def get_best_run(project_name: str, dataset_name: str):
    """
    Retrieve best hyperparameters from W&B sweep.
    
    Args:
        project_name: W&B project name
        dataset_name: Dataset name
    
    Returns:
        config: Dict of best hyperparameters, or None if not found
    
    Looks up sweep_id from data/dataset.csv and returns
    configuration of the run with highest best_test_acc.
    """
```

### Graph Parameters

```python
def get_average_degree(dataset_name, verbose=False):
    """Compute mean node degree across all graphs."""

def get_average_shortest_path(dataset_name, verbose=False):
    """
    Compute mean shortest path length.
    Handles disconnected graphs by computing per component.
    """

def get_graph_diameter(dataset_name, verbose=False):
    """
    Compute mean graph diameter.
    For disconnected graphs, uses largest connected component.
    """

def get_graph_density(dataset_name, verbose=False):
    """Ratio of actual edges to possible edges."""

def get_graph_clustering_coefficient(dataset_name, verbose=False):
    """Average clustering coefficient (triangle density)."""

def get_average_closeness_centrality(dataset_name, verbose=False):
    """Mean closeness centrality across all nodes."""

def get_average_betweenness_centrality(dataset_name, verbose=False):
    """Mean betweenness centrality across all nodes."""

def get_average_eigenvector_centrality(dataset_name, verbose=False):
    """
    Mean eigenvector centrality across all nodes.
    May fail for some graphs; uses max_iter=100000.
    """

def wl_1d_color_count(dataset_name, verbose=False):
    """
    Mean number of colors after 1-WL refinement stabilizes.
    
    Weisfeiler-Leman color refinement algorithm:
        1. Initialize all nodes with color 0
        2. Iteratively update colors based on neighbor colors
        3. Stop when colors stabilize
        4. Count distinct colors
    
    Higher count = more structurally diverse graph
    """
```

### Helper Functions

```python
def read_file_to_list(file_path):
    """
    Read dataset names from text file.
    
    Args:
        file_path: Path to .txt file with dataset names (one per line)
    
    Returns:
        list: Dataset names (stripped of whitespace)
    """

def number_of_graphs(dataset_name):
    """
    Get graph count for a dataset from TUDataset.csv.
    
    Args:
        dataset_name: Dataset name
    
    Returns:
        int: Number of graphs in dataset
    """

def plot_training_results(dataset_name, train_accs, val_accs, 
                          train_losses, val_losses, num_fold, 
                          is_temporal=True):
    """
    Create training curve plots.
    
    Generates two subplots:
        1. Accuracy curves (train vs validation)
        2. Loss curves (train vs validation)
    
    Saved to: results/result_{dataset_name}_on_fold_{num_fold}.pdf
    """

def timeSince(since):
    """Format elapsed time as 'Xm Ys'."""
```

---

## `src/main.py` - Analysis Pipeline

### Parameter Computation

```python
def calcualte_parameters():
    """
    Batch compute graph parameters for dataset list.
    
    Reads from: data/{args.dataset}.txt
    Writes to: results/parameters_{args.dataset}.csv
    
    Computes:
        - Average degree
        - Average shortest path
        - Graph diameter
        - Graph density
        - Clustering coefficient
        - Closeness centrality
        - Betweenness centrality
        - Eigenvector centrality
        - 1-WL color count
    
    Usage:
        python main.py --function parameter --dataset bioinformatics --verbose
    """
```

### Generalization Error

```python
def calculate_generalation_error(file_path: str):
    """
    Compute generalization error from best hyperparameters.
    
    Args:
        file_path: Path to best_hyperparameters CSV
    
    Process:
        1. Load best hyperparameters for each dataset
        2. Train model with those hyperparameters
        3. Compute generalization error (train - test accuracy)
        4. Save to results/generalization_error_{project_name}.csv
    
    CSV format:
        Name, Ave. generalization error, Standard deviation
    """
```

### Correlation Analysis

```python
def get_correlation(model: str = 'SGC'):
    """
    Generate correlation plots between parameters and generalization error.
    
    Args:
        model: Model name for file naming
    
    Reads:
        - results/output_{model}.csv (generalization errors)
        - results/parameters.csv (graph parameters)
    
    Creates:
        - results/correlation_{model}.png
        - results/correlation_ignore_less_than_1000_{model}.png
    
    Plots:
        - Scatter plots for each parameter vs generalization error
        - Color-coded by dataset size (blue/green/red)
        - Correlation coefficient displayed on each subplot
    
    Dataset size categories:
        - Blue: <1000 graphs
        - Green: 1000-4000 graphs
        - Red: >4000 graphs
    """

def compare_generalization_error_and_parameters():
    """
    Legacy function for comparing degree/path vs generalization error.
    Creates 2-subplot figure.
    """
```

### W&B Integration

```python
def get_best_hyperparameters(project_name: str = 'bt_GCN'):
    """
    Extract best hyperparameters from W&B sweeps.
    
    Args:
        project_name: W&B project name
    
    Process:
        1. Read sweep IDs from data/dataset.csv
        2. For each sweep, get run with highest best_test_acc
        3. Extract hyperparameter configuration
        4. Save to results/best_hyperparameters_{project_name}.csv
    
    Saved hyperparameters include:
        - sweep_id
        - model_name
        - batch_size
        - hidden_size
        - dataset_name
        - normlization
        - learning_rate
        - default_patience
        - patience_plateau
        - num_hidden_layers
        - [Model-specific parameters]
    """
```

### Aggregation

```python
def sum_the_parameters():
    """
    Aggregate parameters across all parameter CSV files.
    
    Reads: results/parameters_*.csv
    Writes: results/sum_parameters.csv
    
    Sums all numeric columns across datasets.
    Useful for getting overall statistics.
    """
```

### Interactive Mode

```python
def interactive_mode():
    """
    Present menu of analysis options.
    
    Options:
        1. Get generalization error
        2. Get Parameters
        3. Compare generalization error and parameters
        4. Get correlation
        5. Get best hyperparameters
        6. Sum the parameters
        7. Exit
    """

def handle_option(option):
    """Route user selection to appropriate function."""
```

---

## `src/parameter.py` - Graph Metrics (Alternative Interface)

### Class-Based Interface

```python
class GraphParameters:
    """
    Object-oriented interface for computing graph parameters.
    
    Usage:
        gp = GraphParameters('PROTEINS_full', verbose=True)
        avg_deg = gp.get_average_degree()
        diameter = gp.get_graph_diameter()
    
    Attributes:
        dataset: TUDataset instance
        verbose: Whether to show progress bars
    
    Methods mirror functions in utils.py:
        - get_average_degree()
        - get_average_shortest_path()
        - get_graph_diameter()
        - get_graph_density()
        - get_graph_clustering_coefficient()
        - get_graph_transitivity()
        - get_graph_assortativity()
        - get_average_closeness_centrality()
        - get_average_betweenness_centrality()
        - get_average_eigenvector_centrality()
        - wl_1d_color_count()
    """
```

**Note:** This module provides duplicate functionality to utils.py.
Use utils.py functions for consistency.

---

## `src/run.py` - Sweep Execution

```python
"""
Execute W&B sweeps for hyperparameter optimization.

Usage:
    python run.py --project_name bt_GCN

Process:
    1. Read dataset list from data/test_dataset.txt
    2. For each dataset:
        a. Check if sweep exists in data/dataset.csv
        b. If not, create sweep with setup_wandb_sweep()
        c. If sweep finished, skip
        d. Otherwise, run wandb.agent() for up to 432 trials
    
Command-line arguments:
    -p, --project_name: W&B project name (default: 'bt_MPNN')
    -i, --sweep_id: Specific sweep ID (optional)

The script handles:
    - Automatic sweep creation if needed
    - Skip finished sweeps
    - Resume incomplete sweeps
    - Save sweep metadata to CSV
"""
```

---

## `src/setup_sweep.py` - Sweep Configuration

```python
"""
Setup W&B sweeps without running them.

Usage:
    python setup_sweep.py --project_name bt_GCN
    python setup_sweep.py --project_name bt_GCN --temporory

Process:
    1. Read dataset list
    2. For each dataset, create W&B sweep
    3. Save sweep_id to data/dataset.csv

Command-line arguments:
    -t, --temporory: Add '_temp' suffix for testing

Use case:
    - Pre-create all sweeps before running
    - Test sweep configuration
    - Separate setup from execution
"""
```

---

## `src/generate_seeds.py` - Seed Generation

```python
"""
Generate reproducible shuffling seeds for datasets.

Usage:
    python generate_seeds.py

Process:
    1. Read dataset list from data/runnable_dataset.txt
    2. For each dataset:
        a. Load from TUDataset
        b. Shuffle and get permutation indices
        c. Save permutation to data/seeds/{dataset_name}.pt

Purpose:
    - Ensure consistent train/val/test splits across experiments
    - Enable reproducibility
    - Fair comparison between models

Permutation is applied in utils.load_dataset() when
use_shuffle_seed=True.
"""
```

---

## `src/run_test.py` - Dataset Testing

```python
"""
Test dataset compatibility and training.

Usage:
    python run_test.py

Process:
    1. Read datasets from data/test_dataset.txt
    2. For each dataset:
        a. Try to load and train with train_test()
        b. Record success or error
    3. Print summary

Output:
    Success: [list of working datasets]
    Error: [list of problematic datasets]

Use case:
    - Verify dataset compatibility before running sweeps
    - Debug loading issues
    - Identify problematic datasets
"""
```

---

## Data Files

### `data/dataset.csv`

```csv
name,project,sweep_id
MUTAG,bt_GCN,abc123xyz
PROTEINS,bt_GCN,def456uvw
...
```

**Columns:**
- `name`: Dataset name
- `project`: W&B project name
- `sweep_id`: W&B sweep identifier

**Purpose:** Track which sweeps have been created for which datasets.

### `data/*.txt` (Dataset Lists)

```
MUTAG
PROTEINS
DD
...
```

**Format:** One dataset name per line

**Files:**
- `test_dataset.txt`: Small test set
- `runnable_dataset.txt`: All working datasets
- `bioinformatics.txt`: Bio domain datasets
- `computer_vision.txt`: CV domain datasets
- `social_networks.txt`: Social network datasets
- `small_molecules.txt`: Chemistry datasets
- `synthetic.txt`: Synthetic datasets

### `data/seeds/*.pt`

PyTorch tensor files containing permutation indices.

**Usage:**
```python
perm = torch.load('data/seeds/MUTAG.pt')
dataset = dataset.index_select(perm)
```

---

## Results Files

### `results/parameters_*.csv`

```csv
Name,Ave. degree,Ave. shortest path,Graph diameter,...
MUTAG,2.19,3.42,7.5,...
...
```

### `results/generalization_error_*.csv`

```csv
Name,Ave. generalization error,Standard deviation
MUTAG,0.05,0.02
...
```

### `results/best_hyperparameters_*.csv`

```csv
sweep_id,model_name,batch_size,hidden_size,dataset_name,...
abc123,GCN,128,64,MUTAG,...
...
```

---

## Common Patterns

### Training a Model

```python
from train import train_procedure
from utils import TrainParams, load_dataset

# Load dataset
dataset = load_dataset('PROTEINS', use_shuffle_seed=True)

# Configure training
params = TrainParams(
    hidden_size=64,
    num_hidden_layers=2,
    batch_size=128,
    patience_earlystopping=100,
    patience_plateau=30,
    normlization='batch',
    learning_rate=0.001
)

# Train with 10-fold CV
acc = train_procedure('PROTEINS', 'GCN', params, num_folds=10)
print(f"Best test accuracy: {acc:.4f}")
```

### Computing All Parameters

```python
from utils import *

dataset_name = 'MUTAG'

params = {
    'degree': get_average_degree(dataset_name),
    'path': get_average_shortest_path(dataset_name),
    'diameter': get_graph_diameter(dataset_name),
    'density': get_graph_density(dataset_name),
    'clustering': get_graph_clustering_coefficient(dataset_name),
    'closeness': get_average_closeness_centrality(dataset_name),
    'betweenness': get_average_betweenness_centrality(dataset_name),
    'eigenvector': get_average_eigenvector_centrality(dataset_name),
    'wl_colors': wl_1d_color_count(dataset_name)
}
```

### Running a Complete Experiment

```python
# 1. Generate seeds (once)
# python generate_seeds.py

# 2. Setup sweep
from utils import setup_wandb_sweep
sweep_id = setup_wandb_sweep('bt_GCN', 'MUTAG')

# 3. Run sweep
import wandb
from train import hyperparameter_tuning
wandb.agent(sweep_id, hyperparameter_tuning, count=50, project='bt_GCN')

# 4. Get best config
from main import get_best_hyperparameters
get_best_hyperparameters('bt_GCN')

# 5. Compute generalization error
from main import calculate_generalation_error
calculate_generalation_error('results/best_hyperparameters_bt_GCN.csv')

# 6. Analyze correlations
from main import get_correlation
get_correlation('GCN')
```

---

## Error Handling

Most functions include error handling for common issues:

1. **Disconnected graphs**: Compute per-component
2. **Missing node features**: Use degree (IMDB-BINARY)
3. **W&B connection errors**: Logged but execution continues
4. **Dataset loading errors**: Caught and reported

Progress bars (via Rich) can be disabled with `verbose=False`.

---

## Performance Tips

1. **Use GPU**: Automatic if available
2. **Batch size**: Larger = faster but more memory
3. **Hidden size**: Smaller = faster training
4. **Early stopping**: Prevents unnecessary epochs
5. **Dataset size**: Truncated to 4000 graphs max

---

For more information, see:
- `DOCUMENTATION.md` - Comprehensive guide
- `QUICK_START.md` - Quick reference
- Individual source files - Inline documentation
