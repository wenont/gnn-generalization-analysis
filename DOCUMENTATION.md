# Graph Neural Networks: Generalization and Graph Parameters - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Research Purpose](#research-purpose)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [Code Structure](#code-structure)
6. [Datasets](#datasets)
7. [Experiment Workflow](#experiment-workflow)
8. [Results Interpretation](#results-interpretation)
9. [Troubleshooting](#troubleshooting)

## Overview

This repository contains the code for a bachelor thesis that investigates the impact of various graph parameters on the generalization error of different Graph Neural Network (GNN) architectures. The research conducts extensive experiments on 49 graph classification datasets from TUDataset using four GNN models:
- **GCN** (Graph Convolutional Networks)
- **SGC** (Simplified Graph Convolution)
- **GATv2** (Graph Attention Networks v2)
- **MPNN** (Message Passing Neural Networks)

## Research Purpose

The thesis aims to address a key gap in understanding GNN behavior: **how graph parameters/characteristics influence empirical generalization error**.

### Research Questions
The study examines the relationship between graph structural properties and GNN performance through:

1. **Graph Parameter Computation**: Computing and analyzing graph parameters across 49 datasets from TUDataset
   - Average degree
   - Clustering coefficient
   - Shortest path length
   - Graph diameter
   - Graph density
   - Centrality measures (closeness, betweenness, eigenvector)
   - 1-WL color count

2. **GNN Training & Evaluation**: Training various GNN architectures across multiple datasets to measure generalization error

3. **Correlation Analysis**: Identifying correlations between graph-theoretic parameters and generalization capabilities

### Key Findings

- **Weak correlations** exist between generalization error and basic properties (degree: 0.10, shortest path: -0.07, diameter: 0.03)
- **Moderate positive correlations** with graph density (0.41) and clustering coefficient (0.49)
- **Limited impact** from centrality measures (closeness: 0.16, betweenness: 0.19, eigenvector: 0.21)
- **Slight negative correlation** with 1-WL color count (-0.17)
- **Dataset size** significantly affects generalization stability

### Implications

The results suggest that:
- No single structural metric dominates generalization behavior
- Regularization techniques should be used for dense or highly clustered graphs
- Data augmentation strategies can mitigate biases in datasets with limited structural variability
- Architecture modifications should better capture global and local structural dependencies

## Installation

### Prerequisites
- Conda or Miniconda
- CUDA-compatible GPU (optional but recommended for faster training)
- Python 3.12

### Setup Environment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd GNN
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate bt
   ```

   Or using the Makefile:
   ```bash
   make install
   ```

### Key Dependencies
- **PyTorch** 2.2.0 (with CUDA 12.1 support)
- **PyTorch Geometric (PyG)** 2.5.2
- **NetworkX** 3.3 for graph algorithms
- **Weights & Biases (wandb)** for experiment tracking
- **pandas**, **numpy**, **matplotlib**, **scikit-learn**
- **Rich** for terminal UI

## Usage Guide

### 1. Dataset Preparation

Generate reproducible shuffling seeds for datasets:
```bash
cd src
python generate_seeds.py
```

**What this does:**
- Loads each dataset from TUDataset
- Generates a random permutation for shuffling
- Saves the permutation to `data/seeds/<dataset_name>.pt`
- Ensures reproducibility across different experiments

### 2. Testing Dataset Compatibility

Test if datasets can be loaded correctly:
```bash
cd src
python run_test.py
```

**What this does:**
- Reads dataset names from `data/test_dataset.txt`
- Attempts to load each dataset
- Runs a quick 10-epoch training test
- Reports success/error for each dataset
- Helps identify problematic datasets before running full experiments

### 3. Computing Graph Parameters

Compute graph-theoretic parameters for datasets:

**Interactive mode:**
```bash
cd src
python main.py
# Select option 2: "Get Parameters"
```

**Command-line mode:**
```bash
# Compute parameters for specific category
python main.py --function parameter --dataset test_dataset --verbose

# Using Makefile shortcuts
make p    # small molecules
make p1   # bioinformatics
make p2   # computer vision
make p3   # social networks
make t    # test datasets
```

**Computed parameters:**
- **Average degree**: Mean number of edges per node
- **Average shortest path**: Mean shortest path length between all node pairs
- **Graph diameter**: Maximum shortest path length
- **Graph density**: Ratio of actual to possible edges
- **Clustering coefficient**: Measure of node clustering
- **Closeness centrality**: How close a node is to all other nodes
- **Betweenness centrality**: How often a node lies on shortest paths
- **Eigenvector centrality**: Importance based on neighbor importance
- **1-WL color count**: Number of distinct colors after Weisfeiler-Leman refinement

**Results saved to:** `results/parameters_<category>.csv`

### 4. Hyperparameter Optimization with W&B

Set up and run Weights & Biases sweeps for hyperparameter tuning:

**Setup sweeps:**
```bash
cd src
python setup_sweep.py --project_name bt_GCN
```

**Run sweeps:**
```bash
python run.py --project_name bt_MPNN
```

**Sweep configuration includes:**
- Batch size: [32, 64, 128]
- Hidden size: [32, 64, 128, 256]
- Number of hidden layers: [2, 4, 6]
- Learning rate: uniform[0.0001, 0.01]
- Normalization: ['batch', 'graph']
- Early stopping patience: [100, 200]
- Plateau patience: [10, 20, 30]
- Model-specific parameters:
  - **GATv2**: heads [1, 2, 4], dropout [0, 0.1, 0.2], residual [True, False]
  - **MPNN**: mlp_hidden_dim [16, 32, 64]

**What this does:**
- Creates W&B sweep with Bayesian optimization
- Runs up to 50 trials per dataset
- Uses Hyperband early termination
- Saves best hyperparameters to `data/dataset.csv`
- Automatically resumes if sweep already exists

### 5. Training Models

#### Quick Test Training
```python
from train import train_test

# Test on a dataset
train_test('PROTEINS_full', 'GATv2')
```

#### Custom Training with Specific Hyperparameters
```python
from train import train_procedure
from utils import TrainParams

# Define hyperparameters
trainParams = TrainParams(
    hidden_size=64,
    num_hidden_layers=2,
    batch_size=128,
    patience_earlystopping=100,
    patience_plateau=30,
    normlization='batch',
    learning_rate=0.001
)

# Train with 10-fold cross-validation
train_procedure('PROTEINS_full', 'GCN', trainParams, num_folds=10)
```

#### Training Features
- **10-fold cross-validation**: Splits dataset into 10 parts
- **Early stopping**: Stops if validation loss doesn't improve
- **Learning rate scheduling**: ReduceLROnPlateau with patience
- **Automatic device selection**: Uses GPU if available
- **W&B integration**: Optional logging to Weights & Biases

### 6. Computing Generalization Error

Calculate generalization error from best hyperparameters:

```python
from main import calculate_generalation_error

# Reads best hyperparameters from CSV
calculate_generalation_error('results/best_hyperparameters_bt_GCN.csv')
```

**What this does:**
- Loads best hyperparameters for each dataset
- Trains models with 10-fold cross-validation
- Computes: Generalization Error = Train Accuracy - Test Accuracy
- Saves results to `results/generalization_error_<project>.csv`

### 7. Correlation Analysis

Analyze correlations between graph parameters and generalization error:

```python
from main import get_correlation

# Generate correlation plots
get_correlation(model='GCN')
```

**Output:**
- Scatter plots for each graph parameter vs. generalization error
- Correlation coefficients displayed on each plot
- Color-coded by dataset size:
  - Blue: <1000 graphs
  - Green: 1000-4000 graphs
  - Red: >4000 graphs
- Saved to `results/correlation_<model>.png`

### 8. Interactive Mode

Run the main script interactively:

```bash
cd src
python main.py
```

**Available options:**
1. **Get generalization error**: Compute generalization error from best hyperparameters
2. **Get Parameters**: Compute graph parameters for datasets
3. **Compare generalization error and parameters**: Create comparison plots
4. **Get correlation**: Generate correlation analysis plots
5. **Get best hyperparameters**: Extract best configurations from W&B
6. **Sum the parameters**: Aggregate parameters across categories
7. **Exit**: Close the program

## Code Structure

### Project Layout
```
GNN/
├── src/                          # Source code
│   ├── main.py                   # Main analysis pipeline
│   ├── train.py                  # Training procedures
│   ├── net.py                    # GNN model definitions
│   ├── utils.py                  # Utility functions
│   ├── parameter.py              # Graph parameter computation
│   ├── run.py                    # Sweep execution
│   ├── setup_sweep.py            # WandB sweep configuration
│   ├── generate_seeds.py         # Dataset shuffling seeds
│   └── run_test.py               # Dataset compatibility testing
├── data/                         # Data directory
│   ├── TUDataset/                # Downloaded datasets
│   ├── seeds/                    # Shuffling seeds
│   ├── dataset.csv               # Dataset metadata
│   └── *.txt                     # Dataset lists
├── results/                      # Experimental results
│   ├── generalization_error.csv
│   ├── parameters_*.csv
│   └── training_results/
├── paper/                        # Thesis LaTeX source
├── environment.yml               # Conda environment
├── Makefile                      # Automation commands
└── README.md
```

### Core Modules

#### `net.py` - Neural Network Architectures

**Models implemented:**

1. **GCN (Graph Convolutional Network)**
   - Based on Kipf & Welling (2016)
   - Spectral-based convolution
   - Simple and efficient

2. **GAT (Graph Attention Network)**
   - Attention mechanism for neighbor aggregation
   - Learns importance weights

3. **GATv2 (Graph Attention Network v2)**
   - Improved dynamic attention
   - Better than GAT on many benchmarks
   - Configurable heads, dropout, residual connections

4. **SGC (Simplified Graph Convolution)**
   - Removes non-linearities between layers
   - Faster training
   - Linear complexity

5. **MPNN (Message Passing Neural Network)**
   - General framework for GNNs
   - Edge features through NNConv
   - Configurable MLP for message functions

**Common architecture:**
```python
Input -> GNN Layers -> Global Pooling -> FC Layers -> Output
```

Each model includes:
- Configurable hidden layers
- Batch/graph normalization
- Global mean pooling
- Fully connected output layers
- `reset_parameters()` method for reinitialization

#### `train.py` - Training & Evaluation

**Key functions:**

1. **`train_procedure(dataset_name, model_name, trainParams, is_wandb, num_folds)`**
   - Main training loop
   - k-fold cross-validation (default: 5 or 10 folds)
   - Early stopping with patience
   - Learning rate scheduling
   - W&B logging support

2. **`train_test(dataset_name, model_name)`**
   - Quick testing function
   - Uses predefined hyperparameters
   - Useful for debugging

3. **`hyperparameter_tuning(config)`**
   - W&B integration
   - Called by W&B agent
   - Trains model with sweep configuration

4. **`get_generalization_error_from_a_dataset(dataset_name, model_name, trainParams)`**
   - Computes generalization error
   - Returns mean and standard deviation
   - Uses 10-fold cross-validation

**Training features:**
- Adam optimizer with weight decay (1e-4)
- ReduceLROnPlateau scheduler
- Cross-entropy loss
- Automatic device selection (CUDA/CPU)
- Training/validation/test accuracy tracking

#### `utils.py` - Utility Functions

**Key functions:**

1. **`load_dataset(dataset_name, use_shuffle_seed)`**
   - Loads dataset from TUDataset
   - Special handling for IMDB-BINARY (node features)
   - Applies shuffling seed if specified
   - Limits to 4000 graphs maximum

2. **`setup_wandb_sweep(project_name, dataset_name, temp)`**
   - Creates W&B sweep configuration
   - Bayesian optimization
   - Hyperband early termination
   - Model-specific parameters

3. **Graph parameter functions:**
   - `get_average_degree()`
   - `get_average_shortest_path()`
   - `get_graph_diameter()`
   - `get_graph_density()`
   - `get_graph_clustering_coefficient()`
   - `get_average_closeness_centrality()`
   - `get_average_betweenness_centrality()`
   - `get_average_eigenvector_centrality()`
   - `wl_1d_color_count()`

4. **`TrainParams` dataclass:**
   ```python
   @dataclass
   class TrainParams:
       hidden_size: int
       num_hidden_layers: int
       batch_size: int
       patience_earlystopping: int
       patience_plateau: int
       normlization: str
       learning_rate: float
       heads: Optional[int] = None         # GATv2
       dropout: Optional[float] = None     # GATv2
       residual: Optional[bool] = None     # GATv2
       mlp_hidden_dim: Optional[int] = None  # MPNN
   ```

#### `parameter.py` - Graph Metrics

Provides two interfaces:

1. **Class-based: `GraphParameters`**
   ```python
   gp = GraphParameters('PROTEINS_full', verbose=True)
   avg_degree = gp.get_average_degree()
   clustering = gp.get_graph_clustering_coefficient()
   ```

2. **Function-based:**
   ```python
   avg_degree = get_average_degree('PROTEINS_full', verbose=True)
   ```

**Features:**
- Progress bars with Rich library
- Error handling for disconnected graphs
- NetworkX integration
- Efficient computation

#### `main.py` - Analysis Pipeline

**Main functions:**

1. **`calculate_generalation_error(file_path)`**
   - Reads best hyperparameters from CSV
   - Trains models and computes generalization error
   - Saves results to CSV

2. **`calcualte_parameters()`**
   - Batch computes parameters for dataset list
   - Saves to CSV with tabulated output

3. **`get_correlation(model)`**
   - Creates correlation scatter plots
   - Color-codes by dataset size
   - Computes Pearson correlation coefficients

4. **`get_best_hyperparameters(project_name)`**
   - Extracts best runs from W&B sweeps
   - Saves to CSV

5. **`sum_the_parameters()`**
   - Aggregates parameters across categories

## Datasets

### Data Sources

The project uses the [TUDataset](https://chrsmrrs.github.io/datasets/) collection, containing graph classification benchmarks.

### Dataset Categories

1. **Bioinformatics**
   - Protein structures (PROTEINS, ENZYMES, DD)
   - Molecular activity (MUTAG, NCI1, NCI109)
   
2. **Computer Vision**
   - Image-based graphs (COIL-DEL, COIL-RAG)
   - Shape recognition (Letter-high, Letter-low, Letter-med)

3. **Social Networks**
   - Network structures (IMDB-BINARY, REDDIT-BINARY)
   - Collaboration networks

4. **Small Molecules**
   - Chemical compounds (BZR, COX2, DHFR)
   - Drug discovery datasets

5. **Synthetic**
   - Artificially generated graphs
   - Controlled properties for testing

### Selection Criteria

- Minimum number of graphs for statistical significance (typically >100)
- Availability of node attributes
- Compatibility with graph classification tasks
- Maximum 4000 graphs per dataset (for computational efficiency)
- Datasets must be runnable (no errors during loading)

### Special Handling

**IMDB-BINARY:**
- No node features provided
- Uses node degree as one-hot encoded features
- Preprocessed with `IMDBPreTransform`

## Experiment Workflow

### Complete Pipeline

1. **Dataset Selection**
   - Choose datasets from TUDataset categories
   - Create dataset list files (e.g., `data/test_dataset.txt`)

2. **Seed Generation**
   ```bash
   python generate_seeds.py
   ```
   - Creates reproducible shuffling seeds
   - Ensures consistent splits across experiments

3. **Parameter Computation**
   ```bash
   python main.py --function parameter --dataset bioinformatics --verbose
   ```
   - Calculates graph-theoretic properties
   - Saves to `results/parameters_*.csv`

4. **Sweep Setup**
   ```bash
   python setup_sweep.py --project_name bt_GCN
   ```
   - Configures W&B hyperparameter search
   - Creates sweep for each dataset

5. **Model Training**
   ```bash
   python run.py --project_name bt_GCN
   ```
   - Runs sweeps to find optimal hyperparameters
   - Uses Bayesian optimization
   - Early termination with Hyperband

6. **Best Model Selection**
   ```python
   get_best_hyperparameters('bt_GCN')
   ```
   - Extracts best-performing configurations
   - Saves to `results/best_hyperparameters_*.csv`

7. **Generalization Testing**
   ```python
   calculate_generalation_error('results/best_hyperparameters_bt_GCN.csv')
   ```
   - Evaluates on test sets with 10-fold CV
   - Computes generalization error

8. **Correlation Analysis**
   ```python
   get_correlation(model='GCN')
   ```
   - Compares parameters with generalization error
   - Generates correlation plots

9. **Visualization**
   - Review plots in `results/`
   - Check W&B dashboard for training curves

## Results Interpretation

### Output Formats

1. **CSV files**
   - `parameters_*.csv`: Graph parameters by dataset
   - `generalization_error_*.csv`: Error measurements
   - `best_hyperparameters_*.csv`: Optimal configurations

2. **PNG plots**
   - `correlation_*.png`: Scatter plots with correlations
   - Color-coded by dataset size

3. **W&B dashboards**
   - Interactive training curves
   - Hyperparameter importance
   - Run comparisons

### Generalization Error

**Formula:**
```
Generalization Error = Training Accuracy - Test Accuracy
```

**Interpretation:**
- **Lower values** indicate better generalization
- **Positive values**: Model overfits to training data
- **Negative values**: Unlikely, may indicate issues
- **Close to zero**: Good generalization

### Correlation Coefficients

**Interpretation:**
- **> 0.5**: Strong positive correlation
- **0.3 - 0.5**: Moderate positive correlation
- **0.1 - 0.3**: Weak positive correlation
- **-0.1 - 0.1**: No correlation
- **< -0.5**: Strong negative correlation

## Configuration

### Model Hyperparameters

**Via TrainParams:**
```python
trainParams = TrainParams(
    hidden_size=64,              # Hidden layer dimensions
    num_hidden_layers=2,         # Depth of GNN
    batch_size=128,              # Training batch size
    learning_rate=0.001,         # Optimizer learning rate
    patience_earlystopping=100,  # Early stopping patience
    patience_plateau=30,         # LR scheduler patience
    normlization='batch'         # 'batch' or 'graph'
)
```

### GNN-Specific Parameters

**GATv2:**
```python
trainParams.heads = 4           # Number of attention heads
trainParams.dropout = 0.1       # Dropout rate
trainParams.residual = True     # Residual connections
```

**MPNN:**
```python
trainParams.mlp_hidden_dim = 32  # MLP hidden dimension for messages
```

### W&B Configuration

**In `utils.py` -> `setup_wandb_sweep()`:**
- Method: Bayesian optimization
- Metric: Maximize `best_test_acc`
- Early termination: Hyperband
- Run cap: 50 trials per dataset

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   
   **Symptoms:**
   ```
   RuntimeError: CUDA out of memory
   ```
   
   **Solutions:**
   - Reduce batch size: `trainParams.batch_size = 32`
   - Reduce hidden size: `trainParams.hidden_size = 32`
   - Reduce number of layers: `trainParams.num_hidden_layers = 2`
   - Use CPU: Set `device = torch.device('cpu')`
   - Clear cache: `torch.cuda.empty_cache()`

2. **Dataset Loading Errors**
   
   **Symptoms:**
   ```
   RuntimeError: No graphs in dataset
   KeyError: 'x'
   ```
   
   **Solutions:**
   - Check dataset name spelling
   - Verify TUDataset compatibility
   - Use `run_test.py` to identify problematic datasets
   - Check if node attributes are available
   - For IMDB-BINARY, use special preprocessing

3. **W&B Connection Issues**
   
   **Symptoms:**
   ```
   wandb.errors.CommError
   ```
   
   **Solutions:**
   - Login: `wandb login`
   - Check API key: `wandb login --relogin`
   - Verify internet connection
   - Use offline mode: `wandb.init(mode='offline')`

4. **Missing Dependencies**
   
   **Symptoms:**
   ```
   ModuleNotFoundError: No module named 'xyz'
   ```
   
   **Solutions:**
   ```bash
   # Update environment
   conda env update -f environment.yml
   
   # Or reinstall
   conda env remove -n bt
   conda env create -f environment.yml
   ```

5. **NetworkX Errors**
   
   **Symptoms:**
   ```
   NetworkXError: Graph is not strongly connected
   NetworkXPointlessConcept: Null graph
   ```
   
   **Solutions:**
   - These are handled automatically in the code
   - For disconnected graphs, computations are done per component
   - Check verbose output for error rates

6. **Sweep Not Found**
   
   **Symptoms:**
   ```
   wandb.errors.CommError: Sweep not found
   ```
   
   **Solutions:**
   - Check sweep ID in `data/dataset.csv`
   - Verify project name
   - Create new sweep with `setup_sweep.py`

### Debugging Tips

1. **Enable verbose mode:**
   ```bash
   python main.py --function parameter --dataset test_dataset --verbose
   ```

2. **Test single dataset:**
   ```python
   from train import train_test
   train_test('MUTAG', 'GCN')
   ```

3. **Check dataset compatibility:**
   ```bash
   python run_test.py
   ```

4. **Review logs:**
   - Training logs in terminal output
   - W&B logs at wandb.ai
   - Error logs in `log/` directory

5. **Use smaller datasets first:**
   - Start with `MUTAG` (188 graphs)
   - Then try larger datasets

## Citation

If you use this code for your research, please cite:

```bibtex
@bachelorthesis{gnn_generalization_2025,
  author = {[Author Name]},
  title = {Investigating the Impact of Graph Parameters on Generalization Error in Graph Neural Networks},
  school = {RWTH Aachen University},
  year = {2025},
  type = {Bachelor's Thesis},
  supervisor = {Christopher Morris and Michael Schaub}
}
```

## Contributing

This is a research project for a bachelor thesis. For questions or suggestions:
- Contact the author or supervisor
- Open an issue on GitHub
- Submit a pull request

## License

[Specify license here - e.g., MIT, Apache 2.0, or Academic Use Only]

## Acknowledgements

- **Supervised by:** Prof. Christopher Morris and Prof. Michael Schaub
- **Special thanks to:** Chendi Qian for guidance and support throughout the thesis
- **Data source:** TUDataset for providing benchmark datasets
- **Framework:** PyTorch Geometric community for GNN implementations
- **Experiment tracking:** Weights & Biases for MLOps tools

## References

### Key Papers

1. Kipf & Welling (2016): Semi-Supervised Classification with Graph Convolutional Networks
2. Veličković et al. (2017): Graph Attention Networks
3. Brody et al. (2021): How Attentive are Graph Attention Networks?
4. Gilmer et al. (2017): Neural Message Passing for Quantum Chemistry
5. Morris et al. (2020): TUDataset: A collection of benchmark datasets for learning with graphs
6. Morris et al. (2024): Future Directions in the Theory of Graph Machine Learning

### Useful Links

- [TUDataset](https://chrsmrrs.github.io/datasets/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Weights & Biases](https://wandb.ai/)
- [NetworkX](https://networkx.org/)

---

**Last Updated:** 2025-02-13
**Version:** 1.0.0
