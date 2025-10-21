# QPO Project Status

**Last Updated**: October 20, 2025

## Project Structure

```
portfolio-optimization/
├── qpo/                          # Main package
│   ├── commands/                 # CLI commands
│   │   ├── stocks.py            # List active stocks
│   │   ├── fetch.py             # Download historical data
│   │   └── portfolio.py         # Create portfolio matrix
│   ├── optimizers/              # Portfolio optimization methods
│   │   ├── classical.py         # Mean-variance (CVXPY + GA)
│   │   ├── equal_weight.py      # 1/N baseline
│   │   ├── risk_parity.py       # Equal risk contribution
│   │   ├── regularized.py       # L1/L2 regularization
│   │   ├── quantum.py           # Quantum optimizer (Independent Clusters)
│   │   └── backtest.py          # Backtesting framework
│   ├── qubo/                    # Quantum optimization
│   │   ├── formulation.py       # QUBO formulation
│   │   ├── solver.py            # Quantum solvers
│   │   ├── decoder.py           # Binary to weight decoder
│   │   └── aggregation.py       # Cluster aggregation
│   └── utils/                   # Utilities
│       ├── yahoo_api.py         # Yahoo Finance API
│       └── data_prep.py         # Data preprocessing
├── tests/                       # Test suite (197 tests passing)
│   ├── test_classical_optimizer.py  # Mean-variance tests
│   ├── test_classical_methods.py    # All classical methods
│   ├── test_qubo_*.py               # Quantum optimizer tests
│   └── test_*_clustering.py         # Clustering tests
├── clustering/                  # Asset clustering methods
│   ├── base.py                  # Base clusterer class
│   ├── hierarchical.py          # Legacy correlation clustering
│   ├── correlation.py           # Correlation-based clustering
│   ├── covariance.py            # Covariance-based (2 variants)
│   ├── returns_based.py         # Returns clustering
│   ├── volatility.py            # Volatility clustering
│   ├── sector.py                # Sector/industry clustering
│   ├── factor.py                # Factor (PCA) clustering
│   ├── dtw.py                   # Dynamic Time Warping
│   └── graph.py                 # Graph community detection
├── tools/                       # CLI tools
│   ├── clustering_comparison.py     # Compare all 11 clustering methods
│   ├── quantum_optimizer_demo.py    # Quantum optimizer demo
│   └── ...                          # Other tools
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # System architecture
│   └── CLASSICAL_METHODS.md     # Classical methods overview
├── SPEC.md                      # Technical specification
└── README.md                    # User guide
```

## Completed Components

### ✅ Data Pipeline
- Stock screening (`qpo stocks`)
- Historical data fetching (`qpo fetch`)
- Portfolio matrix creation (`qpo portfolio`)
- Data preprocessing and validation

### ✅ Classical Optimizers (Production-Ready)

1. **Mean-Variance (Markowitz)**
   - CVXPY solver (fast, exact)
   - GA solver (slow, cardinality support)
   - Location: [qpo/optimizers/classical.py](qpo/optimizers/classical.py)

2. **Equal-Weight (1/N)**
   - Simplest baseline
   - Often outperforms optimization (DeMiguel 2009)
   - Location: [qpo/optimizers/equal_weight.py](qpo/optimizers/equal_weight.py)

3. **Risk Parity**
   - Equal risk contribution
   - Best Sharpe ratio in tests (1.71)
   - Location: [qpo/optimizers/risk_parity.py](qpo/optimizers/risk_parity.py)

4. **L1 Regularized (Lasso)**
   - Sparse portfolios
   - Fast alternative to cardinality constraints
   - Location: [qpo/optimizers/regularized.py](qpo/optimizers/regularized.py)

5. **L2 Regularized (Ridge)**
   - Stable, uniform weights
   - Reduces estimation error
   - Location: [qpo/optimizers/regularized.py](qpo/optimizers/regularized.py)

### ✅ Quantum Optimizer (Independent Clusters Approach)

**Complete 5-stage pipeline:**
1. **Clustering** - 11 methods implemented:
   - Hierarchical (correlation-based)
   - Correlation, Covariance (euclidean & spectral), Returns, Volatility
   - Sector/Industry (uses Yahoo Finance metadata)
   - Factor (PCA-based, 3 & 5 components)
   - DTW (Dynamic Time Warping for temporal patterns)
   - Graph (community detection via Louvain/greedy modularity)

2. **QUBO Formulation** - Binary encoding with 10-bit discretization (1024 levels)
   - Location: [qpo/qubo/formulation.py](qpo/qubo/formulation.py)

3. **Quantum Solvers** - Simulated annealing, QPU, hybrid
   - Location: [qpo/qubo/solver.py](qpo/qubo/solver.py)

4. **Binary Decoder** - Converts binary solutions to portfolio weights
   - Location: [qpo/qubo/decoder.py](qpo/qubo/decoder.py)

5. **Aggregation** - 3 strategies (concatenate, proportional, uniform)
   - Location: [qpo/qubo/aggregation.py](qpo/qubo/aggregation.py)

**End-to-end Optimizer:**
- Location: [qpo/optimizers/quantum.py](qpo/optimizers/quantum.py)
- Class: `IndependentClustersOptimizer`

### ✅ Clustering Tools & Features

**Clustering Comparison Tool:**
- Benchmarks all 11 clustering algorithms
- Generates visualizations (heatmaps, metrics plots)
- Computes quality metrics (intra-correlation, separation score)
- Auto-detects portfolio-info.csv for sector clustering
- **New**: `--target-cluster-size` parameter to control granularity
- Location: [tools/clustering_comparison.py](tools/clustering_comparison.py)

**Target Cluster Size Feature:**
- Controls average cluster size to prevent too many small clusters
- Default: `max_cluster_size / 2` (e.g., max=18, target=9)
- Supported by all hierarchical-based algorithms
- Example: `--target-cluster-size 10` aims for ~10 assets per cluster

### ✅ Backtesting Framework
- Rolling window methodology
- Out-of-sample evaluation
- Comprehensive metrics (Sharpe, Sortino, drawdown, etc.)
- Location: [qpo/optimizers/backtest.py](qpo/optimizers/backtest.py)

### ✅ Testing Suite
- **197 tests passing** (all green)
- Parametrized tests (5-96 assets)
- Quantum optimizer tests (decoder, aggregation, end-to-end)
- Clustering tests (all 11 methods)
- Numerical stability tests
- Coverage: Classical optimizers, quantum pipeline, clustering

### ✅ Recent Bug Fixes & Improvements (October 2025)
1. **Fixed numerical stability in Factor clustering**
   - Added `StandardScaler` to normalize returns before PCA
   - Resolved RuntimeWarnings: divide by zero, overflow, invalid value
   - Specified `svd_solver='full'` for more stable decomposition

2. **Installed h5py for tslearn**
   - Resolved UserWarning about missing HDF5 support
   - Enables better DTW model serialization

3. **Added Sector clustering to comparison tool**
   - Auto-detects `portfolio-info.csv` in project root
   - Loads sector/industry data from Yahoo Finance metadata
   - Fastest clustering method (0.000s runtime)

4. **Implemented target cluster size control**
   - Added `--target-cluster-size` parameter to comparison tool
   - Updated `BaseClusterer._cut_dendrogram()` with penalty-based optimization
   - Significantly reduces number of small clusters (e.g., DTW: 91→controllable)

5. **Quantum Benchmarking Framework** (October 20, 2025)
   - **Modified quantum optimizer** for Backtester compatibility:
     - Added `return_dict` parameter to `optimize()` method
     - Returns dict format: `{weights, metrics, runtime, solver_only_runtime}`
     - Created `QuantumOptimizerWrapper` class for automatic format conversion
   - **Preprocessing time tracking**:
     - Separate timing for clustering vs solver execution
     - `solver_only_runtime` excludes clustering preprocessing
     - Enables fair comparison: QPU vs classical methods
   - **Comprehensive visualization suite** ([tools/visualizations.py](../tools/visualizations.py)):
     - Portfolio value comparison (multi-line plot)
     - Runtime comparison (bar chart with speedup annotations)
     - Weight allocation heatmaps (stacked area charts)
     - Risk-return scatter plot (with Sharpe ratio contours)
     - Performance metrics table (CSV export)
   - **Quantum vs Classical comparison tool** ([tools/quantum_classical_comparison.py](../tools/quantum_classical_comparison.py)):
     - Configurable solver types: simulated, QPU, hybrid
     - Supports all classical optimizers (Risk Parity, Equal-Weight, etc.)
     - Rolling window backtesting with customizable parameters
     - Automatic visualization generation
   - **CLI wrapper script** ([tools/run_quantum_benchmark.py](../tools/run_quantum_benchmark.py)):
     - Simple command-line interface
     - Default parameters optimized for typical use
     - Progress indicators and summary output
   - **Comprehensive documentation** ([docs/QUANTUM_BENCHMARK_GUIDE.md](../docs/QUANTUM_BENCHMARK_GUIDE.md)):
     - Quick start examples
     - Command-line options reference
     - Performance interpretation guide
     - Troubleshooting section
     - Advanced usage patterns

## Performance Benchmarks

### Backtest Results (32 assets, 1-year training, monthly rebalancing)

| Method | Sharpe | Annual Return | Max Drawdown | Runtime |
|--------|--------|---------------|--------------|---------|
| **Risk Parity** | **1.71** | 12.3% | **-4.6%** | 0.012s |
| **Equal-Weight** | 1.65 | **12.9%** | -5.0% | **0.0001s** |
| Mean-Variance | 0.29 | 11.9% | -29.7% | 0.004s |
| L1 (λ=0.01) | 0.29 | 11.9% | -29.7% | 0.004s |
| L2 (λ=0.1) | 0.18 | 6.6% | -28.9% | 0.004s |
| GA (k=16) | ~0.81 | ? | ? | 6.0s |

**Key Insights:**
- Simple methods (Equal-Weight, Risk Parity) win on risk-adjusted returns
- Mean-variance suffers from estimation error
- GA is ~1500x slower than CVXPY alternatives

### Clustering Performance (115 assets, real portfolio data)

**With `--target-cluster-size 10`:**

| Algorithm | Clusters | Min | Max | Avg | Runtime | Intra-Corr | Sep Score |
|-----------|----------|-----|-----|-----|---------|------------|-----------|
| Hierarchical | 11 | 4 | 14 | **10.5** | 0.005s | 0.411 | 0.068 |
| Correlation | 11 | 4 | 14 | **10.5** | 0.005s | 0.411 | 0.068 |
| Covariance | 18 | 1 | 16 | 6.4 | 0.013s | 0.383 | 0.130 |
| Covariance (Spectral) | 45 | 1 | 16 | 2.6 | 0.026s | 0.434 | **0.388** |
| Returns | 16 | 1 | 15 | 7.2 | 0.001s | 0.261 | -1.021 |
| Volatility | 14 | 1 | 16 | 8.2 | 0.001s | 0.285 | -0.763 |
| Factor (3) | 11 | 5 | 18 | **10.5** | 0.14s | 0.375 | -0.187 |
| Factor (5) | 12 | 3 | 16 | 9.6 | 0.005s | 0.402 | -0.013 |
| DTW | 91 | 1 | 13 | 1.3 | 12.3s | 0.448 | 0.477 |
| **Graph** | 51 | 1 | 18 | 2.3 | 0.08s | 0.462 | **0.550** |
| Sector | 13 | 1 | 18 | 8.8 | **0.000s** | 0.345 | -0.133 |

**Key Insights:**
- **Graph clustering** achieves best separation score (0.550)
- **Sector clustering** is instantaneous (uses pre-computed metadata)
- **Hierarchical methods** hit target size exactly (10.5 ≈ 10)
- **DTW** produces too many small clusters (needs post-processing)

## Usage Examples

### Basic Workflow
```bash
# 1. Get stock list
qpo stocks --output tickers.txt

# 2. Fetch historical data
qpo fetch --input tickers.txt --period 2y --output-dir data/stocks

# 3. Create portfolio matrix
qpo portfolio data/stocks/ --output portfolio.csv

# 4. Fetch stock metadata (for sector clustering)
qpo fetch-info --input tickers.txt --output portfolio-info.csv
```

### Python API - Classical Optimization
```python
from qpo.optimizers.risk_parity import RiskParityOptimizer
from qpo.optimizers.backtest import Backtester
import pandas as pd

# Load data
prices = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

# Optimize
optimizer = RiskParityOptimizer()
result = optimizer.optimize(returns)

# Backtest
backtester = Backtester(optimizer, train_days=252, test_days=21)
backtest_result = backtester.run(prices, returns)
```

### Python API - Quantum Optimization
```python
from qpo.optimizers.quantum import IndependentClustersOptimizer
from clustering import CorrelationClusterer
import pandas as pd

# Load data
prices = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

# Configure optimizer
clusterer = CorrelationClusterer(max_cluster_size=18, target_cluster_size=10)
optimizer = IndependentClustersOptimizer(
    clusterer=clusterer,
    solver_type='simulated',  # or 'qpu', 'hybrid'
    aggregation_strategy='proportional'
)

# Optimize
result = optimizer.optimize(returns)
print(f"Weights: {result.weights}")
print(f"Clusters: {result.clusters}")
```

### Quantum Optimizer Demo
```bash
# Run quantum optimizer with simulated annealing
python tools/quantum_optimizer_demo.py \
  --portfolio-csv portfolio.csv \
  --solver simulated \
  --compare-strategies

# Run clustering comparison
python tools/clustering_comparison.py \
  --portfolio-csv portfolio.csv \
  --max-cluster-size 18 \
  --target-cluster-size 10
```

## What Remains To Be Done

### Priority 1: Production Features
1. **CLI Commands for Optimization**
   - `qpo optimize-classical` - Run classical optimizers
   - `qpo optimize-quantum` - Run quantum optimizer
   - `qpo compare` - Compare methods side-by-side

2. **Visualization Tools**
   - Portfolio value comparison plots
   - Weight allocation charts
   - Runtime analysis graphs
   - Risk-return scatter plots

3. **Configuration Files**
   - YAML/JSON config for optimization parameters
   - Preset strategies (conservative, aggressive, balanced)

### Priority 2: Quantum-Classical Comparison
1. **Benchmark quantum vs classical methods** on same datasets
2. **Compare Independent Clusters vs Clustered Averaging** (second quantum approach)
3. **Evaluate QPU performance** vs simulated annealing
4. **Cost-benefit analysis** for D-Wave QPU access (~$2000/hour)

### Priority 3: Advanced Methods
1. **MIP Solver** - Replace slow GA with Mixed Integer Programming
2. **Black-Litterman Model** - Incorporate investor views
3. **Robust Optimization** - Handle parameter uncertainty
4. **Transaction Costs** - Model realistic trading costs
5. **Clustered Averaging Approach** - Second quantum method from spec

### Priority 4: Documentation & Packaging
1. User guide with end-to-end examples
2. API reference documentation (Sphinx)
3. PyPI package distribution
4. Docker containerization for reproducibility
5. Jupyter notebooks with tutorials

## Known Issues & Technical Debt

### Current Issues

1. **DTW and Graph Clustering - Too Many Small Clusters**
   - **Problem**: DTW creates 91 clusters for 115 assets (79% singletons)
   - **Reason**: They override `cluster()` method, don't use base class logic
   - **Fix Needed**: Implement post-processing to merge small clusters
   - **Workaround**: Use `--target-cluster-size` with hierarchical methods

2. **GA Solver Performance - 1500x Slower**
   - **Impact**: ~6s runtime vs 0.004s for CVXPY
   - **Recommendation**: Replace with MIP solver (Priority 3, item 1)
   - **Status**: Low priority (L1 regularization is fast alternative)

3. **Quantum Optimizer Not Benchmarked**
   - **Status**: Fully implemented but no performance comparison vs classical
   - **Needed**: Run backtests comparing all methods (Priority 2, item 1)
   - **Blocker**: Need to decide on clustering method and aggregation strategy

4. **Missing Visualization** (flagged as "In Progress")
   - **Status**: Framework exists, plots not implemented
   - **Impact**: Hard to compare results visually
   - **Priority**: P1, item 2

### Technical Debt

1. **Test Coverage**
   - 197 tests passing, but quantum optimizer only has unit tests
   - Missing: Integration tests with real portfolio data
   - Missing: End-to-end backtest tests for quantum optimizer

2. **Documentation**
   - Code has docstrings but no published API documentation
   - No tutorials or Jupyter notebooks
   - SPEC.md exists but needs updating with recent changes

3. **PortfolioClustering Class**
   - Legacy class, doesn't inherit from BaseClusterer
   - Should either inherit from BaseClusterer or be deprecated
   - Used in comparison tool, so can't remove yet

4. **Risk Parity Warnings**
   - RuntimeWarnings in tests (divide by zero, overflow in matmul)
   - Location: [tests/test_risk_parity.py:47](tests/test_risk_parity.py:47)
   - Impact: Tests pass but logs are noisy
   - Fix: Add numerical conditioning in risk parity optimizer

### Open Questions

1. **Quantum Performance**: Does quantum annealing actually help for this problem?
   - **Action**: Need benchmarks to answer (Priority 2, item 1)

2. **Optimal Cluster Size**: What's best for quantum solvers?
   - Current: max=18 (D-Wave Zephyr constraint)
   - Question: Is smaller better for solution quality?
   - **Action**: Experiment with different target sizes

3. **Aggregation Strategy**: Which is best?
   - Options: concatenate, proportional, uniform
   - **Action**: Need empirical comparison (Priority 2, item 1)

4. **Clustering Method**: Which clustering algorithm is best for quantum optimization?
   - Graph has best separation score (0.550)
   - Sector is fastest (0.000s)
   - Correlation hits target size exactly
   - **Action**: Test all methods in full optimizer pipeline

## Next Steps Recommendation

### Immediate (This Week)
1. ✅ ~~Fix numerical stability in Factor clustering~~ (DONE)
2. ✅ ~~Add Sector clustering to comparison tool~~ (DONE)
3. ✅ ~~Implement target cluster size control~~ (DONE)
4. ✅ ~~**Quantum benchmarking framework implemented**~~ (DONE - October 20, 2025)
   - Modified quantum optimizer to support Backtester format
   - Added preprocessing time tracking (clustering vs solver)
   - Created comprehensive visualization suite (5 plot types)
   - Built quantum vs classical comparison tool
   - Documented usage in [QUANTUM_BENCHMARK_GUIDE.md](../docs/QUANTUM_BENCHMARK_GUIDE.md)
5. **TODO**: Fix DTW/Graph clustering to respect `--target-cluster-size`
6. **IN PROGRESS**: Run quantum vs classical benchmark on real portfolio data

### Short-term (This Month)
1. ✅ ~~Add visualization plots (portfolio value, weights, risk-return)~~ (DONE)
2. ✅ ~~Document quantum optimizer usage with examples~~ (DONE)
3. **TODO**: Implement `qpo optimize-quantum` CLI command
4. **TODO**: Create Jupyter notebook tutorial
5. **TODO**: Run comprehensive benchmarks with different clustering algorithms

### Long-term (Next Quarter)
1. Implement Clustered Averaging quantum approach
2. Replace GA with MIP solver
3. Add Black-Litterman and robust optimization
4. Publish package to PyPI
5. Write academic paper comparing quantum vs classical methods

## References

### Academic
- Markowitz (1952): Mean-variance optimization
- DeMiguel et al. (2009): Equal-weight often wins
- Maillard et al. (2010): Risk parity
- Brodie et al. (2009): L1 regularization

### Implementation
- CVXPY: Convex optimization
- D-Wave Ocean SDK: Quantum annealing
- SciPy: Numerical optimization
- scikit-learn: Clustering & PCA
- tslearn: Dynamic Time Warping
- networkx: Graph community detection

## Quick Start Testing

```bash
# Install dependencies
pip install -e .

# Run all tests
pytest -v  # 197 tests should pass

# Run specific test suites
pytest tests/test_classical_methods.py -v      # All classical methods
pytest tests/test_qubo_*.py -v                 # Quantum optimizer
pytest tests/test_*_clustering.py -v           # Clustering algorithms

# Run demos
python tools/quantum_optimizer_demo.py --portfolio-csv portfolio.csv
python tools/clustering_comparison.py --portfolio-csv portfolio.csv
```

## Project Health: 🟢 Strong

**Status**: Core functionality complete, ready for benchmarking and productionization.

**Main Gaps**:
1. Quantum-classical comparison benchmarks
2. Visualization tools
3. Production CLI commands

**Strengths**:
- Solid test coverage (197 tests passing)
- Multiple optimization methods (5 classical + 1 quantum)
- Comprehensive clustering toolkit (11 methods)
- Clean architecture with modular design

## Contact & Support

For questions or issues, see:
- [README.md](README.md) for usage guide
- [SPEC.md](SPEC.md) for technical details
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
