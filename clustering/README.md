# Portfolio Clustering Methods

This module provides **7 different clustering methods** for portfolio optimization. Each method groups assets differently based on specific characteristics.

## Quick Start

```python
from clustering import CorrelationClusterer
import pandas as pd

# Load returns data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Cluster by correlation
clusterer = CorrelationClusterer(max_cluster_size=18, n_bits=10)
clusters = clusterer.cluster(returns)

# View results
print(f"Created {len(clusters)} clusters")
for cluster_id, tickers in clusters.items():
    print(f"{cluster_id}: {tickers}")
```

---

## Available Methods

### 1. CorrelationClusterer ⭐ **Recommended Default**

**What it does**: Groups assets that move together (high correlation)

**Best for**: General purpose, most assets

**Example**:
```python
from clustering import CorrelationClusterer

clusterer = CorrelationClusterer(
    max_cluster_size=18,
    use_absolute=True  # Ignores direction of correlation
)
clusters = clusterer.cluster(returns)
```

**Pros**:
- ✅ Fast and simple
- ✅ Captures co-movement
- ✅ Works well in practice

**Cons**:
- ❌ Ignores return levels (high/low performers mixed)

---

### 2. SectorClusterer 🏢 **Most Interpretable**

**What it does**: Groups assets by industry sector (fetches from Yahoo Finance)

**Best for**: When you want interpretable, stable clusters

**Example**:
```python
from clustering import SectorClusterer

clusterer = SectorClusterer(
    max_cluster_size=18,
    use_industry=False  # Use broad sectors (Technology, Finance, etc.)
)
clusters = clusterer.cluster(returns)

# See what sectors were found
summary = clusterer.get_sector_summary(clusters)
print(summary)
```

**Pros**:
- ✅ Very interpretable (e.g., "Technology sector")
- ✅ Stable over time
- ✅ Respects market structure

**Cons**:
- ❌ Requires Yahoo Finance API call (slower first time)
- ❌ Some tickers may have "Unknown" sector

**Pre-computed sector map** (faster):
```python
sector_map = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'JPM': 'Financials',
    # ...
}
clusterer = SectorClusterer(sector_map=sector_map)
```

---

### 3. ReturnsClusterer 📈 **For Growth/Value Separation**

**What it does**: Groups assets by average return level

**Best for**: Separating growth stocks from value stocks

**Example**:
```python
from clustering import ReturnsClusterer

clusterer = ReturnsClusterer(max_cluster_size=18)
clusters = clusterer.cluster(returns)
```

**Result**: High-return cluster, medium-return, low-return

**Pros**:
- ✅ Simple and fast
- ✅ Separates performers

**Cons**:
- ❌ Unstable (returns change over time)
- ❌ Ignores risk

---

### 4. VolatilityClusterer 📊 **For Risk-Based Allocation**

**What it does**: Groups assets by volatility (standard deviation)

**Best for**: When you want to control risk exposure per cluster

**Example**:
```python
from clustering import VolatilityClusterer

clusterer = VolatilityClusterer(max_cluster_size=18)
clusters = clusterer.cluster(returns)
```

**Result**: Low-vol cluster, medium-vol, high-vol

**Pros**:
- ✅ Risk-aware clustering
- ✅ Good for conservative portfolios

**Cons**:
- ❌ Ignores returns
- ❌ High-vol stocks may have high returns

---

### 5. DTWClusterer 🕐 **For Time Series Patterns** (Advanced)

**What it does**: Groups assets by time series shape similarity (handles lags)

**Best for**: When assets move together but with time delays

**Example**:
```python
from clustering import DTWClusterer

clusterer = DTWClusterer(
    max_cluster_size=18,
    window_size=10  # Sakoe-Chiba band
)
clusters = clusterer.cluster(returns)
```

**Pros**:
- ✅ Captures temporal patterns
- ✅ Robust to time lags

**Cons**:
- ❌ **Slow** (O(N² × T²) complexity)
- ❌ Requires `tslearn`: `pip install tslearn`

**When to use**: Momentum strategies, international markets (different trading hours)

---

### 6. GraphClusterer 🕸️ **For Network Communities** (Advanced)

**What it does**: Builds correlation network, finds communities

**Best for**: Finding natural market groups

**Example**:
```python
from clustering import GraphClusterer

clusterer = GraphClusterer(
    max_cluster_size=18,
    correlation_threshold=0.5,  # Min correlation to create edge
    algorithm='louvain'  # or 'greedy', 'label_prop'
)
clusters = clusterer.cluster(returns)
```

**Algorithms**:
- `louvain`: Best modularity (requires `python-louvain`)
- `greedy`: Fast approximation
- `label_prop`: Fastest

**Pros**:
- ✅ Finds natural communities
- ✅ Respects network structure

**Cons**:
- ❌ Requires `networkx`: `pip install networkx`
- ❌ Sensitive to threshold parameter

---

### 7. FactorClusterer 🔬 **For Factor Models** (Advanced)

**What it does**: Clusters by exposure to latent factors (PCA)

**Best for**: When you believe in factor-based investing

**Example**:
```python
from clustering import FactorClusterer

clusterer = FactorClusterer(
    max_cluster_size=18,
    n_factors=5  # Number of principal components
)
clusters = clusterer.cluster(returns)

# See factor exposures
exposures = clusterer.get_factor_exposures(returns)
print(exposures.head())

# See explained variance
variance = clusterer.get_explained_variance(returns)
print(variance)
```

**Pros**:
- ✅ Captures fundamental drivers
- ✅ Dimensionality reduction

**Cons**:
- ❌ Requires `scikit-learn`
- ❌ Factors may not be interpretable

---

## Choosing a Method

| **Use Case** | **Recommended Method** |
|--------------|----------------------|
| **General purpose** | `CorrelationClusterer` ⭐ |
| **Interpretability** | `SectorClusterer` |
| **Risk control** | `VolatilityClusterer` |
| **Growth vs Value** | `ReturnsClusterer` |
| **Time series patterns** | `DTWClusterer` |
| **Network analysis** | `GraphClusterer` |
| **Factor investing** | `FactorClusterer` |

---

## Common Parameters

All clusterers inherit from `BaseClusterer` and support:

```python
clusterer = AnyClusterer(
    max_cluster_size=18,  # Max assets per cluster (Zephyr constraint)
    n_bits=10,            # Binary discretization bits
    linkage_method='ward' # Hierarchical linkage ('ward', 'single', 'complete', 'average')
)
```

---

## Validation

All clusterers provide validation methods:

```python
# Check constraints are satisfied
is_valid = clusterer.validate_degree_constraint(clusters)

# Get cluster statistics
stats = clusterer.get_cluster_stats(clusters)
print(f"Clusters: {stats['n_clusters']}")
print(f"Avg size: {stats['avg_cluster_size']:.1f}")
print(f"Max size: {stats['max_cluster_size']}")
```

---

## Advanced: Custom Clustering

You can create your own clusterer by inheriting from `BaseClusterer`:

```python
from clustering.base import BaseClusterer
import numpy as np

class MyCustomClusterer(BaseClusterer):
    def compute_distance_matrix(self, returns):
        # Your custom distance logic here
        # Return: N×N distance matrix or condensed array

        # Example: Cluster by Sharpe ratio similarity
        sharpe_ratios = (returns.mean() / returns.std()).values.reshape(-1, 1)
        from scipy.spatial.distance import pdist
        return pdist(sharpe_ratios)
```

---

## Dependencies

**Core** (always needed):
- `numpy`
- `pandas`
- `scipy`

**Optional** (for specific methods):
- `yfinance` - for SectorClusterer
- `tslearn` - for DTWClusterer
- `networkx` - for GraphClusterer
- `python-louvain` - for GraphClusterer with Louvain algorithm
- `scikit-learn` - for FactorClusterer

Install all:
```bash
pip install numpy pandas scipy yfinance tslearn networkx python-louvain scikit-learn
```

---

## Performance

| Method | Speed | RAM Usage | Scalability |
|--------|-------|-----------|-------------|
| Correlation | ⚡⚡⚡ Fast | Low | 1000+ assets |
| Sector | ⚡⚡ Medium | Low | 1000+ assets |
| Returns | ⚡⚡⚡ Fast | Low | 1000+ assets |
| Volatility | ⚡⚡⚡ Fast | Low | 1000+ assets |
| DTW | 🐌 Slow | High | <100 assets |
| Graph | ⚡⚡ Medium | Medium | 500 assets |
| Factor | ⚡⚡ Medium | Medium | 500 assets |

---

## Examples

See [tools/clustering_comparison.py_comparison.py](../../../tools/clustering_comparison.py_comparison.py) for a full comparison of all methods.
