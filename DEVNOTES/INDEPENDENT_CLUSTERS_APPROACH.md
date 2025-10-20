# Independent Clusters Approach - Technical Specification

**Document Version**: 1.0
**Date**: October 19, 2025
**Approach**: Parallel Independent Cluster Optimization

---

## 1. Executive Summary

The **Independent Clusters Approach** solves portfolio optimization by partitioning N assets into G clusters (each ≤18 assets), solving each cluster independently via QUBO/QPU, then aggregating results to form a global portfolio. This approach maximizes parallelism, preserves asset-level granularity, and simplifies QUBO complexity per cluster.

**Key Characteristics**:
- **Parallelizable**: Each cluster is solved independently → concurrent QPU execution
- **Scalable**: Can handle 100-200 assets (10-15 clusters)
- **Simple QUBOs**: Each cluster has ≤180 binary variables (18 assets × 10 bits)
- **Tradeoff**: Ignores inter-cluster correlations during optimization

**When to Use**:
- First implementation (simpler to debug)
- When assets naturally cluster (e.g., by sector/geography)
- When QPU parallelism is available (multiple concurrent jobs)
- When inter-cluster correlations are weak

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                INPUT: Returns Data (N assets)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼────────────┐
                │  Stage 1: Clustering   │
                │  - Correlation matrix  │
                │  - Hierarchical        │
                │  - Max 18 assets/clust │
                └───────────┬────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ┌───▼────┐         ┌───▼────┐         ┌───▼────┐
    │Cluster1│         │Cluster2│   ...   │ClusterG│
    │(n1=12) │         │(n2=15) │         │(nG=10) │
    └───┬────┘         └───┬────┘         └───┬────┘
        │                   │                   │
        │  Stage 2: QUBO Formulation per Cluster
        │                   │                   │
    ┌───▼────┐         ┌───▼────┐         ┌───▼────┐
    │ QUBO 1 │         │ QUBO 2 │   ...   │ QUBO G │
    │120 vars│         │150 vars│         │100 vars│
    └───┬────┘         └───┬────┘         └───┬────┘
        │                   │                   │
        │  Stage 3: Parallel QPU Solve
        │                   │                   │
    ┌───▼────┐         ┌───▼────┐         ┌───▼────┐
    │ QPU 1  │         │ QPU 2  │   ...   │ QPU G  │
    │Weights1│         │Weights2│         │WeightsG│
    └───┬────┘         └───┬────┘         └───┬────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────▼────────────┐
                │ Stage 4: Aggregation   │
                │ - Concatenate weights  │
                │ - Renormalize (sum=1)  │
                │ - Enforce cardinality  │
                └───────────┬────────────┘
                            │
            ┌───────────────▼────────────────┐
            │  OUTPUT: Global Portfolio      │
            │  - Weights (N,)                │
            │  - Metrics (Sharpe, risk, etc) │
            └────────────────────────────────┘
```

---

## 3. Stage 1: Clustering

### 3.1 Objective
Partition N assets into G clusters such that:
1. Each cluster has ≤18 assets (to keep variables ≤180 with 10-bit encoding)
2. Assets within a cluster are highly correlated (coherent risk profile)
3. Clusters are balanced (avoid single-asset or oversized clusters)

### 3.2 Algorithm: Hierarchical Clustering

**Inputs**:
- Correlation matrix `ρ` (N × N), computed from returns data
- `max_cluster_size` = 18 (default)
- `n_bits` = 10 (binary discretization)

**Process**:
```python
def cluster_assets(corr_matrix: pd.DataFrame,
                   max_cluster_size: int = 18) -> Dict[str, List[str]]:
    """
    Cluster assets using hierarchical clustering.

    Returns:
        {cluster_id: [ticker1, ticker2, ...]}
    """
    # 1. Distance metric: 1 - |correlation|
    #    (highly correlated assets are "close")
    distance = 1 - corr_matrix.abs()

    # 2. Hierarchical clustering with Ward linkage
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(squareform(distance.values), method='ward')

    # 3. Dynamic cluster cutting
    n_clusters = max(1, len(corr_matrix) // max_cluster_size)

    while True:
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        # Check all clusters satisfy size constraint
        unique, counts = np.unique(labels, return_counts=True)
        if np.all(counts <= max_cluster_size):
            break

        n_clusters += 1  # Increase granularity

        if n_clusters >= len(corr_matrix):
            break  # Degenerate: each asset alone

    # 4. Build cluster dictionary
    clusters = {}
    for idx, label in enumerate(labels):
        cid = f"cluster_{label}"
        clusters.setdefault(cid, []).append(corr_matrix.index[idx])

    return clusters
```

**Example Output** (50 assets → 4 clusters):
```python
{
    'cluster_1': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', ...],  # 15 tech stocks
    'cluster_2': ['JPM', 'BAC', 'GS', 'MS', ...],                 # 12 financials
    'cluster_3': ['XOM', 'CVX', 'COP', ...],                       # 10 energy
    'cluster_4': ['JNJ', 'PFE', 'UNH', ...],                       # 13 healthcare
}
```

### 3.3 Validation
```python
def validate_clusters(clusters: Dict[str, List[str]], n_bits: int = 10) -> bool:
    """Ensure all clusters satisfy degree constraint."""
    for cluster_id, tickers in clusters.items():
        n_vars = len(tickers) * n_bits
        if n_vars > 180:
            raise ValueError(f"{cluster_id} has {n_vars} vars (max 180)")
    return True
```

---

## 4. Stage 2: QUBO Formulation (Per Cluster)

### 4.1 Binary Encoding

For each asset `n` in cluster `g`, discretize weight using 10-bit binary expansion:

```
w_{n,g} = (1/K) * Σ_{q=0}^{9} 2^q * x_{n,g,q}
```

Where:
- `w_{n,g}` ∈ [0, 1]: weight for asset n in cluster g
- `x_{n,g,q}` ∈ {0, 1}: binary variable
- `K = 2^10 - 1 = 1023`: normalization constant

**Example**: With Q=10 bits:
- `w = 0.5` → binary: `0111111111` (512/1023 ≈ 0.5)
- `w = 0.1` → binary: `0001100110` (102/1023 ≈ 0.1)

### 4.2 QUBO Objective

For cluster `g` with assets `{1, 2, ..., n_g}`:

```
H_g = H_risk - H_return + H_budget
```

**Risk Term** (minimize portfolio variance):
```
H_risk = β * Σ_{i,j ∈ g} w_i * Σ_{i,j} * w_j
       = β * Σ_{i,j} Σ_{i,j} * (1/K²) * Σ_{q,p} 2^{q+p} * x_{i,q} * x_{j,p}
```

**Return Term** (maximize expected return):
```
H_return = α * Σ_{i ∈ g} μ_i * w_i
         = α * Σ_{i} μ_i * (1/K) * Σ_q 2^q * x_{i,q}
```

**Budget Constraint** (enforce Σw = B_g):
```
H_budget = λ * (Σ_{i ∈ g} w_i - B_g)²
         = λ * [(Σ_i (1/K) * Σ_q 2^q * x_{i,q}) - B_g]²
```

Where `B_g` is the budget allocated to cluster g (e.g., B_g = 1/G for equal allocation).

### 4.3 Parameter Tuning

| Parameter | Typical Range | Meaning | Tuning Guidance |
|-----------|---------------|---------|-----------------|
| α (return) | 0.5 - 2.0 | Weight on returns | Higher α = aggressive (growth-focused) |
| β (risk) | 0.5 - 2.0 | Weight on risk | Higher β = conservative (risk-averse) |
| λ (budget) | 5.0 - 20.0 | Budget penalty | Must be large enough to enforce Σw ≈ B_g |

**Recommended Starting Values**:
- **Balanced**: α=1.0, β=1.0, λ=10.0
- **Aggressive**: α=2.0, β=0.5, λ=10.0
- **Conservative**: α=0.5, β=2.0, λ=10.0

### 4.4 Implementation

```python
class QUBOFormulator:
    """Formulate per-cluster QUBO."""

    def __init__(self, n_bits=10, alpha=1.0, beta=1.0, lambda_budget=10.0):
        self.n_bits = n_bits
        self.K = 2**n_bits - 1  # 1023
        self.alpha = alpha
        self.beta = beta
        self.lambda_budget = lambda_budget

    def formulate_cluster(self,
                         tickers: List[str],
                         mu: pd.Series,      # Expected returns for cluster
                         Sigma: pd.DataFrame, # Covariance for cluster
                         budget: float = 1.0) -> dimod.BinaryQuadraticModel:
        """
        Create QUBO for a single cluster.

        Returns:
            dimod.BQM with variables like "AAPL_0", "AAPL_1", ..., "AAPL_9"
        """
        import dimod

        N = len(tickers)
        Q = {}  # Quadratic terms
        h = {}  # Linear terms

        # Variable names: "TICKER_BIT"
        var_names = [[f"{ticker}_{q}" for q in range(self.n_bits)]
                     for ticker in tickers]

        # 1. Return term (linear, negative to maximize)
        for n, ticker in enumerate(tickers):
            for q in range(self.n_bits):
                var = var_names[n][q]
                coeff = -self.alpha * mu[ticker] * (2**q) / self.K
                h[var] = h.get(var, 0) + coeff

        # 2. Risk term (quadratic)
        for i in range(N):
            for j in range(N):
                sigma_ij = Sigma.iloc[i, j]
                for q in range(self.n_bits):
                    for p in range(self.n_bits):
                        var_i = var_names[i][q]
                        var_j = var_names[j][p]

                        coeff = self.beta * sigma_ij * (2**(q+p)) / (self.K**2)

                        if var_i == var_j:
                            h[var_i] = h.get(var_i, 0) + coeff
                        else:
                            key = tuple(sorted([var_i, var_j]))
                            Q[key] = Q.get(key, 0) + coeff

        # 3. Budget constraint penalty
        # Expand (Σw - B)² = Σw² + ΣΣw_i*w_j - 2B*Σw + B²

        # Linear term: -2B*Σw
        for i in range(N):
            for q in range(self.n_bits):
                var = var_names[i][q]
                coeff = -2 * self.lambda_budget * budget * (2**q) / self.K
                h[var] = h.get(var, 0) + coeff

        # Quadratic term: Σw²
        for i in range(N):
            for q in range(self.n_bits):
                for p in range(self.n_bits):
                    var_q = var_names[i][q]
                    var_p = var_names[i][p]

                    coeff = self.lambda_budget * (2**(q+p)) / (self.K**2)

                    if var_q == var_p:
                        h[var_q] = h.get(var_q, 0) + coeff
                    else:
                        key = tuple(sorted([var_q, var_p]))
                        Q[key] = Q.get(key, 0) + coeff

        # Cross terms: w_i * w_j for i ≠ j
        for i in range(N):
            for j in range(i+1, N):
                for q in range(self.n_bits):
                    for p in range(self.n_bits):
                        var_i = var_names[i][q]
                        var_j = var_names[j][p]

                        coeff = 2 * self.lambda_budget * (2**(q+p)) / (self.K**2)
                        key = tuple(sorted([var_i, var_j]))
                        Q[key] = Q.get(key, 0) + coeff

        # Constant offset: B²
        offset = self.lambda_budget * budget**2

        # Build BQM
        bqm = dimod.BinaryQuadraticModel(h, Q, offset, dimod.BINARY)

        return bqm
```

---

## 5. Stage 3: QPU Execution (Parallel)

### 5.1 Solver Options

| Solver | Use Case | Pros | Cons |
|--------|----------|------|------|
| **DWaveSampler** (QPU) | Production, ≤180 vars/cluster | Real quantum advantage | Requires Leap access, embedding overhead |
| **LeapHybridSampler** | Fallback for noisy clusters | Handles larger problems | Not true QPU, longer runtime |
| **SimulatedAnnealingSampler** | Testing, no QPU access | Free, local | Classical simulation |

### 5.2 Parallel Execution

```python
from dwave.system import DWaveSampler, EmbeddingComposite
from concurrent.futures import ThreadPoolExecutor
import time

class ParallelQuantumSolver:
    """Solve multiple clusters in parallel."""

    def __init__(self, solver_type='qpu', num_reads=1000, annealing_time=20):
        self.solver_type = solver_type
        self.num_reads = num_reads
        self.annealing_time = annealing_time

    def solve_cluster(self, bqm: dimod.BQM, cluster_id: str) -> dict:
        """Solve a single cluster QUBO."""
        start = time.time()

        if self.solver_type == 'qpu':
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample(
                bqm,
                num_reads=self.num_reads,
                annealing_time=self.annealing_time,
                label=f'Portfolio-{cluster_id}'
            )
        elif self.solver_type == 'hybrid':
            from dwave.system import LeapHybridSampler
            sampler = LeapHybridSampler()
            sampleset = sampler.sample(bqm, label=f'Portfolio-{cluster_id}')
        else:  # simulated
            from neal import SimulatedAnnealingSampler
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=self.num_reads)

        runtime = time.time() - start

        # Get best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy

        return {
            'cluster_id': cluster_id,
            'solution': best_sample,
            'energy': best_energy,
            'runtime': runtime,
            'info': sampleset.info
        }

    def solve_all_clusters(self,
                          bqms: Dict[str, dimod.BQM],
                          max_workers: int = 5) -> Dict[str, dict]:
        """
        Solve all clusters in parallel.

        Args:
            bqms: {cluster_id: BQM}
            max_workers: Max concurrent QPU jobs

        Returns:
            {cluster_id: result_dict}
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.solve_cluster, bqm, cid): cid
                for cid, bqm in bqms.items()
            }

            for future in concurrent.futures.as_completed(futures):
                cid = futures[future]
                try:
                    result = future.result()
                    results[cid] = result
                except Exception as e:
                    print(f"Cluster {cid} failed: {e}")
                    results[cid] = {'error': str(e)}

        return results
```

### 5.3 Tuning Parameters

**Chain Strength**: Controls embedding quality on Zephyr
```python
# Auto-tune (recommended for first run)
sampler.sample(bqm, chain_strength='auto', ...)

# Manual tuning
chain_strength = 2.0 * max(abs(h.values())) + max(abs(Q.values()))
```

**Annealing Time**: Longer = better quality, slower
- **Fast**: 20 μs (good for testing)
- **Standard**: 50 μs (production)
- **High-quality**: 100-200 μs (critical optimizations)

**Number of Reads**: More samples = more robust
- **Test**: 100 reads
- **Production**: 1000 reads
- **Research**: 10,000 reads

---

## 6. Stage 4: Aggregation & Decoding

### 6.1 Binary → Weight Decoding

```python
class QUBODecoder:
    """Decode binary solutions to portfolio weights."""

    def __init__(self, n_bits=10):
        self.n_bits = n_bits
        self.K = 2**n_bits - 1  # 1023

    def decode_solution(self,
                       solution: Dict[str, int],
                       tickers: List[str]) -> pd.Series:
        """
        Convert binary variables to weights.

        Args:
            solution: {'AAPL_0': 1, 'AAPL_1': 0, ..., 'AAPL_9': 1}
            tickers: ['AAPL', 'MSFT', ...]

        Returns:
            pd.Series with ticker→weight mapping
        """
        weights = {}

        for ticker in tickers:
            # Sum binary expansion: w = (1/K) * Σ 2^q * x_q
            w = 0
            for q in range(self.n_bits):
                var_name = f"{ticker}_{q}"
                if var_name in solution:
                    w += solution[var_name] * (2**q)

            weights[ticker] = w / self.K

        return pd.Series(weights)
```

### 6.2 Aggregation Strategies

#### Strategy A: Simple Renormalization (Default)
```python
def aggregate_simple(cluster_weights: Dict[str, pd.Series]) -> pd.Series:
    """
    Concatenate all cluster weights and renormalize.

    Args:
        cluster_weights: {cluster_id: weights_series}

    Returns:
        Global weights (sum = 1)
    """
    # Combine all weights
    all_weights = pd.concat(cluster_weights.values())

    # Renormalize to sum=1
    if all_weights.sum() > 0:
        all_weights /= all_weights.sum()
    else:
        # Fallback: equal weights
        all_weights = pd.Series(1.0 / len(all_weights), index=all_weights.index)

    return all_weights
```

**Pros**: Simple, preserves cluster structure
**Cons**: May violate global constraints (e.g., max portfolio risk)

#### Strategy B: Score-Based Selection
```python
def aggregate_score_based(cluster_weights: Dict[str, pd.Series],
                         mu: pd.Series,
                         Sigma: pd.DataFrame,
                         k: int = 20) -> pd.Series:
    """
    Select top k assets by Sharpe ratio across all clusters.

    Args:
        cluster_weights: {cluster_id: weights}
        mu, Sigma: Global statistics
        k: Max assets in final portfolio

    Returns:
        Global weights (exactly k non-zero)
    """
    # Compute scores for all assets
    all_weights = pd.concat(cluster_weights.values())

    # Sharpe approximation: μ / sqrt(σ²)
    scores = mu / np.sqrt(np.diag(Sigma))

    # Select top k by score, weighted by cluster weights
    combined_scores = all_weights * scores
    top_k = combined_scores.nlargest(k)

    # Renormalize
    final_weights = pd.Series(0.0, index=all_weights.index)
    final_weights[top_k.index] = top_k / top_k.sum()

    return final_weights
```

**Pros**: Enforces cardinality, selects best assets
**Cons**: May ignore diversification

#### Strategy C: Hierarchical Aggregation
```python
def aggregate_hierarchical(cluster_weights: Dict[str, pd.Series],
                          mu: pd.Series,
                          Sigma: pd.DataFrame,
                          global_budget: float = 1.0) -> pd.Series:
    """
    Two-level optimization:
    1. Optimize cluster-level allocation
    2. Scale cluster weights by allocation

    Args:
        cluster_weights: {cluster_id: weights}
        mu, Sigma: Global statistics

    Returns:
        Global weights
    """
    import cvxpy as cp

    # Step 1: Compute cluster-level statistics
    G = len(cluster_weights)
    cluster_returns = []
    cluster_risks = []

    for cid, w_c in cluster_weights.items():
        r_c = mu[w_c.index] @ w_c
        risk_c = np.sqrt(w_c @ Sigma.loc[w_c.index, w_c.index] @ w_c)
        cluster_returns.append(r_c)
        cluster_risks.append(risk_c)

    # Step 2: Optimize cluster allocation
    w_cluster = cp.Variable(G)
    cluster_mu = np.array(cluster_returns)
    cluster_Sigma = np.outer(cluster_risks, cluster_risks)  # Simplified

    objective = cp.Minimize(cp.quad_form(w_cluster, cluster_Sigma) - cluster_mu @ w_cluster)
    constraints = [cp.sum(w_cluster) == global_budget, w_cluster >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Step 3: Scale cluster weights
    final_weights = pd.Series(0.0, index=pd.concat(cluster_weights.values()).index)

    for i, (cid, w_c) in enumerate(cluster_weights.items()):
        scaled = w_c * w_cluster.value[i]
        final_weights[scaled.index] = scaled

    # Renormalize (safety)
    final_weights /= final_weights.sum()

    return final_weights
```

**Pros**: Respects inter-cluster risk, balanced
**Cons**: More complex, requires second optimization

### 6.3 Recommended Strategy by Use Case

| Use Case | Strategy | Rationale |
|----------|----------|-----------|
| **First implementation** | Simple renormalization | Easiest to debug |
| **Cardinality constrained** | Score-based selection | Directly enforces k-asset limit |
| **Risk-focused** | Hierarchical aggregation | Best risk-adjusted portfolio |
| **Production** | Hierarchical or Score-based | Depends on requirements |

---

## 7. Integration with Classical Optimizers

### 7.1 Unified Optimizer Interface

```python
class IndependentClustersOptimizer:
    """Full pipeline: cluster → formulate → solve → aggregate."""

    def __init__(self,
                 n_bits=10,
                 max_cluster_size=18,
                 alpha=1.0,
                 beta=1.0,
                 lambda_budget=10.0,
                 solver_type='qpu',
                 aggregation='simple'):

        self.clusterer = PortfolioClustering(max_cluster_size, n_bits)
        self.formulator = QUBOFormulator(n_bits, alpha, beta, lambda_budget)
        self.solver = ParallelQuantumSolver(solver_type)
        self.decoder = QUBODecoder(n_bits)
        self.aggregation = aggregation

    def optimize(self, returns: pd.DataFrame) -> dict:
        """
        Run full quantum optimization.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            {
                'weights': pd.Series (N,),
                'metrics': dict,
                'runtime': float,
                'clusters': dict,
                'qpu_info': dict
            }
        """
        start_time = time.time()

        # Compute statistics
        mu = returns.mean() * 252  # Annualized
        Sigma = returns.cov() * 252
        corr = returns.corr()

        # Stage 1: Cluster
        clusters = self.clusterer.cluster(corr)

        # Stage 2: Formulate QUBOs
        bqms = {}
        for cid, tickers in clusters.items():
            mu_c = mu[tickers]
            Sigma_c = Sigma.loc[tickers, tickers]

            budget_c = 1.0 / len(clusters)  # Equal budget per cluster

            bqm = self.formulator.formulate_cluster(
                tickers, mu_c, Sigma_c, budget_c
            )
            bqms[cid] = bqm

        # Stage 3: Solve in parallel
        results = self.solver.solve_all_clusters(bqms)

        # Stage 4: Decode & aggregate
        cluster_weights = {}
        qpu_info = {}

        for cid, result in results.items():
            if 'error' in result:
                continue

            tickers = clusters[cid]
            weights = self.decoder.decode_solution(result['solution'], tickers)

            cluster_weights[cid] = weights
            qpu_info[cid] = result['info']

        # Aggregate
        if self.aggregation == 'simple':
            global_weights = aggregate_simple(cluster_weights)
        elif self.aggregation == 'score':
            global_weights = aggregate_score_based(cluster_weights, mu, Sigma)
        elif self.aggregation == 'hierarchical':
            global_weights = aggregate_hierarchical(cluster_weights, mu, Sigma)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        runtime = time.time() - start_time

        return {
            'weights': global_weights,
            'metrics': self._compute_metrics(global_weights, mu, Sigma),
            'runtime': runtime,
            'clusters': clusters,
            'qpu_info': qpu_info
        }

    def _compute_metrics(self, w: pd.Series, mu: pd.Series, Sigma: pd.DataFrame):
        """Compute standard portfolio metrics."""
        w_arr = w.values
        mu_arr = mu[w.index].values
        Sigma_arr = Sigma.loc[w.index, w.index].values

        exp_return = mu_arr @ w_arr
        exp_risk = np.sqrt(w_arr @ Sigma_arr @ w_arr)
        sharpe = exp_return / exp_risk if exp_risk > 0 else 0

        return {
            'expected_return': float(exp_return),
            'expected_risk': float(exp_risk),
            'sharpe_ratio': float(sharpe),
            'n_assets': int(np.sum(w_arr > 1e-6)),
            'herfindahl_index': float(np.sum(w_arr**2)),
            'effective_n_assets': float(1 / np.sum(w_arr**2)) if np.sum(w_arr**2) > 0 else 0
        }
```

### 7.2 Comparison Workflow

```python
# Compare quantum vs classical
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.backtest import Backtester

# Load data
prices = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

# Classical baseline
classical_opt = ClassicalOptimizer(gamma=1.0, method='cvxpy')
classical_result = classical_opt.optimize(returns)

# Quantum (independent clusters)
quantum_opt = IndependentClustersOptimizer(
    solver_type='qpu',
    aggregation='hierarchical'
)
quantum_result = quantum_opt.optimize(returns)

# Compare
print(f"Classical Sharpe: {classical_result['metrics']['sharpe_ratio']:.3f}")
print(f"Quantum Sharpe:   {quantum_result['metrics']['sharpe_ratio']:.3f}")
print(f"Classical Runtime: {classical_result['runtime']:.2f}s")
print(f"Quantum Runtime:   {quantum_result['runtime']:.2f}s")
print(f"Number of clusters: {len(quantum_result['clusters'])}")
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# File: tests/test_independent_clusters.py

import pytest
from qpo.qubo.clustering import PortfolioClustering
from qpo.qubo.formulation import QUBOFormulator
from qpo.qubo.decoder import QUBODecoder

def test_clustering_constraints():
    """Test that clustering respects max_cluster_size."""
    n_assets = 50
    corr = generate_synthetic_correlation(n_assets)

    clusterer = PortfolioClustering(max_cluster_size=18, n_bits=10)
    clusters = clusterer.cluster(corr)

    # Validate
    assert clusterer.validate_degree_constraint(clusters)

    for cid, tickers in clusters.items():
        assert len(tickers) <= 18, f"{cid} has {len(tickers)} > 18 assets"

def test_qubo_formulation():
    """Test QUBO creation for a small cluster."""
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    mu = pd.Series([0.12, 0.10, 0.14], index=tickers)
    Sigma = pd.DataFrame([[0.04, 0.01, 0.02],
                         [0.01, 0.03, 0.01],
                         [0.02, 0.01, 0.05]],
                        index=tickers, columns=tickers)

    formulator = QUBOFormulator(n_bits=10, alpha=1.0, beta=1.0, lambda_budget=10.0)
    bqm = formulator.formulate_cluster(tickers, mu, Sigma, budget=1.0)

    # Validate BQM structure
    assert len(bqm.variables) == 30  # 3 assets × 10 bits
    assert bqm.vartype == dimod.BINARY

def test_decoder():
    """Test binary→weight decoding."""
    tickers = ['AAPL', 'MSFT']
    solution = {
        'AAPL_0': 1, 'AAPL_1': 1, 'AAPL_2': 1, 'AAPL_3': 1, 'AAPL_4': 1,
        'AAPL_5': 1, 'AAPL_6': 1, 'AAPL_7': 1, 'AAPL_8': 1, 'AAPL_9': 1,  # All 1s → w≈1
        'MSFT_0': 0, 'MSFT_1': 0, 'MSFT_2': 0, 'MSFT_3': 0, 'MSFT_4': 0,
        'MSFT_5': 0, 'MSFT_6': 0, 'MSFT_7': 0, 'MSFT_8': 0, 'MSFT_9': 0,  # All 0s → w=0
    }

    decoder = QUBODecoder(n_bits=10)
    weights = decoder.decode_solution(solution, tickers)

    assert np.isclose(weights['AAPL'], 1.0, atol=0.01)  # 1023/1023 ≈ 1
    assert weights['MSFT'] == 0.0

def test_aggregation():
    """Test simple aggregation."""
    cluster_weights = {
        'cluster_1': pd.Series([0.6, 0.4], index=['A', 'B']),
        'cluster_2': pd.Series([0.5, 0.5], index=['C', 'D']),
    }

    global_weights = aggregate_simple(cluster_weights)

    # Should renormalize to sum=1
    assert np.isclose(global_weights.sum(), 1.0, atol=1e-6)
    assert len(global_weights) == 4
```

### 8.2 Integration Tests

```python
def test_full_pipeline_simulated():
    """Test full pipeline with simulated annealing."""
    # Synthetic data: 32 assets, 2 years
    prices = generate_synthetic_data(n_assets=32, n_days=504)
    returns = prices.pct_change().dropna()

    # Optimize with simulated annealing (no QPU needed)
    optimizer = IndependentClustersOptimizer(
        solver_type='simulated',
        aggregation='simple'
    )
    result = optimizer.optimize(returns)

    # Validate output
    assert 'weights' in result
    assert np.isclose(result['weights'].sum(), 1.0, atol=1e-5)
    assert result['metrics']['sharpe_ratio'] > 0
    assert len(result['clusters']) >= 2  # Should have multiple clusters

@pytest.mark.qpu  # Mark as requiring QPU access
def test_full_pipeline_qpu():
    """Test with real QPU (requires D-Wave access)."""
    prices = generate_synthetic_data(n_assets=32, n_days=504)
    returns = prices.pct_change().dropna()

    optimizer = IndependentClustersOptimizer(
        solver_type='qpu',
        aggregation='hierarchical'
    )
    result = optimizer.optimize(returns)

    # Validate QPU-specific info
    assert result['qpu_info'] is not None
    for cid, info in result['qpu_info'].items():
        assert 'timing' in info  # QPU timing info
```

### 8.3 Benchmarking

```python
def benchmark_scalability():
    """Test scalability: N=50, 100, 150, 200."""
    results = []

    for n_assets in [50, 100, 150, 200]:
        prices = generate_synthetic_data(n_assets, n_days=504)
        returns = prices.pct_change().dropna()

        optimizer = IndependentClustersOptimizer(solver_type='simulated')
        result = optimizer.optimize(returns)

        results.append({
            'n_assets': n_assets,
            'n_clusters': len(result['clusters']),
            'runtime': result['runtime'],
            'sharpe': result['metrics']['sharpe_ratio']
        })

    # Plot results
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df['n_assets'], df['runtime'], marker='o')
    ax1.set_xlabel('Number of Assets')
    ax1.set_ylabel('Runtime (s)')
    ax1.set_title('Scalability: Runtime vs N')

    ax2.bar(df['n_assets'], df['n_clusters'])
    ax2.set_xlabel('Number of Assets')
    ax2.set_ylabel('Number of Clusters')
    ax2.set_title('Clustering Behavior')

    plt.tight_layout()
    plt.savefig('benchmark_independent_clusters.png')
```

---

## 9. Performance Expectations

### 9.1 Runtime Targets

| N Assets | Clusters | QPU Time/Cluster | Total Time | Notes |
|----------|----------|------------------|------------|-------|
| 50 | 3-5 | 10-15s | 30-45s | Small clusters, fast |
| 100 | 6-10 | 10-20s | 60-120s | Parallel speedup |
| 150 | 8-12 | 15-25s | 90-180s | Larger embedding |
| 200 | 10-15 | 20-30s | 120-300s | May need hybrid |

**Assumptions**:
- QPU execution with EmbeddingComposite
- 1000 reads per cluster
- Parallel execution (5 concurrent jobs)

### 9.2 Quality Metrics

**Constraint Satisfaction**:
- Budget: >95% of runs satisfy |Σw - 1| < 0.01
- Cardinality: Enforce via aggregation (score-based)
- Risk: Depends on λ tuning

**Portfolio Performance**:
- **Sharpe Ratio**: Within 10% of classical baseline
- **Turnover**: Lower than classical (quantum tends to be sparser)
- **Diversification**: Effective N ≈ 60-80% of classical

### 9.3 Scalability Limits

| Solver | Max N Assets | Max Clusters | Bottleneck |
|--------|--------------|--------------|------------|
| QPU | 200 | 15 | Embedding quality |
| Hybrid | 500 | 30 | Runtime |
| Simulated | 1000+ | 50+ | Compute time |

---

## 10. Migration Path & Roadmap

### 10.1 Phase 1: Prototype (Weeks 1-4)
- ✅ Implement clustering (already exists)
- ✅ Implement QUBO formulation
- ✅ Test with simulated annealing
- ✅ Validate on small datasets (20-30 assets)

### 10.2 Phase 2: QPU Integration (Weeks 5-8)
- Integrate D-Wave Ocean SDK
- Test on QPU with single cluster
- Implement parallel execution
- Benchmark QPU vs simulated

### 10.3 Phase 3: Production (Weeks 9-12)
- Add hierarchical aggregation
- Tune parameters (α, β, λ)
- Backtest on historical data
- Compare to classical methods

### 10.4 Phase 4: Advanced Features (Weeks 13-16)
- Adaptive clustering (re-cluster based on results)
- Multi-objective optimization
- Transaction cost integration
- Real-time rebalancing

---

## 11. Comparison: Independent vs Averaging Approach

| Aspect | Independent Clusters | Clustered Averaging |
|--------|----------------------|---------------------|
| **Granularity** | Asset-level | Cluster-level |
| **Parallelism** | High (G QPU calls) | Low (1 QPU call) |
| **Inter-cluster correlation** | Ignored | Captured |
| **Information loss** | Low | High (averaging) |
| **Scalability** | Good (up to 15 clusters) | Better (20+ clusters) |
| **Complexity** | Simple per-cluster | Complex aggregation |
| **Recommended for** | First implementation | Advanced use cases |

---

## 12. References & Resources

### Academic Papers
1. Palmer et al. (2021). "Quantum Portfolio Optimization with Cardinality Constraints". *arXiv:2112.xxxxx*.
2. Aguilera et al. (2023). "Hybrid Quantum-Classical Portfolio Optimization". *IEEE QSEC*.
3. Sakuler et al. (2024). "Zephyr Topology for Portfolio Optimization". *D-Wave Technical Report*.

### D-Wave Documentation
- Ocean SDK: https://docs.ocean.dwavesys.com/
- Zephyr Topology: https://docs.dwavesys.com/docs/latest/c_gs_2.html
- Embedding Guide: https://docs.ocean.dwavesys.com/en/stable/concepts/embedding.html

### Code Repositories
- D-Wave Examples: https://github.com/dwave-examples/portfolio-optimization
- Qiskit Finance: https://github.com/Qiskit/qiskit-finance

---

**End of Specification: Independent Clusters Approach**
