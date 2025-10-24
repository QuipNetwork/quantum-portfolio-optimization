# Fixed Template Embeddings for QUBO-to-Hardware Mapping

This document explains our fixed template embedding approach for the quantum portfolio optimization pipeline: what it solves, how it works, where it is implemented, and how to use/tune it. It connects the implementation to D-Wave topology constraints (Zephyr/Pegasus) and the Independent Clusters Approach.

**Strategy**: As of v1.0, we use a template-only approach for QPU execution. Pre-computed embedding templates are required and loaded at runtime with deterministic variable relabeling. No on-the-fly embedding is performed; if a template is missing, the solver crashes with a clear error message.

Key code files:
- qpo/qubo/fixed_embeddings.py (template loading and FixedEmbeddingComposite creation)
- qpo/qubo/discrete_levels.py (discrete levels formulation + template embedding utilities)
- tools/find_optimal_template.py (search for a maximal, reusable multi-cluster template)
- tools/generate_portfolio_embedding_template.py (generate templates for both discrete_levels and qubo_bits formulators)
- qpo/qubo/solver.py (template selection pipeline: template-fixed → fixed-cache → lazy-fixed fallback)
- Embedding artifacts live in embeddings/templates/

---

## QUBO Formulation Details

### Unified Multi-Cluster Formulation

The quantum optimizer uses a unified single-pass QUBO formulation with cluster representatives for efficient inter-cluster optimization:

**Key Components**:
1. **Intra-cluster optimization**: Each cluster's assets optimize with return/risk objectives
2. **Cluster representatives**: Meta-variables (CLUSTER_{id}_{level}) that represent cluster-level allocations
3. **Inter-cluster correlations**: Representatives are coupled based on average cross-cluster correlations
4. **Meta-cluster budget constraint**: Budget constraint applied to representatives only (not individual assets)

**QUBO Terms** (for unified formulation):
```
E = Σ_clusters E_intra-cluster(assets)           # Return/risk within clusters
  + E_thermometer(all_variables)                  # Thermometer encoding constraints
  + E_inter-cluster(representatives)              # Cross-cluster risk coupling
  + E_asset-to-rep(assets, representatives)       # Cluster membership coupling
  + λ * (Σ_representatives - 1)²                  # Budget constraint on meta-cluster
```

**Budget Constraint Design**:
- Applied to cluster representatives (CLUSTER_* variables) only, not individual assets
- Efficiency: O(C²) couplings where C = number of clusters (typically 5-15)
- Contrast: Applying to all assets would be O(N²) where N = number of assets (100+)
- Graph sparsity: Significantly reduces QUBO degree, improving QPU embeddability
- Semantics: Representatives determine inter-cluster capital allocation; assets optimize relative weights within clusters

**Budget Constraint Expansion**:
```
λ * (Σw_cluster - 1)² = λ * (Σw_cluster)² - 2λ * Σw_cluster + λ
                        \_______________/   \_______________/   \_/
                         Quadratic terms     Linear terms    Constant offset
```

**Critical Implementation Detail**: The constant offset `+λ` is essential:
- Without it: E(all zeros) = 0 → solver has no incentive to allocate capital
- With it: E(all zeros) = +λ → all-zero solutions are penalized, forcing allocation
- When budget met (Σw=1): E_budget = λ - 2λ + λ = 0 (no penalty)

**Why This Architecture**:
1. **Scalability**: Reducing budget constraint from O(N²) to O(C²) allows larger portfolios
2. **Hierarchical optimization**: Clusters optimize internal allocation; representatives optimize inter-cluster allocation
3. **Graph structure**: Matches pre-computed template topologies (designed for C clusters with representatives)
4. **QPU embeddability**: Sparser graphs embed more reliably with better chain lengths

See `qpo/qubo/discrete_levels.py:273-470` (formulate_with_cluster_representatives) for implementation details.

---

## Overview

Problem: D‑Wave minor‑embedding (mapping logical QUBO graph to hardware graph) is expensive (30–60s per structure) and variable. Repeated backtests re-solve similar QUBO structures (same cluster size and bit encoding) across rebalances, wasting time and introducing variance.

Solution: Pre-compute and reuse fixed "template" embeddings with deterministic variable relabeling:
- Generate templates for common problem shapes with dummy variable names (e.g., ASSET_0_0, ASSET_1_3)
- At runtime, load the matching template based on formulator type, cluster count, assets per cluster, and levels/bits
- Deterministically relabel template variables to actual tickers based on cluster ordering
- Use FixedEmbeddingComposite with the relabeled template for zero embedding overhead

Benefits:
- **Eliminate embedding overhead**: First run and subsequent runs use the same pre-computed embedding (instant)
- **Deterministic placement**: Fixed qubit chains produce stable chain lengths and eliminate run-to-run variance from embedding randomness
- **Better benchmarking**: Consistent physical mapping allows fair performance comparisons
- **Scalability**: Full-portfolio templates allow larger problems without composition/offset complexity

---

## Template Selection Strategy (Runtime)

File: qpo/qubo/solver.py (QuantumSolver._solve_qpu)

The solver uses a three-tier fallback strategy for QPU execution:

1. **Template-fixed** (preferred): Load pre-computed template from `embeddings/templates/`
   - Fast: Zero embedding overhead
   - Deterministic: Same qubit chains every run
   - Requires: Matching template for problem shape (C, A, L/B) and topology

2. **Fixed-cache** (fallback): Persistent structure-based cache from previous runs
   - Fast: Reuses computed embeddings across sessions
   - Non-deterministic: First run computes embedding (30-60s), subsequent runs instant
   - File: `embeddings/{topology}_{structure_hash}.pkl`

3. **Lazy-fixed** (last resort): Session-based cache (Ocean default)
   - Slow on first run: Computes embedding per session
   - Fast on repeat: Caches within Python session only
   - Not persistent: Re-computes on restart

**Environment variable**: Set `QPO_USE_FIXED_TEMPLATES=0` to disable template loading (forces fallback to fixed-cache or lazy-fixed)

---

## Technical Details

### A) Template generation and loading
- Generation: tools/generate_portfolio_embedding_template.py
- Loading: qpo/qubo/fixed_embeddings.py (load_template_embedding)
- Approach: Generate templates with dummy variable names (ASSET_{cluster}_{asset}_{bit}), save with metadata (formulator, shape, topology, sparsification_signature), load at runtime and relabel to actual tickers.

Template naming convention:
```
portfolio_{C}c_{A}a_{L}l_{topo}.json     # discrete_levels formulator
portfolio_{C}c_{A}a_{B}b_{topo}.json     # qubo_bits formulator
cluster_{A}a_{L}l_{topo}.json            # per-cluster templates
```

Where: C = clusters, A = assets/cluster, L = levels, B = bits, topo = topology (e.g., zephyr2)

Template metadata (JSON):
```json
{
  "version": "1.0.0",
  "formulator": "discrete_levels" | "qubo_bits",
  "topology": {"name": "Advantage2_system1.6", "family": "zephyr2"},
  "shape": {"n_clusters": 13, "assets_per_cluster": 9, "n_levels": 10},
  "sparsification_signature": "3b6e5c...",
  "created_at": "2025-10-24T12:00:00Z",
  "embedding": {...}
}
```

### B) Structure-based caching (persistent cache, fallback tier 2)
- File: qpo/qubo/fixed_embeddings.py (EmbeddingCache)
- Approach: Hash the logical BQM graph (variables, edge set) → use it as cache key per topology. Cache embeddings to disk for reuse across sessions.

### C) Lazy in‑memory caching (fallback tier 3)
- File: qpo/qubo/solver.py
- Uses LazyFixedEmbeddingComposite for automatic per‑session caching by problem structure (Ocean SDK default)
- **Note**: Only used when templates and persistent cache unavailable

### D) Variable relabeling and deterministic mapping
- File: qpo/qubo/discrete_levels.py (CachedEmbeddingManager.get_full_portfolio_embedding)
- Approach: Templates use generic names (ASSET_0_0, ASSET_1_3, etc.). At runtime:
  1. Load template matching problem shape (C clusters, A assets/cluster, L levels)
  2. Validate configuration (n_levels, cluster layout)
  3. Build deterministic mapping: asset_index → ticker based on cluster order
  4. Relabel all template variables: ASSET_{i}_{q} → {ticker}_{q}
  5. Return relabeled embedding ready for FixedEmbeddingComposite

Example:
```python
# Template has: ASSET_0_0, ASSET_0_1, ..., ASSET_1_0, ...
# Cluster 0: ['AAPL', 'MSFT'], Cluster 1: ['GOOGL', 'AMZN']
# Relabeled: AAPL_0, AAPL_1, MSFT_0, MSFT_1, GOOGL_0, ...
```

### E) Where embeddings are applied in the solve path
- Pipeline: Clustering → QUBO Formulation → **Template Loading** → Solving → Decoding
- File: qpo/qubo/solver.py (QuantumSolver._solve_qpu)
- Process:
  1. Load/create base DWaveSampler
  2. Attempt template load (if use_fixed_templates=True)
  3. Create FixedEmbeddingComposite with template or fall back to cached/lazy embedding
  4. Sample on QPU with fixed embedding
  5. Log embedding_strategy, template_file, and chain_stats

---

## Integration in the pipeline

- Clustering → QUBO Formulation → Embedding → Solving → Decoding
  1) Clustering yields clusters ≤ max_cluster_size
  2) For each cluster, formulate BQM (either standard n_bits or discrete levels)
  3) Embedding:
     - Default: LazyFixedEmbeddingComposite (per‑session cache)
     - Persistent: create_fixed_embedding_sampler (fixed embedding from cache)
     - Templates: load template for cluster size, remap ASSET_i_q → tickers, optionally combine/offset for multi‑cluster
  4) Solve on QPU/Hybrid/Simulated
  5) Decode binary solution to weights (qpo/qubo/decoder.py or discrete_levels decode)

---

## Benefits

- Speed: Avoid repeated minorminer runs per rebalance → large wall‑clock savings.
- Consistency: Fixed/templated qubit chains produce stable chain lengths and reduce run‑to‑run variance.
- Scalability: Combining templates (offsetting chains or using full‑portfolio template) allows larger effective problems under Zephyr constraints.
- Topology‑aware: Templates and caches are tied to topology (e.g., Advantage2 Zephyr) and chip capability.

---

## Limitations and trade‑offs

- Topology- and chip-specific: Templates depend on hardware graph; changing solver (Pegasus ↔ Zephyr) invalidates templates.
- Structure dependency: Cache keys are derived from graph structure; changing discretization (levels/bits), sparsification, or budget coupling alters the graph and bypasses the cache.
- Size matching: Template embeddings expect exact asset count (n_assets) and n_levels; otherwise remapping/offsetting or a different template is required.
- Chain quality: A cached/template embedding may not be optimal for every instance; some problems may benefit from re‑embedding (different chain strengths/regions).
- Multi‑cluster placement: Offsetting/combining requires careful bookkeeping to avoid chain overlap; the full‑portfolio template avoids this but is larger to compute.

---

## Configuration knobs

**Optimizer level:**
- `use_fixed_templates: bool = True`: Enable/disable template loading (default: True)
- `eliminate_intra_asset_couplings: bool = False`: Remove off-diagonal intra-asset couplings for sparser graphs
- `bit_aligned_risk_couplings: bool = False`: Only couple same-index bits for risk term (approximation)
- Environment variable: `QPO_USE_FIXED_TEMPLATES=0/1` (overrides optimizer setting)

**Template generation:**
- `--formulator`: Choose 'discrete_levels' or 'qubo_bits' formulator
- `--num-clusters`: Number of clusters (discrete_levels adds meta-cluster automatically)
- `--cluster-size`: Assets per cluster
- `--n-levels` / `--n-bits`: Discretization parameter (must match optimizer)
- `--solver` / `--topology`: Live QPU or saved topology file
- `--timeout`, `--tries`, `--chainlength-patience`: Minorminer parameters

**Template location:**
- Directory: `embeddings/templates/` (default)
- Naming: `portfolio_{C}c_{A}a_{L/B}{l/b}_{topo}.json`
- Override cache dir: `CachedEmbeddingManager(cache_dir=...)`

---

## Example usages

### 1) Generate a template for qubo_bits formulator (13 clusters, 9 assets, 10 bits):
```bash
python tools/generate_portfolio_embedding_template.py \
  --solver Advantage2_system1.6 \
  --formulator qubo_bits \
  --num-clusters 13 \
  --cluster-size 9 \
  --n-bits 10 \
  --output embeddings/templates/portfolio_13c_9a_10b.json
```

### 2) Generate a template for discrete_levels formulator (12 clusters, 10 assets, 10 levels):
```bash
python tools/generate_portfolio_embedding_template.py \
  --solver Advantage2_system1.6 \
  --formulator discrete_levels \
  --num-clusters 12 \
  --cluster-size 10 \
  --n-levels 10 \
  --output embeddings/templates/portfolio_12c_10a_10l.json
```

### 3) Use fixed templates in optimizer:
```python
from qpo.optimizers.quantum import IndependentClustersOptimizer

optimizer = IndependentClustersOptimizer(
    solver_type='qpu',
    n_bits=10,
    max_cluster_size=24,
    use_fixed_templates=True,  # Default: True
    eliminate_intra_asset_couplings=False,
    bit_aligned_risk_couplings=False
)

result = optimizer.optimize(returns)
```

### 4) Load template embedding directly (advanced):
```python
from qpo.qubo.fixed_embeddings import load_template_embedding

embedding, metadata = load_template_embedding(
    formulator='qubo_bits',
    shape={'n_clusters': 13, 'assets_per_cluster': 9, 'n_bits': 10},
    topology='zephyr'
)

print(f"Loaded template: {metadata['template_file']}")
print(f"QPU utilization: {metadata['statistics']['qpu_utilization_pct']:.1f}%")
```

### 5) Disable templates (use fallback caching):
```python
# Option 1: Environment variable
import os
os.environ['QPO_USE_FIXED_TEMPLATES'] = '0'

# Option 2: Optimizer parameter
optimizer = IndependentClustersOptimizer(
    solver_type='qpu',
    use_fixed_templates=False  # Forces fixed-cache or lazy-fixed fallback
)
```

---

## Performance notes

- First run embedding time (per unique structure) can be tens of seconds; subsequent runs with cached or template embeddings are effectively instant for the embedding step.
- Chain statistics (avg/max length) from templates help anticipate chain_strength choices and error rates.
- Discrete levels formulation (bit-aligned couplings) reduces graph degree, improving embeddability and chain lengths before embedding.

---

## Future work

- Automatic selection: Choose between session cache, fixed cache, and template based on problem structure and cluster sizes.
- Template library expansion: Cover additional n_assets and n_levels commonly encountered; maintain per-topology catalogs.
- Pegasus support: Generate and manage parallel templates for Pegasus/other QPUs.
- Multi-cluster placement tooling: Robust qubit offset/tiling utilities and collision detection.
- Validation tooling: Quick smoke tests to verify a template against current chip calibration (chain break metrics, yields).

---

## Pointers to code and artifacts

**Template generation:**
- tools/generate_portfolio_embedding_template.py: Template generator for both discrete_levels and qubo_bits formulators
- tools/find_optimal_template.py: Search for maximal multi-cluster templates with tuned minorminer parameters

**Template loading and usage:**
- qpo/qubo/fixed_embeddings.py:173-261 (load_template_embedding), 264-277 (create_fixed_embedding_sampler_from_template)
- qpo/qubo/solver.py:110-209 (QuantumSolver._solve_qpu with template selection pipeline)
- qpo/qubo/discrete_levels.py:593-693 (CachedEmbeddingManager.get_full_portfolio_embedding with relabeling)

**Optimizer integration:**
- qpo/optimizers/quantum.py:61-86 (IndependentClustersOptimizer.__init__ with use_fixed_templates, sparsification toggles)
- qpo/optimizers/discrete_levels.py:39-64 (DiscreteLevelsOptimizer with template support)

**Artifacts:**
- Templates: `embeddings/templates/portfolio_*.json`
- Persistent cache: `embeddings/{topology}_{structure_hash}.pkl`

