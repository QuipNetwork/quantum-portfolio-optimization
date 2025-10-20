# QPO Architecture Overview

## Classical Optimizer Structure

The `ClassicalOptimizer` is **one algorithm** with **two solver backends**:

```
┌─────────────────────────────────────────────────────────┐
│           ClassicalOptimizer (Mean-Variance)            │
│                                                         │
│  Solves:  minimize  γ·risk - return                    │
│           subject to: Σw = 1, w ≥ 0, ||w||₀ ≤ k        │
└─────────────────────────────────────────────────────────┘
                          │
                          │ method parameter
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
    ┌──────────────┐           ┌──────────────┐
    │    CVXPY     │           │      GA      │
    │              │           │              │
    │ • Convex QP  │           │ • Heuristic  │
    │ • Fast       │           │ • Slow       │
    │ • Exact      │           │ • Approx     │
    │ • No k limit │           │ • k support  │
    └──────────────┘           └──────────────┘
```

### When to Use Each

**CVXPY** (Continuous Relaxation):
```python
optimizer = ClassicalOptimizer(gamma=1.0, method='cvxpy')
# Best for: Fast, exact solutions when you don't need cardinality limits
# Limitation: May select ALL assets with tiny weights
```

**GA** (Discrete Optimization):
```python
optimizer = ClassicalOptimizer(gamma=1.0, k=20, method='ga')
# Best for: When you MUST limit to exactly k assets
# Limitation: Slow, approximate, may not find global optimum
```

## Comparison Table

| Feature | CVXPY | GA |
|---------|-------|-----|
| **Speed (16 assets)** | 0.003s | 5.4s |
| **Speed (64 assets)** | 0.011s | ~60s (too slow) |
| **Cardinality constraint** | ❌ No | ✅ Yes |
| **Optimality** | ✅ Global | ⚠️ Local |
| **Deterministic** | ✅ Yes | ❌ No (stochastic) |
| **Recommended for** | Production | Research/Special cases |

## Algorithm Details

### CVXPY Backend

**Problem formulation:**
```
minimize    γ·w^T Σ w - μ^T w
subject to  Σw_i = 1
            w_i ≥ 0
```

**Solver:** ECOS (Embedded Conic Solver)
- Interior point method
- Polynomial time: O(N³) for N assets
- Provably optimal solution

**Why it's fast:**
- Exploits convex structure
- Direct path to optimum
- No random search needed

### GA Backend

**Problem formulation:**
```
minimize    γ·w^T Σ w - μ^T w + penalty(||w||₀ > k)
subject to  Σw_i = 1
            w_i ≥ 0
```

**Solver:** Differential Evolution (SciPy)
- Population size: 15·N candidates
- Iterations: 1000 max
- Evolutionary operators: mutation, crossover, selection

**Why it's slow:**
- Explores combinatorial space (2^N possible asset selections)
- No gradient information (black-box optimization)
- Stochastic: needs many evaluations for convergence
- Constraint penalty requires tuning

**Post-processing:**
We enforce cardinality by selecting top-k assets after optimization:
```python
top_k_indices = np.argsort(w_opt)[-k:]  # Keep only top k
```

## Both Are "Classical"

Both CVXPY and GA are **classical algorithms** (run on regular CPUs) as opposed to:

- **Quantum algorithms** (run on QPUs like D-Wave)
  - QUBO formulation
  - Quantum annealing
  - Clustering-based decomposition

## Usage Recommendation

**For most use cases:** Use CVXPY
```python
# Fast, exact, production-ready
optimizer = ClassicalOptimizer(gamma=1.0, method='cvxpy')
```

**Only use GA if:**
1. You have a hard cardinality constraint (regulatory/operational limit)
2. Portfolio is small (<20 assets)
3. You can tolerate 100-1000x slower runtime
4. Approximation is acceptable

**Alternative for cardinality:**
Instead of GA, consider post-processing CVXPY results:
```python
# Get CVXPY solution
weights = optimizer.optimize(returns)['weights']

# Keep only top k assets
top_k = weights.nlargest(k)
top_k /= top_k.sum()  # Renormalize
```

This is faster than GA and often gives similar results!

## Future: Mixed-Integer Programming

A better alternative to GA for cardinality constraints:
```python
# Not yet implemented
optimizer = ClassicalOptimizer(gamma=1.0, k=20, method='mip')
# Would use MILP solvers like Gurobi/CPLEX
# Exact solution, faster than GA, slower than CVXPY
```
