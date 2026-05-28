# QUBO-Based Portfolio Selection with Budget, Duration, and Cardinality Constraints

This document describes an implementation of QUBO matrix construction for portfolio optimization, selecting assets to maximize scores while respecting budget, duration, and cardinality constraints.

# Quick Start and Overview

## Installation

### Core dependencies (required for QUBO/Slack/CQM/NL paths)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy dimod dwave-neal dwave-system dwave-optimization scipy pytest
```

- `dimod` - D-Wave's binary quadratic model (BQM) library
- `dwave-neal` - Simulated annealing sampler for QUBO/Ising problems
- `dwave-system` - D-Wave QPU access (only needed for live hardware runs)
- `dwave-optimization` - Non-linear (NL) model used by `nl_portfolio.py`

### Optional solver dependencies

The benchmark wires up several optional solver integrations. Each is
imported lazily, so the rest of the suite runs cleanly without any of
them. To populate the corresponding benchmark rows, install the ones
you want:

```bash
# QHDOPT (CPU)
pip install qhdopt                # Python 3.10-3.12
# On Python 3.13, see linux-instructions.md §3 for the --no-deps workaround.

# OpenPhiSolve (from source, CPU or GPU)
git clone https://github.com/Artephi-Computing/OpenPhiSolve.git
pip install -e ./OpenPhiSolve

# Pasqal stack (CPU emulators only)
pip install qubo-solver maximum-independent-set pulser pulser-simulation qutip

# NVIDIA cuOpt (Linux + CUDA 12.x)
pip install --extra-index-url=https://pypi.nvidia.com \
    "cuopt-cu12" "nvidia-nvjitlink-cu12" "rapids-logger"
```

Full instructions, troubleshooting, and Python-version notes are in
`linux-instructions.md` at the repo root.

## Running the Code

```bash
# Run demos
python simple_portfolio_qubo.py
python slack_portfolio_qubo.py

# Run tests
python -m pytest . -v

# Run benchmark
python benchmark_simple_qubo.py --problem-set simple
python benchmark_simple_qubo.py --problem-set random --num-assets 12 --num-trials 5 --seed 42   
```

## Files

All files are located in `experiments/simple/`:

Core implementations:
- `simple_portfolio_qubo.py` - Soft-constraint QUBO (penalty matrix)
- `slack_portfolio_qubo.py` - Slack-variable QUBO (true inequalities)
- `cqm_portfolio.py` - D-Wave ConstrainedQuadraticModel
- `nl_portfolio.py` - D-Wave non-linear (NL) model for Stride hybrid

Optional solver integrations (lazy imports; install per "Optional
solver dependencies" above):
- `qhd_portfolio.py` - QHDOPT (continuous relaxation of QUBO)
- `cuopt_portfolio.py` - NVIDIA cuOpt (GPU MILP + QP)
- `phi_portfolio.py` - OpenPhiSolve (QIHD + PDQP refinement)
- `pasqal_portfolio.py` - Pasqal neutral-atom stack (qubosolver,
  slack-QUBO, MIS, raw Pulser)

Benchmark and tests:
- `benchmark_simple_qubo.py` - Benchmark comparing all solvers
- `test_simple_portfolio_qubo.py` - Simple-QUBO tests (45)
- `test_slack_portfolio_qubo.py` - Slack-QUBO tests (24)
- `test_cqm_portfolio.py` - CQM tests (23)
- `test_nl_portfolio.py` - NL tests (31)
- `test_qhd_portfolio.py` - QHDOPT tests (28)
- `test_cuopt_portfolio.py` - cuOpt tests (skip without cuopt)
- `test_phi_portfolio.py` - PhiSolve tests (30)
- `test_pasqal_portfolio.py` - Pasqal tests (28)

Documentation:
- `README.md` - This file
- `linux-instructions.md` (at repo root) - Install and setup details

# Technical Explanation

**Note**: I did a first pass on this then had an LLM rewrite with instructions to drop in mathematical symbols to make it a bit easier to parse as well as create examples and fill in details that were tedious to type out. 

## Problem Statement

Select a subset of assets to **maximize total score** while respecting:
- **Budget constraint**: Total price should not exceed budget B (≤ inequality)
- **Duration constraint**: Total duration should not exceed maximum D (≤ inequality)
- **Cardinality constraint**: Select at most K assets (≤ inequality)

**Note**: A simple formulation uses equality constraints which become soft penalties for deviation from targets in QUBO. As you'll see later, I modified this to check satisfaction using inequality constraints (≤), which is more practical for real portfolio problems BUT uses more qubits.

## QUBO Formulation

The problem is encoded as a Quadratic Unconstrained Binary Optimization (QUBO) matrix where each binary variable x_i ∈ {0,1} represents whether asset i is selected.

### Matrix Construction

**Diagonal entries** (individual asset contributions):
```
Q[i,i] = -score_i + λ_b(p_i² - 2·B·p_i) + λ_d(d_i² - 2·D·d_i) + λ_c(1 - 2·K)
```

**Off-diagonal entries** (pairwise interactions, for i < j):
```
Q[i,j] = 2·λ_b·p_i·p_j + 2·λ_d·d_i·d_j + 2·λ_c
```

The matrix is symmetric: Q[j,i] = Q[i,j]

### Energy Function

For a selection vector x ∈ {0,1}ⁿ:
```
E(x) = Σᵢ Q[i,i]·xᵢ + Σᵢ<ⱼ Q[i,j]·xᵢ·xⱼ
```

The optimizer finds x that **minimizes** E(x).

## How Constraints Work

Each constraint is encoded as a **quadratic penalty term** that adds energy when violated.

### Penalty Term Derivation

For a constraint like "total price = B", we want to penalize (Σ pᵢxᵢ - B)²:

```
λ_b·(Σ pᵢxᵢ - B)² = λ_b·[Σᵢ pᵢ²xᵢ + 2·Σᵢ<ⱼ pᵢpⱼxᵢxⱼ - 2B·Σᵢ pᵢxᵢ + B²]
```

Since xᵢ² = xᵢ for binary variables, this expands to:
- **Diagonal contribution**: λ_b·(pᵢ² - 2·B·pᵢ)
- **Off-diagonal contribution**: 2·λ_b·pᵢ·pⱼ
- **Constant B²**: Dropped (doesn't affect optimization)

The same pattern applies to duration and cardinality constraints.

### Why Penalties Are "Soft"

Because the constant terms (B², D², K²) are dropped in QUBO formulation, the penalties don't represent the absolute violation cost. Instead, they represent the **relative cost** of different selections.

This means:
1. A solution violating a constraint can still be optimal if the score benefit outweighs the penalty
2. The λ weights determine how strictly constraints are enforced
3. Higher λ values make violations more expensive relative to score gains

This is OK, and might be desirable, but it means that you can get answers where the constraints are violated. To make constraints "hard", you need to use one of the techniques below.

## Making Constraints "Hard"

When thinking about QUBO it is often easier to default to soft constraints, but there are approaches to enforce hard constraints:

### Approach 1: Very High Penalty Weights

Increase λ values until violation cost exceeds any possible score benefit:

```python
# Soft constraints (defaults)
optimizer = SimplePortfolioQUBO(
    assets=assets,
    budget=3.0,
    max_duration=15,
    max_cardinality=3,
    lambda_budget=2.0,      # Soft
    lambda_duration=10.0,   # Soft
    lambda_cardinality=5.0  # Soft
)

# Hard constraints via high penalties
optimizer = SimplePortfolioQUBO(
    assets=assets,
    budget=3.0,
    max_duration=15,
    max_cardinality=3,
    lambda_budget=1000.0,      # Hard - any budget violation is very expensive
    lambda_duration=1000.0,    # Hard
    lambda_cardinality=1000.0  # Hard
)
```

**Tradeoff**: Very high penalties can cause numerical issues and make the energy landscape difficult to optimize.

### Approach 2: Post-Processing Filter

Keep soft constraints during optimization, then filter invalid solutions:

```python
result = optimizer.solve(num_reads=1000)
sampleset = result['sampleset']

# Filter to only valid solutions
valid_samples = []
for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy']):
    selection = np.array([sample[a.id] for a in optimizer.assets])
    check = optimizer.check_constraints(selection)

    if (check['cardinality_satisfied'] and
        check['duration_satisfied'] and
        abs(check['total_price'] - optimizer.budget) < 0.1):  # Budget tolerance
        valid_samples.append((selection, energy))

# Best valid solution
if valid_samples:
    best = min(valid_samples, key=lambda x: x[1])
```

**Tradeoff**: May discard good solutions; requires more samples to find valid ones.

### Approach 3: Inequality Constraint Encoding

For inequality constraints like "duration ≤ D" (rather than "duration = D"), we can use slack variables:

* Standard equality: penalize (duration - D)²
** This penalizes both over AND under
* For inequality (duration ≤ D), only penalize over:
** Add slack variable s and enforce: duration + s = D, s ≥ 0
** Or use: max(0, duration - D)² penalty

This requires modifying the QUBO construction to treat duration as a one-sided constraint.

### Approach 4: Adaptive Penalty Tuning

Start with low penalties and increase until constraints are satisfied:

```python
def solve_with_adaptive_penalties(optimizer, target_satisfaction=0.95):
    lambda_multiplier = 1.0

    for iteration in range(10):
        # Scale all penalties
        opt = SimplePortfolioQUBO(
            assets=optimizer.assets,
            budget=optimizer.budget,
            max_duration=optimizer.max_duration,
            max_cardinality=optimizer.max_cardinality,
            lambda_budget=optimizer.lambda_b * lambda_multiplier,
            lambda_duration=optimizer.lambda_d * lambda_multiplier,
            lambda_cardinality=optimizer.lambda_c * lambda_multiplier
        )

        result = opt.solve(num_reads=1000)
        check = opt.check_constraints(result['selection'])

        if all([check['budget_satisfied'],
                check['duration_satisfied'],
                check['cardinality_satisfied']]):
            return result

        lambda_multiplier *= 2.0  # Double penalties

    return result  # Best effort
```

## Inequality vs Equality Constraints

One could do the following:
- Budget: Σ p_i x_i = B (penalizes both over AND under budget)
- Duration: Σ d_i x_i = D (penalizes both over AND under duration)
- Cardinality: Σ x_i = K (penalizes selecting more OR fewer than K)

**Implementation uses inequality constraints:**
- Budget: Σ p_i x_i ≤ B (only penalizes over-budget)
- Duration: Σ d_i x_i ≤ D (only penalizes over-duration)
- Cardinality: Σ x_i ≤ K (only penalizes selecting too many)

Equality formulation is mathematically elegant but impractical:
- Under-budget solutions are penalized equally to over-budget
- Selecting fewer assets (even if optimal) incurs cardinality penalty
- This causes QUBO to find solutions that violate constraints in practice

### Solver implementations in this directory

Core (always available):

- **Simple (soft penalty) - `simple_portfolio_qubo.py`**: Quadratic
  penalty formulation, checks constraints post-hoc as ≤ inequalities.
  Fast but may produce infeasible solutions (use filtering).
- **Slack variable - `slack_portfolio_qubo.py`**: Binary slack
  variables encode true inequalities (`Σp_ix_i + s = B`, `s ≥ 0`).
  No penalty for under-budget solutions. More qubits but higher
  quality.
- **CQM - `cqm_portfolio.py`**: `dimod.ConstrainedQuadraticModel`
  with native inequality constraints. `solve_exact()` and `solve_sa()`
  paths.
- **NL - `nl_portfolio.py`**: `dwave-optimization.Model` tensor-DAG
  formulation. Inequalities are first-class; targets D-Wave's Stride
  hybrid solver (`LeapHybridNLSampler`) which scales to ~2M
  variables.

Optional solver integrations (lazy imports; benchmark rows fall back
to FAIL when the library is absent):

- **QHDOPT - `qhd_portfolio.py`**: Two paths. `solve_qp` runs QHDOPT's
  Quadratic Programming relaxation on the QUBO penalty matrix.
  `solve_sympy` uses QHDOPT's native SymPy-based MIQP formulation
  with explicit linear constraints (consistently competitive on
  benchmarks).
- **cuOpt - `cuopt_portfolio.py`**: NVIDIA's GPU optimizer. `solve_milp`
  for binary integer programming (matches ILP on score, faster at
  scale); `solve_qp` for an LP relaxation of the indefinite QUBO
  penalty matrix.
- **PhiSolve - `phi_portfolio.py`**: Artephi Computing's
  Quantum-Inspired Hamiltonian Descent (QIHD) with PDQP refinement.
  `solve_qubo` on the penalty matrix; `solve_miqp` with native
  linear constraints.
- **Pasqal - `pasqal_portfolio.py`**: Four paths on the neutral-atom
  stack, all running on local emulators. `solve_qubo` (qubosolver
  LocalEmulator on the QUBO penalty matrix), `solve_qubo_slack`
  (qubosolver on the slack-variable matrix; capped at n ≤ 8 due to
  emulator scaling), `solve_mis` (Maximum Independent Set on a
  pairwise budget/duration conflict graph plus score-aware greedy
  post-selection), `solve_pulser` (hand-built Rydberg adiabatic
  pulse sequence; capped at n ≤ 12 by Qutip emulation cost).
  Comparability and encoding caveats are in the module docstring
  at the top of `pasqal_portfolio.py`.

# Implementation Details

## Usage Example

```python
from simple_portfolio_qubo import SimplePortfolioQUBO

# Define assets
assets = [
    {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
    {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
    {'id': 'C', 'price': 0.89, 'duration': 4, 'score': 5},
    {'id': 'D', 'price': 1.05, 'duration': 3, 'score': 1},
    {'id': 'E', 'price': 1.02, 'duration': 9, 'score': 9},
]

# Create optimizer
optimizer = SimplePortfolioQUBO(
    assets=assets,
    budget=3.0,
    max_duration=15,
    max_cardinality=3,
    lambda_budget=2.0,
    lambda_duration=10.0,
    lambda_cardinality=5.0
)

# Solve with simulated annealing
result = optimizer.solve(num_reads=1000, seed=42)

print(f"Selected: {result['selected_assets']}")
print(f"Energy: {result['energy']:.2f}")
print(f"Total Price: ${result['total_price']:.2f}")
print(f"Total Duration: {result['total_duration']}")
print(f"Total Score: {result['total_score']}")

# Check constraint satisfaction
check = optimizer.check_constraints(result['selection'])
print(f"Budget satisfied: {check['budget_satisfied']}")
print(f"Duration satisfied: {check['duration_satisfied']}")
print(f"Cardinality satisfied: {check['cardinality_satisfied']}")
```

## Slack Variable Implementation

The `SlackPortfolioQUBO` class uses binary-encoded slack variables to represent true inequality constraints. This section details how the slack variables are introduced and encoded.

### Variable Structure

For a problem with n assets:
```
Variables: [x_0, ..., x_{n-1}, sb_0, ..., sb_{k1}, sd_0, ..., sd_{k2}, sc_0, ..., sc_{k3}]
           |___ assets ___|    |__ budget __|  |_ duration _|  |__ card __|
                                   slack          slack          slack
```

Where:
- k1 = ceil(log₂(budget/precision))
- k2 = ceil(log₂(max_duration/precision))
- k3 = ceil(log₂(max_cardinality))

### Slack Encoding

Slack variables are binary-encoded with powers of 2:
```
s = Σ_{k=0}^{K} precision × 2^k × s_k
```

For budget with precision 0.1: s ∈ {0, 0.1, 0.2, 0.3, ..., 3.1}

## Test Validation

The test suite (`test_simple_portfolio_qubo.py`) validates constraint encoding with 45 tests:

### Formula Verification Tests
- `test_qubo_diagonal_q00`: Verifies Q[0,0] = -1483.00
- `test_qubo_offdiagonal_q01`: Verifies Q[0,1] = 853.96 
- `test_ising_h_vector`, `test_ising_J_matrix`: Verify Ising conversion matches expected

### Constraint Penalty Tests (`TestConstraintPenaltyEncoding`)

These tests isolate each constraint by zeroing other penalties:

1. **`test_budget_penalty_encoding`**
   - Creates 2 assets with prices 1.0 and 2.0, budget target 1.5
   - Verifies: selecting both (price=3.0, overshoots by 1.5) has higher energy than selecting one

2. **`test_duration_penalty_encoding`**
   - Creates 2 assets with durations 5 and 10, max duration 8
   - Verifies: selecting both (duration=15, overshoots by 7) has highest energy

3. **`test_cardinality_penalty_encoding`**
   - Creates 3 assets, max cardinality 2
   - Verifies: selecting exactly 2 has lower energy than selecting 1 or 3
   - This confirms the penalty is symmetric around the target

4. **`test_penalty_magnitude_scales_with_lambda`**
   - Compares λ_c=1.0 vs λ_c=100.0
   - Verifies: higher penalty leads to stricter constraint enforcement

5. **`test_combined_constraints_tradeoff`**
   - Tests that different λ weights lead to different optimal solutions

There is a separate test suite (`test_slack_portfolio_qubo.py`) for the slack variable implementation with 24 tests covering slack variable encoding, energy evaluation, constraint satisfaction, and solving.

## Benchmark Results

Two representative runs from the current benchmark
(`benchmark_simple_qubo.py`). Optional solvers (QHD, cuOpt, Phi,
Pasqal) only produce real numbers when their respective libraries
are installed; cuOpt requires NVIDIA hardware.

### Simple PDF example (n = 5, optimum = 17.0, pick A + E)

| Solver | Score | Gap% | Time (ms) |
|---|---:|---:|---:|
| Brute Force / ILP / CQM-Exact / NL-Exact | 17.0 | 0.0% | 0.1–186 |
| QUBO (SA / Filtered / HighPen / Slack / Slack+Filt) | 17.0 | 0.0% | 35–136 |
| Greedy / Random | 17.0 | 0.0% | <10 |
| QHD-SymPy, Phi-MIQP (native-constraint MIQP) | 17.0 | 0.0% | 63–3245 |
| **Pasqal-Pulser** | 17.0 | 0.0% | 40 |
| **Pasqal-QUBO**, **Pasqal-MIS** | 14.0 | 17.6% | 13–2200 |
| **Pasqal-Slack** | 13.0 | 23.5% | 3089 |
| QHD-QP, Phi-QUBO (QUBO penalty matrix on QI backends) | 9.0 | 47.1% | 150–2050 |

### Random n = 12 (optimum = 70.0)

| Solver | Score | Gap% | Time (ms) |
|---|---:|---:|---:|
| Brute Force / ILP / CQM-Exact / NL-Exact | 70.0 | 0.0% | 3–184 |
| QHD-SymPy, Phi-MIQP | 70.0 | 0.0% | 84–3281 |
| **Pasqal-MIS** | 70.0 | 0.0% | 25 |
| QUBO (Slack+Filt) | 69.0 | 1.4% | 267 |
| Random / Greedy | 62–68 | 2.9–11.4% | <10 |
| **Pasqal-Pulser** | 58.0 | 17.1% | 810 |
| **Pasqal-QUBO** | 55.0 | 21.4% | 14000 |
| **Pasqal-Slack** | — | skipped | n > 8 cap |
| CQM (SA) / QHD-QP / Phi-QUBO | 11–37 | 47–84% | 165–3281 |
| QUBO (SA), QUBO (Slack) | infeasible | — | 120–260 |

A few observations from these runs:

- The MIQP-style native-constraint solvers (CQM-Exact, NL-Exact,
  QHD-SymPy, Phi-MIQP) consistently match ILP at these sizes.
  Pasqal-MIS is the speed surprise: 25 ms to the global optimum at
  n = 12 via a pairwise conflict-graph relaxation.
- The QUBO-penalty-matrix solvers on quantum-inspired backends
  (QHD-QP, Phi-QUBO, Pasqal-QUBO) cluster together at the
  high-gap end. Same energy landscape, different backends, very
  similar quality — a property of the penalty-matrix formulation
  rather than the backends.
- At n = 12, soft-penalty QUBO (SA, Slack) starts producing
  infeasible solutions. Slack-with-filter recovers feasibility at
  the cost of a small gap.

The original "slack achieves 0.5% gap" result still reproduces on
QUBO (Slack+Filter); the wider table here shows where the optional
solvers slot in once they're installed.

# D-Wave API Details

The D-Wave API has specific limits on the magnitude of coefficients that can be submitted. Formulations produce values that are often outside these limits, especially for larger problems. The `slack_portfolio_qubo.py` implementation addresses this by introducing slack variables to encode true inequality constraints, which helps keep coefficients within the allowed range.

## D-Wave QPU Coefficient Limits

When submitting to D-Wave quantum hardware, coefficients must fit within specific ranges.

### Hardware Limits (from live API query)

| Property | Advantage (Pegasus) | Advantage2 (Zephyr) |
|----------|---------------------|---------------------|
| **h_range** (linear) | [-4.0, 4.0] | [-6.0, 6.0] |
| **j_range** (coupling) | [-1.0, 1.0] | [-1.0, 1.0] |
| **extended_j_range** | [-2.0, 1.0] | [-2.0, 1.0] |
| **num_qubits** | 5760 | 4800 |
| **topology** | Pegasus | Zephyr |

### The Problem: Large Coefficient Values

Our simple formulation produces values far outside these limits:

| Parameter | Our Values | D-Wave Limit | Ratio |
|-----------|------------|--------------|-------|
| h (linear) | [-53, -19] | [-6, 6] | ~9x too large |
| J (coupling) | [63, 319] | [-1, 1] | ~319x too large |

**Why?** The formulation uses raw constraint values. Duration penalties like `λ_d × d² = 10 × 81 = 810` produce large coefficients.

### Normalization vs Reformulation

There are three approaches to handle this:

**Approach 1: Post-hoc Normalization (implemented)**

Scale h and J after computing the QUBO to fit QPU limits:

```python
# Get normalized Ising parameters for Advantage2
h_norm, J_norm, scale = optimizer.normalize_for_qpu(
    h_range=(-6.0, 6.0),
    j_range=(-1.0, 1.0)
)

# Or get a normalized BQM ready for QPU submission
bqm, scale_factor = optimizer.to_bqm_normalized()

# View normalization statistics
info = optimizer.get_normalization_info()
print(f"Compression ratio: {info['compression_ratio']:.1f}x")
```

Pros:
- Preserves original simple formulation
- Simple to implement and verify
- Can target any QPU's limits

Cons:
- Large compression ratio (~319x) reduces precision
- Small energy differences may be lost to noise

**Approach 2: Normalize Inputs Before QUBO Construction**

Normalize constraint values to [0,1] before building the QUBO:

```python
price_norm = (price - price_min) / (price_max - price_min)
duration_norm = duration / max_duration
score_norm = score / max_score
```

Pros:
- Naturally smaller QUBO values
- Better QPU dynamic range utilization

Cons:
- Changes problem semantics
- Requires re-tuning penalty weights

**Approach 3: Scale Penalty Weights**

Reduce λ weights to account for large squared terms:

```python
# Instead of λ_d = 10
# Use λ_d = 10 / max_duration² = 10 / 225 ≈ 0.044
lambda_duration_scaled = lambda_duration / (max_duration ** 2)
```

Pros:
- Simple adjustment to existing formulation
- Directly controls coefficient magnitudes

Cons:
- Requires knowledge of data ranges
- Problem-specific tuning

### Normalization Example

```python
optimizer = SimplePortfolioQUBO(assets=assets, **constraints)

# Check how much compression is needed
info = optimizer.get_normalization_info()
print(f"Original h range: {info['original_h_range']}")
print(f"Original J range: {info['original_j_range']}")
print(f"Compression ratio: {info['compression_ratio']:.1f}x")

# Get normalized BQM for QPU
bqm, scale = optimizer.to_bqm_normalized()

# After QPU returns energy, convert back to original scale:
# original_energy = qpu_energy / scale
```

## Auto-Scaling

D-Wave's `auto_scale` parameter (enabled by default) performs similar normalization automatically. However, explicit normalization gives you:
- Visibility into the compression ratio
- Control over target ranges
- Ability to assess precision loss

# Extended Usage Examples

## Running the Code

All commands should be run from the `experiments/simple/` directory:

```bash
cd experiments/simple
```

### Run the Optimizers Directly

```bash
# Run simple QUBO demo
python simple_portfolio_qubo.py

# Run slack variable QUBO demo
python slack_portfolio_qubo.py
```

### Run Tests

```bash
# Run all tests
python -m pytest . -v

# Run specific test file
python -m pytest test_simple_portfolio_qubo.py -v
python -m pytest test_slack_portfolio_qubo.py -v
```

### Run Benchmark

```bash
# Simple example problem
python benchmark_simple_qubo.py --problem-set simple

# Random problem with multiple trials
python benchmark_simple_qubo.py --problem-set random --num-assets 12 --num-trials 5 --seed 42

# Larger problem
python benchmark_simple_qubo.py --problem-set random --num-assets 15 --num-trials 3
```

### Benchmark Options

```
--problem-set {simple,random}  Problem type (default: simple)
--num-assets N              Number of assets for random problems (default: 5)
--num-trials N              Number of trials to run (default: 1)
--qubo-reads N              Number of QUBO annealing reads (default: 1000)
--random-samples N          Number of random samples (default: 1000)
--seed N                    Random seed for reproducibility
```
