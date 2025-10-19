# Test Suite for QPO

## Running Tests

```bash
# All tests (including slow GA tests)
pytest tests/ -v

# CVXPY tests only (fast, recommended)
pytest tests/test_classical_optimizer.py::TestClassicalOptimizerCVXPY -v

# Specific asset count
pytest tests/test_classical_optimizer.py -k "16" -v

# Quick smoke test
pytest tests/test_classical_optimizer.py::TestClassicalOptimizerCVXPY::test_optimization_and_backtest[5] -v
```

## Test Coverage

### Classical Optimizer Tests

**CVXPY Continuous Optimization** (`TestClassicalOptimizerCVXPY`)
- Tests portfolio optimization at scales: 5, 8, 16, 32, 48, 64, 80, 96 assets
- Each test includes:
  - Single optimization validation
  - Rolling window backtest (1-year train, monthly rebalance)
  - Comparison to equal-weight benchmark
  - Performance metrics (Sharpe, Sortino, drawdown, win rate)
  - Runtime tracking

**Genetic Algorithm** (`TestClassicalOptimizerGA`)
- Tests cardinality-constrained optimization: 5, 8, 16 assets
- Validates k = n_assets // 2 constraint
- Includes backtest with bi-monthly rebalancing
- NOTE: GA tests are slower (~5-10s per test)

### Performance Benchmarks

Typical runtimes on M-series Mac:

| Assets | CVXPY (single) | CVXPY (backtest) | GA (single) |
|--------|----------------|------------------|-------------|
| 5      | 0.004s         | 0.02s           | 0.6s        |
| 16     | 0.002s         | 0.03s           | 5-7s        |
| 32     | 0.007s         | 0.04s           | N/A (slow)  |
| 64     | 0.011s         | 0.10s           | N/A         |
| 96     | 0.015s         | 0.14s           | N/A         |

### Metrics Validated

- **Weight constraints**: sum=1, non-negative
- **Cardinality**: ≤ k assets for GA
- **Return bounds**: -100% to +300% annualized
- **Risk bounds**: 0% to 200% volatility
- **Backtest quality**: positive final value, valid metrics

## Example Output

```
======================================================================
CVXPY Optimizer - 16 Assets
======================================================================
Single Optimization:
  Expected Return: 54.12%
  Expected Risk:   27.71%
  Sharpe Ratio:    1.953
  Assets Selected: 2/16
  Runtime:         0.0022s

Backtest Results (1-year train, monthly rebalance):
  Total Return:      4.27%
  Annualized Return: 4.27%
  Volatility:        39.38%
  Sharpe Ratio:      0.108
  Sortino Ratio:     0.182
  Max Drawdown:      -31.93%
  Win Rate:          47.0%
  Final Value:       $104,272.45
  Rebalances:        12
  Avg Runtime:       0.0022s
  Total Runtime:     0.0264s

Vs. Equal-Weight Benchmark:
  Relative Return:   -29.31%
  Tracking Error:    48.52%
  Information Ratio: -0.643
  Outperformance:    121/251 days
```
