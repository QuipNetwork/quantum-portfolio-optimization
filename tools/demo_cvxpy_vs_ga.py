"""Demo comparing CVXPY vs GA for classical optimization."""

import numpy as np
import pandas as pd
import time
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.utils.data_prep import compute_returns

# Generate synthetic data
np.random.seed(42)
n_assets = 16
n_days = 252

dates = pd.date_range('2023-01-01', periods=n_days)
tickers = [f'ASSET_{i:02d}' for i in range(n_assets)]

prices_data = {}
for i, ticker in enumerate(tickers):
    drift = 0.0005 * (i + 1)
    vol = 0.02
    returns = np.random.normal(drift, vol, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    prices_data[ticker] = prices

prices = pd.DataFrame(prices_data, index=dates)
returns = compute_returns(prices, method='log')

print("="*70)
print("CVXPY vs GA Comparison Demo")
print("="*70)
print(f"\nPortfolio: {n_assets} assets, {n_days} days of data\n")

# Test 1: CVXPY (no cardinality)
print("─" * 70)
print("Test 1: CVXPY - Continuous Optimization (No Cardinality)")
print("─" * 70)

optimizer_cvxpy = ClassicalOptimizer(gamma=1.0, method='cvxpy')
start = time.time()
result_cvxpy = optimizer_cvxpy.optimize(returns)
cvxpy_time = time.time() - start

weights_cvxpy = result_cvxpy['weights']
n_selected_cvxpy = (weights_cvxpy > 1e-6).sum()

print(f"Runtime: {cvxpy_time:.4f}s")
print(f"Assets selected: {n_selected_cvxpy}/{n_assets}")
print(f"Sharpe ratio: {result_cvxpy['metrics']['sharpe_ratio']:.3f}")
print(f"Expected return: {result_cvxpy['metrics']['expected_return']*100:.2f}%")
print(f"Expected risk: {result_cvxpy['metrics']['expected_risk']*100:.2f}%")

print("\nTop 5 holdings:")
for ticker, weight in weights_cvxpy.nlargest(5).items():
    print(f"  {ticker}: {weight*100:6.2f}%")

# Test 2: GA with cardinality k=8
print("\n" + "─" * 70)
print("Test 2: GA - Discrete Optimization with Cardinality (k=8)")
print("─" * 70)

k = 8
optimizer_ga = ClassicalOptimizer(gamma=1.0, k=k, method='ga')
start = time.time()
result_ga = optimizer_ga.optimize(returns)
ga_time = time.time() - start

weights_ga = result_ga['weights']
n_selected_ga = (weights_ga > 1e-6).sum()

print(f"Runtime: {ga_time:.4f}s  (slowdown: {ga_time/cvxpy_time:.1f}x)")
print(f"Assets selected: {n_selected_ga}/{n_assets} (target: {k})")
print(f"Sharpe ratio: {result_ga['metrics']['sharpe_ratio']:.3f}")
print(f"Expected return: {result_ga['metrics']['expected_return']*100:.2f}%")
print(f"Expected risk: {result_ga['metrics']['expected_risk']*100:.2f}%")

print("\nTop holdings:")
for ticker, weight in weights_ga[weights_ga > 1e-6].items():
    print(f"  {ticker}: {weight*100:6.2f}%")

# Test 3: Post-processed CVXPY (top-k selection)
print("\n" + "─" * 70)
print("Test 3: CVXPY + Post-Processing (Select Top-8)")
print("─" * 70)

start = time.time()
result_cvxpy_pp = optimizer_cvxpy.optimize(returns)
# Post-process: keep only top k
weights_pp = result_cvxpy_pp['weights'].nlargest(k)
weights_pp /= weights_pp.sum()
pp_time = time.time() - start

# Compute metrics for post-processed weights
mu = returns.mean().values * 252
Sigma = returns.cov().values * 252
w_arr = weights_pp.reindex(returns.columns, fill_value=0).values

exp_return = mu @ w_arr
exp_risk = np.sqrt(w_arr @ Sigma @ w_arr)
sharpe = exp_return / exp_risk if exp_risk > 0 else 0

print(f"Runtime: {pp_time:.4f}s  (speedup: {ga_time/pp_time:.1f}x vs GA)")
print(f"Assets selected: {k}/{n_assets}")
print(f"Sharpe ratio: {sharpe:.3f}")
print(f"Expected return: {exp_return*100:.2f}%")
print(f"Expected risk: {exp_risk*100:.2f}%")

print("\nTop holdings:")
for ticker, weight in weights_pp.items():
    print(f"  {ticker}: {weight*100:6.2f}%")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n{'Method':<25} {'Runtime':>10} {'Assets':>8} {'Sharpe':>8}")
print("─" * 70)
print(f"{'CVXPY':<25} {cvxpy_time:>9.4f}s {n_selected_cvxpy:>8} {result_cvxpy['metrics']['sharpe_ratio']:>8.3f}")
print(f"{'GA (k=8)':<25} {ga_time:>9.4f}s {n_selected_ga:>8} {result_ga['metrics']['sharpe_ratio']:>8.3f}")
print(f"{'CVXPY + Top-K':<25} {pp_time:>9.4f}s {k:>8} {sharpe:>8.3f}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\n✅ For production: Use CVXPY (fast, exact)")
print("⚠️  For cardinality: Use CVXPY + post-processing (fast approximation)")
print("❌ Avoid GA: Too slow, not guaranteed to satisfy k constraint")
print("\n   GA is ~{:.0f}x slower than CVXPY".format(ga_time/cvxpy_time))
print("   Post-processing is ~{:.0f}x faster than GA".format(ga_time/pp_time))
