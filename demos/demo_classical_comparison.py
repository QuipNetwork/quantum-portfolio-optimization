"""Comprehensive comparison of all portfolio optimization methods."""

import numpy as np
import pandas as pd
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.equal_weight import EqualWeightOptimizer
from qpo.optimizers.risk_parity import RiskParityOptimizer
from qpo.optimizers.regularized import L1RegularizedOptimizer, L2RegularizedOptimizer
from qpo.optimizers.backtest import Backtester
from qpo.utils.data_prep import compute_returns

# Generate synthetic data
np.random.seed(42)
n_assets = 32
n_days = 504  # 2 years

dates = pd.date_range('2022-01-01', periods=n_days)
tickers = [f'ASSET_{i:02d}' for i in range(n_assets)]

prices_data = {}
for i, ticker in enumerate(tickers):
    drift = 0.0003 + (i / n_assets) * 0.0007
    vol = 0.015 + (i / n_assets) * 0.025
    returns = np.random.normal(drift, vol, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    prices_data[ticker] = prices

prices = pd.DataFrame(prices_data, index=dates)
returns = compute_returns(prices, method='log')

print("="*80)
print("Portfolio Optimization Method Comparison")
print("="*80)
print(f"\nData: {n_assets} assets, {n_days} days (~2 years)")
print(f"Test: 1-year training, monthly rebalancing\n")

# Define all optimizers
optimizers = {
    'Equal-Weight (1/N)': EqualWeightOptimizer(),
    'Risk Parity': RiskParityOptimizer(),
    'Mean-Variance': ClassicalOptimizer(gamma=1.0, method='cvxpy'),
    'L1 (λ=0.005)': L1RegularizedOptimizer(gamma=1.0, lambda_l1=0.005),
    'L1 (λ=0.01)': L1RegularizedOptimizer(gamma=1.0, lambda_l1=0.01),
    'L2 (λ=0.05)': L2RegularizedOptimizer(gamma=1.0, lambda_l2=0.05),
    'L2 (λ=0.1)': L2RegularizedOptimizer(gamma=1.0, lambda_l2=0.1),
}

# Run backtests
print("="*80)
print("Running backtests...")
print("="*80)

backtest_results = {}
for name, optimizer in optimizers.items():
    print(f"\nTesting: {name}...")
    backtester = Backtester(
        optimizer,
        train_days=252,   # 1 year training
        test_days=21,     # 1 month holding
        step_days=21      # Monthly rebalancing
    )
    result = backtester.run(prices, returns)
    backtest_results[name] = result
    print(f"  ✓ Complete (Sharpe: {result['metrics']['sharpe_ratio']:.3f}, "
          f"Return: {result['metrics']['annualized_return']*100:.1f}%)")

# Print results table
print("\n" + "="*80)
print("BACKTEST RESULTS")
print("="*80)
print(f"\n{'Method':<22} {'Return':>10} {'Vol':>8} {'Sharpe':>8} {'Sortino':>8} "
      f"{'MaxDD':>8} {'Win%':>6} {'Time':>8}")
print("-"*80)

for name, result in backtest_results.items():
    m = result['metrics']
    print(f"{name:<22} {m['annualized_return']*100:>9.1f}% {m['volatility']*100:>7.1f}% "
          f"{m['sharpe_ratio']:>8.3f} {m['sortino_ratio']:>8.3f} {m['max_drawdown']*100:>7.1f}% "
          f"{m['win_rate']*100:>5.1f}% {m['avg_runtime']:>7.4f}s")

# Find best performers
sharpe_ranking = sorted(backtest_results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'], reverse=True)
return_ranking = sorted(backtest_results.items(), key=lambda x: x[1]['metrics']['annualized_return'], reverse=True)

print("\n" + "="*80)
print("RANKINGS")
print("="*80)

print("\nBy Sharpe Ratio:")
for i, (name, result) in enumerate(sharpe_ranking[:5], 1):
    print(f"  {i}. {name:<22} {result['metrics']['sharpe_ratio']:.3f}")

print("\nBy Annualized Return:")
for i, (name, result) in enumerate(return_ranking[:5], 1):
    print(f"  {i}. {name:<22} {result['metrics']['annualized_return']*100:.2f}%")

# Portfolio concentration analysis
print("\n" + "="*80)
print("PORTFOLIO CHARACTERISTICS (Final Rebalance)")
print("="*80)
print(f"\n{'Method':<22} {'Assets':>8} {'Eff N':>8} {'Herfindahl':>11} {'Top 5 Weight':>12}")
print("-"*80)

for name, result in backtest_results.items():
    if result['weights_history']:
        last_weights = pd.Series(result['weights_history'][-1]['weights'])
        n_assets_used = (last_weights > 1e-6).sum()
        herfindahl = np.sum(last_weights.values ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else 0
        top5_weight = last_weights.nlargest(5).sum()

        print(f"{name:<22} {n_assets_used:>8} {effective_n:>8.1f} {herfindahl:>11.4f} {top5_weight*100:>11.1f}%")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

best_sharpe = sharpe_ranking[0]
print(f"\n✅ Best Risk-Adjusted (Sharpe): {best_sharpe[0]}")
print(f"   Sharpe: {best_sharpe[1]['metrics']['sharpe_ratio']:.3f}, "
      f"Return: {best_sharpe[1]['metrics']['annualized_return']*100:.1f}%")

best_return = return_ranking[0]
print(f"\n💰 Highest Return: {best_return[0]}")
print(f"   Return: {best_return[1]['metrics']['annualized_return']*100:.1f}%, "
      f"Sharpe: {best_return[1]['metrics']['sharpe_ratio']:.3f}")

# Find most stable (lowest drawdown)
drawdown_ranking = sorted(backtest_results.items(),
                         key=lambda x: abs(x[1]['metrics']['max_drawdown']))
most_stable = drawdown_ranking[0]
print(f"\n🛡️  Most Stable (Min Drawdown): {most_stable[0]}")
print(f"   Max Drawdown: {most_stable[1]['metrics']['max_drawdown']*100:.1f}%, "
      f"Sharpe: {most_stable[1]['metrics']['sharpe_ratio']:.3f}")

# Fastest
runtime_ranking = sorted(backtest_results.items(),
                         key=lambda x: x[1]['metrics']['avg_runtime'])
fastest = runtime_ranking[0]
print(f"\n⚡ Fastest: {fastest[0]}")
print(f"   Avg Time: {fastest[1]['metrics']['avg_runtime']:.6f}s")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("\n1. For Production (Speed + Stability):")
print("   → Equal-Weight or Risk Parity")
print("   → Fast, robust to estimation error")

print("\n2. For Performance (Risk-Adjusted Returns):")
print(f"   → {sharpe_ranking[0][0]}")
print("   → Highest Sharpe ratio")

print("\n3. For Sparsity (Few Assets):")
print("   → L1 Regularized")
print("   → Automatically selects subset of assets")

print("\n4. For Stability (Low Turnover):")
print("   → Risk Parity or L2 Regularized")
print("   → More uniform weights = less rebalancing")

print("\n" + "="*80)
