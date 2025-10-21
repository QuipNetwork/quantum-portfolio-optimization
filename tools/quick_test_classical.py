# Copyright (C) 2025 Postquant Labs Incorporated
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quick test of classical optimizer."""

import numpy as np
import pandas as pd
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.utils.data_prep import compute_returns

# Generate synthetic price data
np.random.seed(42)
n_days = 252
n_assets = 5

# Simulate prices with different characteristics
dates = pd.date_range('2023-01-01', periods=n_days)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

# Random walk prices
prices_data = {}
for i, ticker in enumerate(tickers):
    # Different drift and volatility per asset
    drift = 0.0005 * (i + 1)  # Annualized ~12.5% for ticker 1
    vol = 0.02 * (1 + i * 0.2)  # Different volatilities

    returns = np.random.normal(drift, vol, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    prices_data[ticker] = prices

prices_df = pd.DataFrame(prices_data, index=dates)

print("=" * 60)
print("Testing Classical Optimizer")
print("=" * 60)
print(f"\nGenerated synthetic data:")
print(f"  Assets: {n_assets}")
print(f"  Days: {n_days}")
print(f"\nFinal prices:")
print(prices_df.tail())

# Compute returns
returns = compute_returns(prices_df, method='log')

print(f"\nAnnualized statistics:")
print(f"  Mean returns: {(returns.mean() * 252 * 100).round(2).to_dict()}")
print(f"  Volatilities: {(returns.std() * np.sqrt(252) * 100).round(2).to_dict()}")

# Test CVXPY optimizer
print("\n" + "=" * 60)
print("Test 1: CVXPY Continuous Optimization")
print("=" * 60)

optimizer_cvxpy = ClassicalOptimizer(gamma=1.0, method='cvxpy')
result_cvxpy = optimizer_cvxpy.optimize(returns)

print(f"\nOptimal weights:")
for ticker, weight in result_cvxpy['weights'].items():
    print(f"  {ticker}: {weight*100:.2f}%")

print(f"\nPortfolio metrics:")
print(f"  Expected return: {result_cvxpy['metrics']['expected_return']*100:.2f}%")
print(f"  Expected risk: {result_cvxpy['metrics']['expected_risk']*100:.2f}%")
print(f"  Sharpe ratio: {result_cvxpy['metrics']['sharpe_ratio']:.3f}")
print(f"  Number of assets: {result_cvxpy['metrics']['n_assets']}")
print(f"  Runtime: {result_cvxpy['runtime']:.4f}s")

# Test GA optimizer with cardinality
print("\n" + "=" * 60)
print("Test 2: Genetic Algorithm with Cardinality (k=3)")
print("=" * 60)

optimizer_ga = ClassicalOptimizer(gamma=1.0, k=3, method='ga')
result_ga = optimizer_ga.optimize(returns)

print(f"\nOptimal weights:")
for ticker, weight in result_ga['weights'].items():
    if weight > 1e-6:
        print(f"  {ticker}: {weight*100:.2f}%")

print(f"\nPortfolio metrics:")
print(f"  Expected return: {result_ga['metrics']['expected_return']*100:.2f}%")
print(f"  Expected risk: {result_ga['metrics']['expected_risk']*100:.2f}%")
print(f"  Sharpe ratio: {result_ga['metrics']['sharpe_ratio']:.3f}")
print(f"  Number of assets: {result_ga['metrics']['n_assets']}")
print(f"  Runtime: {result_ga['runtime']:.4f}s")

# Test data preprocessing
print("\n" + "=" * 60)
print("Test 3: Data Preprocessing")
print("=" * 60)

from qpo.utils.data_prep import preprocess_portfolio, PortfolioStatistics

# Add some missing data
prices_with_gaps = prices_df.copy()
prices_with_gaps.iloc[10:13, 0] = np.nan  # 3-day gap in AAPL
prices_with_gaps.iloc[50, 2] = np.nan      # 1-day gap in GOOGL

print(f"\nOriginal data shape: {prices_df.shape}")
print(f"Data with gaps shape: {prices_with_gaps.shape}")
print(f"Missing values: {prices_with_gaps.isna().sum().sum()}")

cleaned = preprocess_portfolio(prices_with_gaps, max_missing_pct=0.05, max_ffill_days=3)
print(f"\nCleaned data shape: {cleaned.shape}")
print(f"Remaining missing values: {cleaned.isna().sum().sum()}")

# Test statistics computation
print("\n" + "=" * 60)
print("Test 4: Portfolio Statistics")
print("=" * 60)

stats = PortfolioStatistics(returns)
mu = stats.expected_returns()
Sigma = stats.covariance_matrix()
corr = stats.correlation_matrix()

print(f"\nExpected returns (annualized):")
print(mu.round(4))

print(f"\nCorrelation matrix:")
print(corr.round(3))

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
