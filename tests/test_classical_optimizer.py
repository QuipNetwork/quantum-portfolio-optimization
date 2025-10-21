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

"""Pytest suite for classical portfolio optimization with backtesting."""

import pytest
import numpy as np
import pandas as pd
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.backtest import Backtester
from qpo.utils.data_prep import compute_returns


def generate_synthetic_data(n_assets: int, n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic price data with realistic characteristics.

    Args:
        n_assets: Number of assets
        n_days: Number of trading days (default 504 = ~2 years for train/test)
        seed: Random seed for reproducibility

    Returns:
        DataFrame of prices (dates × tickers)
    """
    np.random.seed(seed)

    dates = pd.date_range('2022-01-01', periods=n_days)
    tickers = [f'ASSET_{i:03d}' for i in range(n_assets)]

    # Generate prices with different drift and volatility per asset
    prices_data = {}
    for i, ticker in enumerate(tickers):
        # Vary characteristics across assets
        drift = 0.0003 + (i / n_assets) * 0.0007  # 7.5% to 25% annualized
        vol = 0.015 + (i / n_assets) * 0.025      # 23% to 63% annualized volatility

        # Random walk with drift
        returns = np.random.normal(drift, vol, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        prices_data[ticker] = prices

    return pd.DataFrame(prices_data, index=dates)


def create_equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    """
    Create equal-weight portfolio benchmark.

    Args:
        prices: Price DataFrame

    Returns:
        Series of benchmark portfolio values
    """
    returns = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    equal_weights = np.ones(n_assets) / n_assets

    portfolio_returns = returns @ equal_weights
    portfolio_values = 100000 * (1 + portfolio_returns).cumprod()

    # Add initial value
    portfolio_values = pd.concat([
        pd.Series([100000], index=[prices.index[0]]),
        portfolio_values
    ])

    return portfolio_values


# Parametrized test cases: different asset counts
ASSET_COUNTS = [5, 8, 16, 32, 48, 64, 80, 96]


@pytest.mark.parametrize("n_assets", ASSET_COUNTS)
class TestClassicalOptimizerCVXPY:
    """Test CVXPY optimizer with different portfolio sizes."""

    def test_optimization_and_backtest(self, n_assets):
        """
        Test classical optimizer (CVXPY) with backtesting.

        This test:
        1. Generates synthetic data
        2. Runs optimization
        3. Performs rolling window backtest
        4. Compares to equal-weight benchmark
        5. Validates metrics and runtime
        """
        # Generate data (2 years)
        prices = generate_synthetic_data(n_assets, n_days=504)
        returns = compute_returns(prices, method='log')

        # Single optimization test
        optimizer = ClassicalOptimizer(gamma=1.0, method='cvxpy')
        result = optimizer.optimize(returns)

        # Validate single optimization
        assert 'weights' in result
        assert 'metrics' in result
        assert 'runtime' in result

        # Check weights sum to 1
        assert np.isclose(result['weights'].sum(), 1.0, atol=1e-6)

        # Check no negative weights
        assert (result['weights'] >= -1e-6).all()

        # Check metrics are reasonable
        metrics = result['metrics']
        assert -1.0 <= metrics['expected_return'] <= 3.0  # -100% to 300% annualized
        assert 0.0 <= metrics['expected_risk'] <= 2.0     # 0% to 200% volatility
        assert metrics['n_assets'] <= n_assets

        print(f"\n{'='*70}")
        print(f"CVXPY Optimizer - {n_assets} Assets")
        print(f"{'='*70}")
        print(f"Single Optimization:")
        print(f"  Expected Return: {metrics['expected_return']*100:.2f}%")
        print(f"  Expected Risk:   {metrics['expected_risk']*100:.2f}%")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
        print(f"  Assets Selected: {metrics['n_assets']}/{n_assets}")
        print(f"  Runtime:         {result['runtime']:.4f}s")

        # Backtesting
        backtester = Backtester(
            optimizer,
            train_days=252,   # 1 year training
            test_days=21,     # 1 month holding
            step_days=21      # Monthly rebalancing
        )

        backtest_results = backtester.run(prices, returns)

        # Validate backtest results
        assert 'portfolio_values' in backtest_results
        assert 'metrics' in backtest_results
        assert 'runtimes' in backtest_results

        bt_metrics = backtest_results['metrics']

        print(f"\nBacktest Results (1-year train, monthly rebalance):")
        print(f"  Total Return:      {bt_metrics['total_return']*100:.2f}%")
        print(f"  Annualized Return: {bt_metrics['annualized_return']*100:.2f}%")
        print(f"  Volatility:        {bt_metrics['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio:      {bt_metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio:     {bt_metrics['sortino_ratio']:.3f}")
        print(f"  Max Drawdown:      {bt_metrics['max_drawdown']*100:.2f}%")
        print(f"  Win Rate:          {bt_metrics['win_rate']*100:.1f}%")
        print(f"  Final Value:       ${bt_metrics['final_value']:,.2f}")
        print(f"  Rebalances:        {bt_metrics['n_rebalances']}")
        print(f"  Avg Runtime:       {bt_metrics['avg_runtime']:.4f}s")
        print(f"  Total Runtime:     {bt_metrics['total_runtime']:.4f}s")

        # Compare to equal-weight benchmark
        benchmark = create_equal_weight_benchmark(prices)

        from qpo.optimizers.backtest import compare_to_benchmark
        comparison = compare_to_benchmark(backtest_results, benchmark)

        print(f"\nVs. Equal-Weight Benchmark:")
        print(f"  Relative Return:   {comparison['relative_return']*100:.2f}%")
        print(f"  Tracking Error:    {comparison['tracking_error']*100:.2f}%")
        print(f"  Information Ratio: {comparison['information_ratio']:.3f}")
        print(f"  Outperformance:    {comparison['outperformance_days']}/{comparison['total_days']} days")

        # Assertions on backtest quality
        assert bt_metrics['final_value'] > 0
        assert -1.0 <= bt_metrics['total_return'] <= 10.0
        assert bt_metrics['volatility'] >= 0
        assert bt_metrics['n_rebalances'] > 0
        assert bt_metrics['avg_runtime'] > 0

        # Runtime scaling check (should be relatively fast)
        if n_assets <= 32:
            assert bt_metrics['avg_runtime'] < 0.1, f"CVXPY too slow for {n_assets} assets"
        elif n_assets <= 64:
            assert bt_metrics['avg_runtime'] < 0.5
        else:
            assert bt_metrics['avg_runtime'] < 2.0


@pytest.mark.parametrize("n_assets", [5, 8, 16])  # Limit GA to smaller sizes (slower)
class TestClassicalOptimizerGA:
    """Test Genetic Algorithm optimizer with cardinality constraints."""

    def test_optimization_with_cardinality(self, n_assets):
        """
        Test GA optimizer with cardinality constraint.

        Tests with k = n_assets // 2 (half the assets).
        """
        # Generate data
        prices = generate_synthetic_data(n_assets, n_days=504)
        returns = compute_returns(prices, method='log')

        # Cardinality: select half the assets
        k = max(3, n_assets // 2)

        # Optimize
        optimizer = ClassicalOptimizer(gamma=1.0, k=k, method='ga')
        result = optimizer.optimize(returns)

        # Validate
        assert 'weights' in result
        assert np.isclose(result['weights'].sum(), 1.0, atol=1e-4)
        assert (result['weights'] >= -1e-6).all()

        # Check cardinality constraint (approximately satisfied)
        n_selected = (result['weights'] > 1e-6).sum()
        assert n_selected <= k + 2, f"Cardinality violated: {n_selected} > {k}"

        metrics = result['metrics']

        print(f"\n{'='*70}")
        print(f"GA Optimizer - {n_assets} Assets (k={k})")
        print(f"{'='*70}")
        print(f"  Expected Return: {metrics['expected_return']*100:.2f}%")
        print(f"  Expected Risk:   {metrics['expected_risk']*100:.2f}%")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
        print(f"  Assets Selected: {n_selected}/{n_assets} (target: {k})")
        print(f"  Runtime:         {result['runtime']:.4f}s")

        # Backtesting
        backtester = Backtester(
            optimizer,
            train_days=252,
            test_days=21,
            step_days=42  # Bi-monthly (GA is slower)
        )

        backtest_results = backtester.run(prices, returns)
        bt_metrics = backtest_results['metrics']

        print(f"\nBacktest Results:")
        print(f"  Total Return:      {bt_metrics['total_return']*100:.2f}%")
        print(f"  Annualized Return: {bt_metrics['annualized_return']*100:.2f}%")
        print(f"  Sharpe Ratio:      {bt_metrics['sharpe_ratio']:.3f}")
        print(f"  Avg Runtime:       {bt_metrics['avg_runtime']:.4f}s")

        # Runtime assertions (GA is slower than CVXPY)
        if n_assets <= 8:
            assert result['runtime'] < 5.0, "GA too slow for small portfolios"
        elif n_assets <= 16:
            assert result['runtime'] < 10.0, "GA too slow for medium portfolios"
        else:
            assert result['runtime'] < 20.0, "GA too slow"


def test_data_preprocessing():
    """Test data preprocessing utilities."""
    from qpo.utils.data_prep import preprocess_portfolio, PortfolioStatistics

    # Generate data with gaps
    prices = generate_synthetic_data(10, n_days=252)
    prices_with_gaps = prices.copy()

    # Add missing data
    prices_with_gaps.iloc[10:13, 0] = np.nan
    prices_with_gaps.iloc[50, 2] = np.nan

    # Preprocess
    cleaned = preprocess_portfolio(prices_with_gaps, max_missing_pct=0.05)

    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned.columns) == 10  # No columns removed

    # Test statistics
    returns = compute_returns(prices, method='log')
    stats = PortfolioStatistics(returns)

    mu = stats.expected_returns()
    Sigma = stats.covariance_matrix()
    corr = stats.correlation_matrix()

    assert len(mu) == 10
    assert Sigma.shape == (10, 10)
    assert corr.shape == (10, 10)
    assert np.allclose(np.diag(corr), 1.0)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
