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

"""Test suite for regularized portfolio optimizers (L1/L2)."""

import pytest
import numpy as np
import pandas as pd
from qpo.optimizers.regularized import L1RegularizedOptimizer, L2RegularizedOptimizer
from qpo.utils.data_prep import compute_returns


def generate_synthetic_data(n_assets: int, n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data."""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n_days)
    tickers = [f'ASSET_{i:03d}' for i in range(n_assets)]

    prices_data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0003 + (i / n_assets) * 0.0007
        vol = 0.015 + (i / n_assets) * 0.025
        returns = np.random.normal(drift, vol, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        prices_data[ticker] = prices

    return pd.DataFrame(prices_data, index=dates)


class TestL1Regularization:
    """Test L1 (Lasso) regularized optimizer."""

    @pytest.mark.parametrize("lambda_l1", [0.001, 0.01, 0.1])
    def test_l1_regularization(self, lambda_l1):
        """Test L1 (Lasso) regularization with different lambda values."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = L1RegularizedOptimizer(gamma=1.0, lambda_l1=lambda_l1)
        result = optimizer.optimize(returns)

        weights = result['weights']
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        # Allow small negative values due to numerical precision
        assert (weights >= -1e-10).all()

        # Higher lambda should produce sparser portfolios
        n_nonzero = (weights > 1e-6).sum()

        print(f"\n{'='*70}")
        print(f"L1-Regularized Portfolio - λ={lambda_l1}")
        print(f"{'='*70}")
        print(f"  Assets selected:  {n_nonzero}/{n_assets}")
        print(f"  Sharpe ratio:     {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:      {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Herfindahl:       {result['metrics']['herfindahl_index']:.4f}")
        print(f"  Runtime:          {result['runtime']:.4f}s")

        # Sparsity should increase with lambda
        if lambda_l1 == 0.1:
            assert n_nonzero < n_assets * 0.5, "High lambda should produce sparse portfolio"

    def test_sparsity_increases_with_lambda(self):
        """Test that higher lambda produces sparser portfolios."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        lambdas = [0.001, 0.01, 0.1]
        sparsities = []

        for lambda_l1 in lambdas:
            optimizer = L1RegularizedOptimizer(gamma=1.0, lambda_l1=lambda_l1)
            result = optimizer.optimize(returns)
            n_nonzero = (result['weights'] > 1e-6).sum()
            sparsities.append(n_nonzero)

        # Sparsity should generally decrease (fewer assets selected) as lambda increases
        # Allow for some variation due to optimization dynamics
        assert sparsities[-1] <= sparsities[0], "Higher lambda should not increase portfolio size"

    def test_metrics(self):
        """Test that all expected metrics are present."""
        prices = generate_synthetic_data(20)
        returns = compute_returns(prices, method='log')
        optimizer = L1RegularizedOptimizer(lambda_l1=0.01)
        result = optimizer.optimize(returns)

        assert 'metrics' in result
        metrics = result['metrics']

        required_metrics = [
            'expected_return',
            'expected_risk',
            'sharpe_ratio',
            'n_assets',
            'herfindahl_index',
            'effective_n_assets'
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float))

    def test_performance(self):
        """Test that L1 optimizer is reasonably fast."""
        prices = generate_synthetic_data(50)
        returns = compute_returns(prices, method='log')
        optimizer = L1RegularizedOptimizer(lambda_l1=0.01)
        result = optimizer.optimize(returns)

        # Should complete in reasonable time
        assert result['runtime'] < 1.0


class TestL2Regularization:
    """Test L2 (Ridge) regularized optimizer."""

    def test_l2_regularization(self):
        """Test L2 (Ridge) regularization."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = L2RegularizedOptimizer(gamma=1.0, lambda_l2=0.1)
        result = optimizer.optimize(returns)

        weights = result['weights']
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        # Allow small negative values due to numerical precision
        assert (weights >= -1e-10).all()

        # L2 should produce more uniform weights than unregularized
        print(f"\n{'='*70}")
        print(f"L2-Regularized Portfolio")
        print(f"{'='*70}")
        print(f"  Sharpe ratio:     {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:      {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Weight std:       {weights.std():.4f}")
        print(f"  Runtime:          {result['runtime']:.4f}s")

    @pytest.mark.parametrize("lambda_l2", [0.01, 0.1, 1.0])
    def test_different_lambda_values(self, lambda_l2):
        """Test L2 regularization with different lambda values."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = L2RegularizedOptimizer(gamma=1.0, lambda_l2=lambda_l2)
        result = optimizer.optimize(returns)

        weights = result['weights']
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights >= -1e-10).all()

        print(f"\n{'='*70}")
        print(f"L2-Regularized Portfolio - λ={lambda_l2}")
        print(f"{'='*70}")
        print(f"  Sharpe ratio:     {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:      {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Weight std:       {weights.std():.4f}")
        print(f"  Runtime:          {result['runtime']:.4f}s")

    def test_uniformity_increases_with_lambda(self):
        """Test that higher lambda produces more uniform weights."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        lambdas = [0.01, 0.1, 1.0]
        weight_stds = []

        for lambda_l2 in lambdas:
            optimizer = L2RegularizedOptimizer(gamma=1.0, lambda_l2=lambda_l2)
            result = optimizer.optimize(returns)
            weight_stds.append(result['weights'].std())

        # Higher lambda should generally lead to more uniform weights (lower std)
        # Allow for some variation
        assert weight_stds[-1] <= weight_stds[0], "Higher lambda should not increase weight dispersion"

    def test_metrics(self):
        """Test that all expected metrics are present."""
        prices = generate_synthetic_data(20)
        returns = compute_returns(prices, method='log')
        optimizer = L2RegularizedOptimizer(lambda_l2=0.1)
        result = optimizer.optimize(returns)

        assert 'metrics' in result
        metrics = result['metrics']

        required_metrics = [
            'expected_return',
            'expected_risk',
            'sharpe_ratio',
            'n_assets',
            'herfindahl_index',
            'effective_n_assets'
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float))

    def test_performance(self):
        """Test that L2 optimizer is reasonably fast."""
        prices = generate_synthetic_data(50)
        returns = compute_returns(prices, method='log')
        optimizer = L2RegularizedOptimizer(lambda_l2=0.1)
        result = optimizer.optimize(returns)

        # Should complete in reasonable time
        assert result['runtime'] < 1.0

    def test_edge_cases(self):
        """Test edge cases."""
        # Very high lambda should push towards equal weights
        prices = generate_synthetic_data(10)
        returns = compute_returns(prices, method='log')
        optimizer = L2RegularizedOptimizer(lambda_l2=100.0)
        result = optimizer.optimize(returns)

        weights = result['weights']
        # Should be relatively uniform
        assert weights.std() < 0.1

    def test_zero_lambda(self):
        """Test that lambda=0 works (should be like unregularized)."""
        prices = generate_synthetic_data(20)
        returns = compute_returns(prices, method='log')
        optimizer = L2RegularizedOptimizer(lambda_l2=0.0)
        result = optimizer.optimize(returns)

        assert np.isclose(result['weights'].sum(), 1.0, atol=1e-6)
