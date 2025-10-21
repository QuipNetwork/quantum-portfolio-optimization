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

"""Test suite for equal-weight portfolio optimizer."""

import pytest
import numpy as np
import pandas as pd
from qpo.optimizers.equal_weight import EqualWeightOptimizer
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


class TestEqualWeight:
    """Test equal-weight portfolio optimizer."""

    @pytest.mark.parametrize("n_assets", [16, 32, 64])
    def test_equal_weight(self, n_assets):
        """Test equal-weight portfolio (1/N)."""
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = EqualWeightOptimizer()
        result = optimizer.optimize(returns)

        # Validate
        assert 'weights' in result
        weights = result['weights']

        # All weights should be equal
        expected_weight = 1.0 / n_assets
        assert np.allclose(weights.values, expected_weight, atol=1e-10)
        assert np.isclose(weights.sum(), 1.0)

        # Should be instant
        assert result['runtime'] < 0.001

        print(f"\n{'='*70}")
        print(f"Equal-Weight Portfolio - {n_assets} Assets")
        print(f"{'='*70}")
        print(f"  Weight per asset: {expected_weight*100:.4f}%")
        print(f"  Sharpe ratio:     {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:      {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Runtime:          {result['runtime']:.6f}s")

    def test_deterministic(self):
        """Test that equal-weight always produces the same results."""
        prices = generate_synthetic_data(10)
        returns = compute_returns(prices, method='log')

        optimizer = EqualWeightOptimizer()
        result1 = optimizer.optimize(returns)
        result2 = optimizer.optimize(returns)

        # Should be identical
        assert np.array_equal(result1['weights'].values, result2['weights'].values)
        assert result1['metrics']['sharpe_ratio'] == result2['metrics']['sharpe_ratio']

    def test_edge_cases(self):
        """Test edge cases."""
        # Single asset
        prices = generate_synthetic_data(1)
        returns = compute_returns(prices, method='log')
        optimizer = EqualWeightOptimizer()
        result = optimizer.optimize(returns)
        assert result['weights'].values[0] == 1.0

        # Two assets
        prices = generate_synthetic_data(2)
        returns = compute_returns(prices, method='log')
        result = optimizer.optimize(returns)
        assert np.allclose(result['weights'].values, 0.5)

    def test_metrics(self):
        """Test that all expected metrics are present."""
        prices = generate_synthetic_data(20)
        returns = compute_returns(prices, method='log')
        optimizer = EqualWeightOptimizer()
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

        # For equal-weight, effective_n should equal n_assets (within numerical precision)
        assert np.isclose(metrics['effective_n_assets'], metrics['n_assets'], atol=1e-10)
