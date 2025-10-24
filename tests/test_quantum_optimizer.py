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

"""Tests for discrete levels portfolio optimizer."""

import pytest
import numpy as np
import pandas as pd
from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer


class TestDiscreteLevelsOptimizer:
    """Test suite for DiscreteLevelsOptimizer."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD']

        # Generate correlated returns
        mean_returns = np.random.randn(len(tickers)) * 0.001
        cov_matrix = np.random.rand(len(tickers), len(tickers))
        cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make symmetric
        cov_matrix += np.eye(len(tickers)) * 0.01  # Add diagonal

        returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, size=len(dates))
        returns = pd.DataFrame(returns_data, index=dates, columns=tickers)

        return returns

    @pytest.fixture
    def small_returns(self):
        """Create very small returns dataset for fast testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

        returns_data = np.random.randn(len(dates), len(tickers)) * 0.02
        returns = pd.DataFrame(returns_data, index=dates, columns=tickers)

        return returns

    def test_initialization_default(self):
        """Test optimizer initialization with default parameters."""
        optimizer = DiscreteLevelsOptimizer()

        assert optimizer.max_cluster_size == 19
        assert optimizer.n_levels == 6
        assert optimizer.alpha == 10.0
        assert optimizer.beta == 2.0
        assert optimizer.budget_penalty == 0.0
        assert optimizer.thermometer_penalty == 10.0
        assert optimizer.solver_type == 'simulated'
        assert optimizer.num_reads == 256

    def test_initialization_custom(self):
        """Test optimizer initialization with custom parameters."""
        optimizer = DiscreteLevelsOptimizer(
            max_cluster_size=10,
            n_levels=4,
            alpha=2.0,
            beta=0.5,
            solver_type='simulated',
            num_reads=100
        )

        assert optimizer.max_cluster_size == 10
        assert optimizer.n_levels == 4
        assert optimizer.alpha == 2.0
        assert optimizer.beta == 0.5
        assert optimizer.solver_type == 'simulated'
        assert optimizer.num_reads == 100

    def test_optimize_basic(self, small_returns):
        """Test basic optimization workflow."""
        optimizer = DiscreteLevelsOptimizer(
            solver_type='simulated',
            num_reads=100  # Reduce for speed
        )

        result = optimizer.optimize(small_returns)

        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'metrics' in result
        assert len(result['weights']) == len(small_returns.columns)
        assert np.isclose(result['weights'].sum(), 1.0, atol=0.01)
        assert all(result['weights'] >= 0)
        assert result['metrics']['runtime'] > 0

    def test_optimize_result_structure(self, small_returns):
        """Test that result contains all expected fields."""
        optimizer = DiscreteLevelsOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(small_returns)

        assert 'weights' in result
        assert 'metrics' in result
        metrics = result['metrics']
        assert 'return' in metrics
        assert 'risk' in metrics
        assert 'sharpe' in metrics
        assert 'n_clusters' in metrics
        assert 'runtime' in metrics

    def test_optimize_metrics(self, small_returns):
        """Test that portfolio metrics are calculated correctly."""
        optimizer = DiscreteLevelsOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(small_returns)

        metrics = result['metrics']
        weights = result['weights']

        assert metrics['return'] is not None
        assert metrics['risk'] > 0
        assert metrics['sharpe'] is not None

        # Manually verify metrics
        mu = small_returns.mean() * 252
        Sigma = small_returns.cov() * 252

        expected_return = (weights * mu).sum()
        volatility = np.sqrt(weights @ Sigma @ weights)

        assert np.isclose(metrics['return'], expected_return, rtol=0.01)
        assert np.isclose(metrics['risk'], volatility, rtol=0.01)

    def test_optimize_with_k_spread(self, small_returns):
        """Test optimization with k_spread parameter for concentration."""
        optimizer = DiscreteLevelsOptimizer(
            solver_type='simulated',
            num_reads=50,
            k_spread=1.0  # Apply spread adjustment
        )

        result = optimizer.optimize(small_returns)

        assert 'weights' in result
        assert np.isclose(result['weights'].sum(), 1.0, atol=0.01)

    def test_optimize_different_parameters(self, small_returns):
        """Test optimization with different parameter combinations."""
        param_sets = [
            {'alpha': 5.0, 'beta': 1.0},
            {'alpha': 1.0, 'beta': 5.0},
            {'n_levels': 4}
        ]
        results = {}

        for i, params in enumerate(param_sets):
            optimizer = DiscreteLevelsOptimizer(
                solver_type='simulated',
                num_reads=50,
                **params
            )
            result = optimizer.optimize(small_returns)
            results[i] = result

            assert 'weights' in result
            assert np.isclose(result['weights'].sum(), 1.0, atol=0.01)

        # All should produce valid portfolios
        for i, result in results.items():
            assert len(result['weights']) == len(small_returns.columns)

    def test_cluster_statistics(self, small_returns):
        """Test cluster statistics in result."""
        optimizer = DiscreteLevelsOptimizer(
            max_cluster_size=2,  # Force multiple clusters
            solver_type='simulated',
            num_reads=50
        )

        result = optimizer.optimize(small_returns)

        metrics = result['metrics']
        assert metrics['n_clusters'] > 0

    def test_solver_runtime(self, small_returns):
        """Test solver runtime tracking."""
        optimizer = DiscreteLevelsOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(small_returns)

        metrics = result['metrics']
        assert 'runtime' in metrics
        assert metrics['runtime'] > 0

    def test_optimizer_attributes(self):
        """Test optimizer attribute access."""
        optimizer = DiscreteLevelsOptimizer(
            max_cluster_size=15,
            n_levels=4,
            alpha=2.0
        )

        assert optimizer.max_cluster_size == 15
        assert optimizer.n_levels == 4
        assert optimizer.alpha == 2.0

    def test_aggressive_parameters(self, small_returns):
        """Test with aggressive parameters (high alpha, low beta)."""
        optimizer = DiscreteLevelsOptimizer(
            solver_type='simulated',
            num_reads=50,
            alpha=20.0,  # High return weight
            beta=0.5     # Low risk weight
        )

        result = optimizer.optimize(small_returns)
        assert 'weights' in result

    def test_conservative_parameters(self, small_returns):
        """Test with conservative parameters (low alpha, high beta)."""
        optimizer = DiscreteLevelsOptimizer(
            solver_type='simulated',
            num_reads=50,
            alpha=1.0,   # Low return weight
            beta=10.0    # High risk weight
        )

        result = optimizer.optimize(small_returns)
        assert 'weights' in result

    def test_small_cluster_size(self, small_returns):
        """Test with very small cluster size to force many clusters."""
        optimizer = DiscreteLevelsOptimizer(
            max_cluster_size=2,
            solver_type='simulated',
            num_reads=50
        )

        result = optimizer.optimize(small_returns)

        assert 'weights' in result
        # 4 assets with max_size=2 should create 2 clusters
        assert result['metrics']['n_clusters'] == 2

    def test_large_cluster_size(self, small_returns):
        """Test with large cluster size (single cluster)."""
        optimizer = DiscreteLevelsOptimizer(
            max_cluster_size=100,  # Larger than dataset
            solver_type='simulated',
            num_reads=50
        )

        result = optimizer.optimize(small_returns)

        assert 'weights' in result
        # All assets should fit in one cluster
        assert result['metrics']['n_clusters'] == 1

    def test_different_n_levels(self, small_returns):
        """Test with different discretization levels."""
        for n_levels in [3, 4, 6]:
            optimizer = DiscreteLevelsOptimizer(
                n_levels=n_levels,
                solver_type='simulated',
                num_reads=50
            )

            result = optimizer.optimize(small_returns)
            assert 'weights' in result

    def test_zero_variance_asset(self):
        """Test handling of asset with zero variance."""
        returns = pd.DataFrame({
            'AAPL': np.random.randn(50) * 0.02,
            'CONST': np.zeros(50),  # Zero variance
            'MSFT': np.random.randn(50) * 0.02
        })

        optimizer = DiscreteLevelsOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(returns)

        # Should handle gracefully
        assert 'weights' in result or 'error' in result  # Either is acceptable

    def test_negative_returns(self):
        """Test with predominantly negative returns."""
        np.random.seed(42)
        returns = pd.DataFrame({
            'STOCK1': -np.abs(np.random.randn(50) * 0.02),
            'STOCK2': -np.abs(np.random.randn(50) * 0.02),
            'STOCK3': -np.abs(np.random.randn(50) * 0.02)
        })

        optimizer = DiscreteLevelsOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(returns)

        # Should complete even with negative returns
        assert 'weights' in result

    def test_minimal_dataset(self):
        """Test with minimal dataset (2 assets, short history)."""
        returns = pd.DataFrame({
            'AAPL': [0.01, -0.01, 0.02, -0.005, 0.01],
            'MSFT': [-0.01, 0.02, -0.01, 0.01, -0.005]
        })

        optimizer = DiscreteLevelsOptimizer(solver_type='simulated', num_reads=20)
        result = optimizer.optimize(returns)

        assert isinstance(result, dict)
        # May succeed or fail, but should not crash

    def test_highly_correlated_assets(self):
        """Test with highly correlated assets."""
        np.random.seed(42)
        base = np.random.randn(50) * 0.02

        returns = pd.DataFrame({
            'AAPL': base + np.random.randn(50) * 0.001,
            'MSFT': base + np.random.randn(50) * 0.001,
            'GOOGL': base + np.random.randn(50) * 0.001
        })

        optimizer = DiscreteLevelsOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(returns)

        assert 'weights' in result
        # Should cluster all together
        assert result['metrics']['n_clusters'] == 1

    def test_uncorrelated_assets(self):
        """Test with completely uncorrelated assets."""
        np.random.seed(42)
        returns = pd.DataFrame({
            'AAPL': np.random.randn(50) * 0.02,
            'MSFT': np.random.randn(50) * 0.02,
            'GOOGL': np.random.randn(50) * 0.02,
            'AMZN': np.random.randn(50) * 0.02
        })

        optimizer = DiscreteLevelsOptimizer(
            max_cluster_size=2,
            solver_type='simulated',
            num_reads=50
        )
        result = optimizer.optimize(returns)

        assert 'weights' in result

    @pytest.mark.parametrize("n_levels", [3, 4, 6])
    def test_all_level_counts_valid(self, small_returns, n_levels):
        """Test that all discretization levels produce valid portfolios."""
        optimizer = DiscreteLevelsOptimizer(
            solver_type='simulated',
            num_reads=50,
            n_levels=n_levels
        )

        result = optimizer.optimize(small_returns)

        assert 'weights' in result
        assert np.isclose(result['weights'].sum(), 1.0, atol=0.01)
        assert all(result['weights'] >= 0)
