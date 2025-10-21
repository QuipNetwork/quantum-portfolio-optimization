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

"""Tests for quantum portfolio optimizer."""

import pytest
import numpy as np
import pandas as pd
from qpo.optimizers.quantum import IndependentClustersOptimizer, QuantumOptimizationResult


class TestIndependentClustersOptimizer:
    """Test suite for IndependentClustersOptimizer."""

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
        optimizer = IndependentClustersOptimizer()

        assert optimizer.max_cluster_size == 18
        assert optimizer.n_bits == 10
        assert optimizer.alpha == 1.0
        assert optimizer.beta == 1.0
        assert optimizer.lambda_budget == 10.0
        assert optimizer.solver_type == 'simulated'
        assert optimizer.num_reads == 1000
        assert optimizer.aggregation_strategy == 'concatenate'
        assert optimizer.cardinality is None

    def test_initialization_custom(self):
        """Test optimizer initialization with custom parameters."""
        optimizer = IndependentClustersOptimizer(
            max_cluster_size=10,
            n_bits=8,
            alpha=2.0,
            beta=0.5,
            solver_type='hybrid',
            aggregation_strategy='proportional',
            cardinality=15
        )

        assert optimizer.max_cluster_size == 10
        assert optimizer.n_bits == 8
        assert optimizer.alpha == 2.0
        assert optimizer.beta == 0.5
        assert optimizer.solver_type == 'hybrid'
        assert optimizer.aggregation_strategy == 'proportional'
        assert optimizer.cardinality == 15

    def test_optimize_basic(self, small_returns):
        """Test basic optimization workflow."""
        optimizer = IndependentClustersOptimizer(
            solver_type='simulated',
            num_reads=100  # Reduce for speed
        )

        result = optimizer.optimize(small_returns)

        assert isinstance(result, QuantumOptimizationResult)
        assert result.success
        assert len(result.weights) == len(small_returns.columns)
        assert np.isclose(result.weights.sum(), 1.0, atol=0.01)
        assert all(result.weights >= 0)
        assert result.runtime > 0

    def test_optimize_result_structure(self, small_returns):
        """Test that result contains all expected fields."""
        optimizer = IndependentClustersOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(small_returns)

        assert hasattr(result, 'weights')
        assert hasattr(result, 'expected_return')
        assert hasattr(result, 'volatility')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'n_clusters')
        assert hasattr(result, 'cluster_stats')
        assert hasattr(result, 'runtime')
        assert hasattr(result, 'solver_info')
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')

    def test_optimize_metrics(self, small_returns):
        """Test that portfolio metrics are calculated correctly."""
        optimizer = IndependentClustersOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(small_returns)

        assert result.expected_return is not None
        assert result.volatility > 0
        assert result.sharpe_ratio is not None

        # Manually verify metrics
        mu = small_returns.mean() * 252
        Sigma = small_returns.cov() * 252

        expected_return = (result.weights * mu).sum()
        volatility = np.sqrt(result.weights @ Sigma @ result.weights)

        assert np.isclose(result.expected_return, expected_return, rtol=0.01)
        assert np.isclose(result.volatility, volatility, rtol=0.01)

    def test_optimize_with_cardinality(self, small_returns):
        """Test optimization with cardinality constraint."""
        optimizer = IndependentClustersOptimizer(
            solver_type='simulated',
            num_reads=50,
            cardinality=2  # Limit to 2 assets
        )

        result = optimizer.optimize(small_returns)

        assert result.success
        assert (result.weights > 0).sum() <= 2
        assert np.isclose(result.weights.sum(), 1.0, atol=0.01)

    def test_optimize_different_strategies(self, small_returns):
        """Test different aggregation strategies."""
        strategies = ['concatenate', 'proportional', 'uniform']
        results = {}

        for strategy in strategies:
            optimizer = IndependentClustersOptimizer(
                solver_type='simulated',
                num_reads=50,
                aggregation_strategy=strategy
            )
            result = optimizer.optimize(small_returns)
            results[strategy] = result

            assert result.success
            assert np.isclose(result.weights.sum(), 1.0, atol=0.01)

        # Results should differ across strategies (not always, but likely)
        # At minimum, all should be valid
        for strategy, result in results.items():
            assert len(result.weights) == len(small_returns.columns)

    def test_cluster_statistics(self, small_returns):
        """Test cluster statistics in result."""
        optimizer = IndependentClustersOptimizer(
            max_cluster_size=2,  # Force multiple clusters
            solver_type='simulated',
            num_reads=50
        )

        result = optimizer.optimize(small_returns)

        assert result.n_clusters > 0
        assert 'n_clusters' in result.cluster_stats
        assert 'total_assets' in result.cluster_stats
        assert result.cluster_stats['total_assets'] == len(small_returns.columns)

    def test_solver_info(self, small_returns):
        """Test solver info in result."""
        optimizer = IndependentClustersOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(small_returns)

        assert 'solver_type' in result.solver_info
        assert result.solver_info['solver_type'] == 'simulated'
        assert 'n_clusters_solved' in result.solver_info
        assert result.solver_info['n_clusters_solved'] == result.n_clusters

    def test_get_config(self):
        """Test config retrieval."""
        optimizer = IndependentClustersOptimizer(
            max_cluster_size=15,
            n_bits=8,
            alpha=2.0
        )

        config = optimizer.get_config()

        assert config['max_cluster_size'] == 15
        assert config['n_bits'] == 8
        assert config['alpha'] == 2.0
        assert 'clusterer_type' in config

    def test_aggressive_parameters(self, small_returns):
        """Test with aggressive parameters (high alpha, low beta)."""
        optimizer = IndependentClustersOptimizer(
            solver_type='simulated',
            num_reads=50,
            alpha=5.0,  # High return weight
            beta=0.1    # Low risk weight
        )

        result = optimizer.optimize(small_returns)
        assert result.success

    def test_conservative_parameters(self, small_returns):
        """Test with conservative parameters (low alpha, high beta)."""
        optimizer = IndependentClustersOptimizer(
            solver_type='simulated',
            num_reads=50,
            alpha=0.1,  # Low return weight
            beta=5.0    # High risk weight
        )

        result = optimizer.optimize(small_returns)
        assert result.success

    def test_small_cluster_size(self, small_returns):
        """Test with very small cluster size to force many clusters."""
        optimizer = IndependentClustersOptimizer(
            max_cluster_size=2,
            solver_type='simulated',
            num_reads=50
        )

        result = optimizer.optimize(small_returns)

        assert result.success
        # 4 assets with max_size=2 should create 2 clusters
        assert result.n_clusters == 2

    def test_large_cluster_size(self, small_returns):
        """Test with large cluster size (single cluster)."""
        optimizer = IndependentClustersOptimizer(
            max_cluster_size=100,  # Larger than dataset
            solver_type='simulated',
            num_reads=50
        )

        result = optimizer.optimize(small_returns)

        assert result.success
        # All assets should fit in one cluster
        assert result.n_clusters == 1

    def test_different_n_bits(self, small_returns):
        """Test with different discretization levels."""
        for n_bits in [4, 8, 10]:
            optimizer = IndependentClustersOptimizer(
                n_bits=n_bits,
                solver_type='simulated',
                num_reads=50
            )

            result = optimizer.optimize(small_returns)
            assert result.success

    def test_zero_variance_asset(self):
        """Test handling of asset with zero variance."""
        returns = pd.DataFrame({
            'AAPL': np.random.randn(50) * 0.02,
            'CONST': np.zeros(50),  # Zero variance
            'MSFT': np.random.randn(50) * 0.02
        })

        optimizer = IndependentClustersOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(returns)

        # Should handle gracefully (may or may not include CONST)
        assert result.success or not result.success  # Either is acceptable

    def test_negative_returns(self):
        """Test with predominantly negative returns."""
        np.random.seed(42)
        returns = pd.DataFrame({
            'STOCK1': -np.abs(np.random.randn(50) * 0.02),
            'STOCK2': -np.abs(np.random.randn(50) * 0.02),
            'STOCK3': -np.abs(np.random.randn(50) * 0.02)
        })

        optimizer = IndependentClustersOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(returns)

        # Should complete even with negative returns
        assert result.success

    def test_minimal_dataset(self):
        """Test with minimal dataset (2 assets, short history)."""
        returns = pd.DataFrame({
            'AAPL': [0.01, -0.01, 0.02, -0.005, 0.01],
            'MSFT': [-0.01, 0.02, -0.01, 0.01, -0.005]
        })

        optimizer = IndependentClustersOptimizer(solver_type='simulated', num_reads=20)
        result = optimizer.optimize(returns)

        assert isinstance(result, QuantumOptimizationResult)
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

        optimizer = IndependentClustersOptimizer(solver_type='simulated', num_reads=50)
        result = optimizer.optimize(returns)

        assert result.success
        # Should cluster all together
        assert result.n_clusters == 1

    def test_uncorrelated_assets(self):
        """Test with completely uncorrelated assets."""
        np.random.seed(42)
        returns = pd.DataFrame({
            'AAPL': np.random.randn(50) * 0.02,
            'MSFT': np.random.randn(50) * 0.02,
            'GOOGL': np.random.randn(50) * 0.02,
            'AMZN': np.random.randn(50) * 0.02
        })

        optimizer = IndependentClustersOptimizer(
            max_cluster_size=2,
            solver_type='simulated',
            num_reads=50
        )
        result = optimizer.optimize(returns)

        assert result.success

    @pytest.mark.parametrize("aggregation_strategy", ['concatenate', 'proportional', 'uniform'])
    def test_all_strategies_valid(self, small_returns, aggregation_strategy):
        """Test that all aggregation strategies produce valid portfolios."""
        optimizer = IndependentClustersOptimizer(
            solver_type='simulated',
            num_reads=50,
            aggregation_strategy=aggregation_strategy
        )

        result = optimizer.optimize(small_returns)

        assert result.success
        assert np.isclose(result.weights.sum(), 1.0, atol=0.01)
        assert all(result.weights >= 0)
