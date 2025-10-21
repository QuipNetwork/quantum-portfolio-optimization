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

"""Tests for QUBO aggregation strategies."""

import pytest
import numpy as np
import pandas as pd
from qpo.qubo.aggregation import ClusterAggregator, AggregationStrategy


class TestClusterAggregator:
    """Test suite for ClusterAggregator."""

    @pytest.fixture
    def sample_cluster_weights(self):
        """Create sample cluster weights for testing."""
        return {
            'cluster_0': pd.Series({'AAPL': 0.3, 'MSFT': 0.5}),
            'cluster_1': pd.Series({'GOOGL': 0.4, 'AMZN': 0.2, 'TSLA': 0.1})
        }

    @pytest.fixture
    def sample_clusters(self):
        """Create sample cluster definitions."""
        return {
            'cluster_0': ['AAPL', 'MSFT'],
            'cluster_1': ['GOOGL', 'AMZN', 'TSLA']
        }

    def test_initialization_valid(self):
        """Test initialization with valid strategies."""
        for strategy in ['concatenate', 'proportional', 'uniform']:
            aggregator = ClusterAggregator(strategy)
            assert aggregator.strategy == strategy

    def test_initialization_invalid(self):
        """Test initialization with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            ClusterAggregator('invalid_strategy')

    def test_concatenate_strategy(self, sample_cluster_weights, sample_clusters):
        """Test concatenate aggregation strategy."""
        aggregator = ClusterAggregator('concatenate')
        weights = aggregator.aggregate(sample_cluster_weights, sample_clusters)

        # Should combine all weights and normalize
        assert len(weights) == 5
        assert np.isclose(weights.sum(), 1.0)

        # Check relative proportions preserved
        # Original: AAPL=0.3, MSFT=0.5, total_0=0.8
        #           GOOGL=0.4, AMZN=0.2, TSLA=0.1, total_1=0.7
        #           global_total=1.5
        # Normalized: AAPL=0.3/1.5=0.2, MSFT=0.5/1.5≈0.333, etc.
        assert np.isclose(weights['AAPL'], 0.3 / 1.5)
        assert np.isclose(weights['MSFT'], 0.5 / 1.5)
        assert np.isclose(weights['GOOGL'], 0.4 / 1.5)

    def test_proportional_strategy(self, sample_cluster_weights, sample_clusters):
        """Test proportional aggregation strategy."""
        aggregator = ClusterAggregator('proportional')
        weights = aggregator.aggregate(sample_cluster_weights, sample_clusters)

        # Cluster 0: 2 assets (40% of total 5 assets) → gets 0.4 budget
        # Cluster 1: 3 assets (60% of total 5 assets) → gets 0.6 budget

        assert len(weights) == 5
        assert np.isclose(weights.sum(), 1.0)

        # Cluster 0 should get 2/5 = 0.4 of budget
        cluster_0_total = weights['AAPL'] + weights['MSFT']
        assert np.isclose(cluster_0_total, 0.4)

        # Cluster 1 should get 3/5 = 0.6 of budget
        cluster_1_total = weights['GOOGL'] + weights['AMZN'] + weights['TSLA']
        assert np.isclose(cluster_1_total, 0.6)

        # Within cluster, proportions preserved
        # Cluster 0: AAPL=0.3, MSFT=0.5, ratio = 3:5
        assert np.isclose(weights['AAPL'] / weights['MSFT'], 0.3 / 0.5)

    def test_uniform_strategy(self, sample_cluster_weights, sample_clusters):
        """Test uniform aggregation strategy."""
        aggregator = ClusterAggregator('uniform')
        weights = aggregator.aggregate(sample_cluster_weights, sample_clusters)

        # Each cluster gets equal budget (0.5 each)
        assert len(weights) == 5
        assert np.isclose(weights.sum(), 1.0)

        cluster_0_total = weights['AAPL'] + weights['MSFT']
        cluster_1_total = weights['GOOGL'] + weights['AMZN'] + weights['TSLA']

        assert np.isclose(cluster_0_total, 0.5)
        assert np.isclose(cluster_1_total, 0.5)

    def test_custom_budget(self, sample_cluster_weights, sample_clusters):
        """Test aggregation with custom target budget."""
        aggregator = ClusterAggregator('concatenate')
        weights = aggregator.aggregate(
            sample_cluster_weights,
            sample_clusters,
            target_budget=2.0
        )

        assert np.isclose(weights.sum(), 2.0)

    def test_cardinality_constraint(self, sample_cluster_weights, sample_clusters):
        """Test cardinality constraint."""
        aggregator = ClusterAggregator('concatenate')

        # Limit to top 3 assets
        weights = aggregator.aggregate(
            sample_cluster_weights,
            sample_clusters,
            cardinality=3
        )

        assert (weights > 0).sum() == 3
        assert np.isclose(weights.sum(), 1.0)

        # Check that top 3 by weight are selected
        original_weights = aggregator.aggregate(sample_cluster_weights, sample_clusters)
        top_3 = original_weights.nlargest(3).index
        assert all(weights[ticker] > 0 for ticker in top_3)

    def test_cardinality_zero(self, sample_cluster_weights, sample_clusters):
        """Test cardinality constraint with k=0."""
        aggregator = ClusterAggregator('concatenate')
        weights = aggregator.aggregate(
            sample_cluster_weights,
            sample_clusters,
            cardinality=0
        )

        # Should return original (no constraint)
        assert len(weights) == 5

    def test_cardinality_larger_than_assets(self, sample_cluster_weights, sample_clusters):
        """Test cardinality constraint larger than number of assets."""
        aggregator = ClusterAggregator('concatenate')
        weights = aggregator.aggregate(
            sample_cluster_weights,
            sample_clusters,
            cardinality=100
        )

        # Should return all assets
        assert (weights > 0).sum() == 5

    def test_empty_cluster(self):
        """Test handling of clusters with zero weights."""
        cluster_weights = {
            'cluster_0': pd.Series({'AAPL': 0.0, 'MSFT': 0.0}),
            'cluster_1': pd.Series({'GOOGL': 0.5})
        }
        clusters = {
            'cluster_0': ['AAPL', 'MSFT'],
            'cluster_1': ['GOOGL']
        }

        aggregator = ClusterAggregator('concatenate')
        weights = aggregator.aggregate(cluster_weights, clusters)

        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
        assert weights['AAPL'] == 0.0
        assert weights['MSFT'] == 0.0
        assert np.isclose(weights['GOOGL'], 1.0)

    def test_single_cluster(self):
        """Test aggregation with single cluster."""
        cluster_weights = {
            'cluster_0': pd.Series({'AAPL': 0.3, 'MSFT': 0.7})
        }
        clusters = {
            'cluster_0': ['AAPL', 'MSFT']
        }

        # All strategies should produce same result with single cluster
        for strategy in ['concatenate', 'proportional', 'uniform']:
            aggregator = ClusterAggregator(strategy)
            weights = aggregator.aggregate(cluster_weights, clusters)

            assert len(weights) == 2
            assert np.isclose(weights.sum(), 1.0)
            assert np.isclose(weights['AAPL'], 0.3)
            assert np.isclose(weights['MSFT'], 0.7)

    def test_get_allocation_info(self, sample_cluster_weights, sample_clusters):
        """Test allocation info generation."""
        aggregator = ClusterAggregator('concatenate')
        info = aggregator.get_allocation_info(sample_cluster_weights, sample_clusters)

        assert len(info) == 2
        assert 'cluster_id' in info.columns
        assert 'n_assets' in info.columns
        assert 'total_weight' in info.columns

        # Check cluster 0
        c0_info = info[info['cluster_id'] == 'cluster_0'].iloc[0]
        assert c0_info['n_assets'] == 2
        assert c0_info['n_nonzero'] == 2
        assert np.isclose(c0_info['total_weight'], 0.8)

        # Check cluster 1
        c1_info = info[info['cluster_id'] == 'cluster_1'].iloc[0]
        assert c1_info['n_assets'] == 3
        assert c1_info['n_nonzero'] == 3
        assert np.isclose(c1_info['total_weight'], 0.7)

    def test_compare_strategies(self, sample_cluster_weights, sample_clusters):
        """Test strategy comparison."""
        aggregator = ClusterAggregator('concatenate')
        results = aggregator.compare_strategies(sample_cluster_weights, sample_clusters)

        assert len(results) == 3
        assert 'concatenate' in results
        assert 'proportional' in results
        assert 'uniform' in results

        # All should sum to 1.0
        for strategy, weights in results.items():
            assert np.isclose(weights.sum(), 1.0)

        # Proportional and uniform should differ
        assert not np.allclose(
            results['proportional'].values,
            results['uniform'].values
        )

    def test_get_cluster_contributions(self, sample_cluster_weights, sample_clusters):
        """Test cluster contribution calculation."""
        aggregator = ClusterAggregator('uniform')
        weights = aggregator.aggregate(sample_cluster_weights, sample_clusters)
        contributions = aggregator.get_cluster_contributions(weights, sample_clusters)

        assert len(contributions) == 2
        # With uniform strategy, each cluster gets 0.5
        assert np.isclose(contributions['cluster_0'], 0.5)
        assert np.isclose(contributions['cluster_1'], 0.5)

    def test_unbalanced_clusters(self):
        """Test with highly unbalanced cluster sizes."""
        cluster_weights = {
            'cluster_0': pd.Series({'AAPL': 0.5}),  # 1 asset
            'cluster_1': pd.Series({f'STOCK_{i}': 0.1 for i in range(10)})  # 10 assets
        }
        clusters = {
            'cluster_0': ['AAPL'],
            'cluster_1': [f'STOCK_{i}' for i in range(10)]
        }

        # Proportional should heavily favor cluster 1
        prop = ClusterAggregator('proportional')
        prop_weights = prop.aggregate(cluster_weights, clusters)

        c0_budget = prop_weights['AAPL']
        c1_budget = sum(prop_weights[f'STOCK_{i}'] for i in range(10))

        # Cluster 0: 1/11 ≈ 0.091, Cluster 1: 10/11 ≈ 0.909
        assert np.isclose(c0_budget, 1/11)
        assert np.isclose(c1_budget, 10/11)

        # Uniform should give equal budget
        unif = ClusterAggregator('uniform')
        unif_weights = unif.aggregate(cluster_weights, clusters)

        c0_budget = unif_weights['AAPL']
        c1_budget = sum(unif_weights[f'STOCK_{i}'] for i in range(10))

        assert np.isclose(c0_budget, 0.5)
        assert np.isclose(c1_budget, 0.5)

    def test_all_zero_weights(self):
        """Test handling of all-zero cluster weights."""
        cluster_weights = {
            'cluster_0': pd.Series({'AAPL': 0.0, 'MSFT': 0.0}),
            'cluster_1': pd.Series({'GOOGL': 0.0})
        }
        clusters = {
            'cluster_0': ['AAPL', 'MSFT'],
            'cluster_1': ['GOOGL']
        }

        aggregator = ClusterAggregator('concatenate')
        weights = aggregator.aggregate(cluster_weights, clusters)

        # Should handle gracefully
        assert len(weights) == 3
        assert all(weights == 0.0)

    def test_many_clusters(self):
        """Test with many clusters."""
        n_clusters = 10
        cluster_weights = {
            f'cluster_{i}': pd.Series({f'STOCK_{i}': 1.0})
            for i in range(n_clusters)
        }
        clusters = {
            f'cluster_{i}': [f'STOCK_{i}']
            for i in range(n_clusters)
        }

        # Uniform should give each cluster 0.1
        aggregator = ClusterAggregator('uniform')
        weights = aggregator.aggregate(cluster_weights, clusters)

        assert len(weights) == n_clusters
        assert np.isclose(weights.sum(), 1.0)
        assert all(np.isclose(w, 0.1) for w in weights.values)
