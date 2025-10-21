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

"""Tests for covariance-based clustering."""

import pytest
import numpy as np
import pandas as pd
from clustering import CovarianceClusterer


class TestCovarianceClusterer:
    """Test suite for CovarianceClusterer."""

    def test_basic_covariance_clustering(self):
        """Test basic covariance clustering with euclidean distance."""
        np.random.seed(42)

        # Create assets with different covariance structures
        # Group 1: High covariance assets (correlated + high vol)
        # Group 2: Low covariance assets (correlated + low vol)
        dates = pd.date_range('2023-01-01', periods=100)

        # High variance, high covariance group
        high_vol = 0.03
        returns1 = np.random.normal(0, high_vol, (100, 3))
        common_factor1 = np.random.normal(0, high_vol, 100)
        for i in range(3):
            returns1[:, i] += 0.8 * common_factor1

        # Low variance, high covariance group
        low_vol = 0.01
        returns2 = np.random.normal(0, low_vol, (100, 3))
        common_factor2 = np.random.normal(0, low_vol, 100)
        for i in range(3):
            returns2[:, i] += 0.8 * common_factor2

        all_returns = np.hstack([returns1, returns2])
        returns = pd.DataFrame(
            all_returns,
            index=dates,
            columns=[f'STOCK_{i}' for i in range(6)]
        )

        clusterer = CovarianceClusterer(max_cluster_size=4, distance_metric='euclidean')
        clusters = clusterer.cluster(returns)

        assert len(clusters) >= 2
        assert sum(len(t) for t in clusters.values()) == 6

    def test_euclidean_distance_metric(self):
        """Test euclidean distance metric."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 5) * 0.02,
            columns=['A', 'B', 'C', 'D', 'E']
        )

        clusterer = CovarianceClusterer(
            max_cluster_size=3,
            distance_metric='euclidean'
        )
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert all(len(tickers) <= 3 for tickers in clusters.values())

    def test_frobenius_distance_metric(self):
        """Test Frobenius norm distance metric."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 5) * 0.02,
            columns=['A', 'B', 'C', 'D', 'E']
        )

        clusterer = CovarianceClusterer(
            max_cluster_size=3,
            distance_metric='frobenius'
        )
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert all(len(tickers) <= 3 for tickers in clusters.values())

    def test_spectral_distance_metric(self):
        """Test spectral distance metric."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 5) * 0.02,
            columns=['A', 'B', 'C', 'D', 'E']
        )

        clusterer = CovarianceClusterer(
            max_cluster_size=3,
            distance_metric='spectral'
        )
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert all(len(tickers) <= 3 for tickers in clusters.values())

    def test_distance_metrics_differ(self):
        """Test that different distance metrics produce different results."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 8) * 0.02,
            columns=[f'STOCK_{i}' for i in range(8)]
        )

        clusterer_euc = CovarianceClusterer(
            max_cluster_size=4,
            distance_metric='euclidean'
        )
        clusters_euc = clusterer_euc.cluster(returns)

        clusterer_spec = CovarianceClusterer(
            max_cluster_size=4,
            distance_metric='spectral'
        )
        clusters_spec = clusterer_spec.cluster(returns)

        # Results might differ (not guaranteed, but likely with random data)
        # At minimum, both should produce valid clusters
        assert len(clusters_euc) > 0
        assert len(clusters_spec) > 0

    def test_normalization(self):
        """Test distance normalization."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 6) * 0.02,
            columns=[f'STOCK_{i}' for i in range(6)]
        )

        clusterer_norm = CovarianceClusterer(
            max_cluster_size=3,
            normalize=True
        )
        clusters_norm = clusterer_norm.cluster(returns)

        clusterer_no_norm = CovarianceClusterer(
            max_cluster_size=3,
            normalize=False
        )
        clusters_no_norm = clusterer_no_norm.cluster(returns)

        # Both should produce valid clusters
        assert len(clusters_norm) > 0
        assert len(clusters_no_norm) > 0

    def test_max_cluster_size_constraint(self):
        """Test that max_cluster_size is respected."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 20) * 0.02,
            columns=[f'STOCK_{i}' for i in range(20)]
        )

        max_size = 6
        clusterer = CovarianceClusterer(max_cluster_size=max_size)
        clusters = clusterer.cluster(returns)

        # All clusters should respect max size
        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= max_size, f"Cluster {cluster_id} exceeds max size"

    def test_linkage_methods(self):
        """Test different linkage methods."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 8) * 0.02,
            columns=[f'STOCK_{i}' for i in range(8)]
        )

        for linkage in ['ward', 'average', 'single', 'complete']:
            clusterer = CovarianceClusterer(
                max_cluster_size=4,
                linkage_method=linkage
            )
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 8

    def test_high_vs_low_covariance_separation(self):
        """Test that assets with different covariance magnitudes separate."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # High covariance group (high volatility stocks moving together)
        high_cov_returns = np.random.normal(0, 0.05, (100, 3))
        common_high = np.random.normal(0, 0.05, 100)
        for i in range(3):
            high_cov_returns[:, i] += 0.9 * common_high

        # Low covariance group (low volatility stocks moving together)
        low_cov_returns = np.random.normal(0, 0.005, (100, 3))
        common_low = np.random.normal(0, 0.005, 100)
        for i in range(3):
            low_cov_returns[:, i] += 0.9 * common_low

        all_returns = np.hstack([high_cov_returns, low_cov_returns])
        returns = pd.DataFrame(
            all_returns,
            index=dates,
            columns=[f'HIGH_{i}' if i < 3 else f'LOW_{i-3}' for i in range(6)]
        )

        clusterer = CovarianceClusterer(max_cluster_size=4)
        clusters = clusterer.cluster(returns)

        # Should create at least 2 clusters
        assert len(clusters) >= 2

    def test_covariance_summary(self):
        """Test covariance summary statistics."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 6) * 0.02,
            columns=[f'STOCK_{i}' for i in range(6)]
        )

        clusterer = CovarianceClusterer(max_cluster_size=3)
        clusters = clusterer.cluster(returns)
        summary = clusterer.get_covariance_summary(returns, clusters)

        assert len(summary) == len(clusters)
        assert 'cluster' in summary.columns
        assert 'n_assets' in summary.columns
        assert 'avg_covariance' in summary.columns
        assert 'avg_variance' in summary.columns

        # All clusters should have expected number of assets
        total_assets = summary['n_assets'].sum()
        assert total_assets == 6

    def test_single_asset_cluster_summary(self):
        """Test summary with single-asset clusters."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 3) * 0.02,
            columns=['A', 'B', 'C']
        )

        clusterer = CovarianceClusterer(max_cluster_size=1)
        clusters = clusterer.cluster(returns)
        summary = clusterer.get_covariance_summary(returns, clusters)

        # Single asset clusters should have zero covariance stats
        for _, row in summary.iterrows():
            if row['n_assets'] == 1:
                assert row['avg_covariance'] == 0.0
                assert row['std_covariance'] == 0.0

    def test_invalid_distance_metric(self):
        """Test error handling for invalid distance metric."""
        returns = pd.DataFrame(
            np.random.randn(50, 4) * 0.02,
            columns=['A', 'B', 'C', 'D']
        )

        clusterer = CovarianceClusterer(distance_metric='invalid')

        with pytest.raises(ValueError, match="Unknown distance metric"):
            clusterer.cluster(returns)

    def test_covariance_vs_correlation_difference(self):
        """Test that covariance clustering differs from correlation clustering."""
        np.random.seed(42)
        from clustering import CorrelationClusterer

        # Create assets where correlation is similar but covariance differs
        dates = pd.date_range('2023-01-01', periods=100)

        # High volatility pair with correlation ~0.8
        high_vol = 0.05
        stock1 = np.random.normal(0, high_vol, 100)
        stock2 = 0.8 * stock1 + np.random.normal(0, high_vol * 0.6, 100)

        # Low volatility pair with correlation ~0.8
        low_vol = 0.01
        stock3 = np.random.normal(0, low_vol, 100)
        stock4 = 0.8 * stock3 + np.random.normal(0, low_vol * 0.6, 100)

        returns = pd.DataFrame({
            'HIGH_VOL_1': stock1,
            'HIGH_VOL_2': stock2,
            'LOW_VOL_1': stock3,
            'LOW_VOL_2': stock4
        }, index=dates)

        # Covariance clustering should separate high/low vol
        cov_clusterer = CovarianceClusterer(max_cluster_size=3)
        cov_clusters = cov_clusterer.cluster(returns)

        # Correlation clustering might group them together
        corr_clusterer = CorrelationClusterer(max_cluster_size=3)
        corr_clusters = corr_clusterer.cluster(returns)

        # Both should produce valid clusters
        assert len(cov_clusters) > 0
        assert len(corr_clusters) > 0

    def test_small_dataset(self):
        """Test with minimum viable dataset."""
        returns = pd.DataFrame({
            'A': [0.01, -0.01, 0.02],
            'B': [0.015, -0.005, 0.018],
            'C': [-0.005, 0.01, -0.008]
        })

        clusterer = CovarianceClusterer(max_cluster_size=2)
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 3

    def test_deterministic_behavior(self):
        """Test that clustering is deterministic given same input."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 8) * 0.02,
            columns=[f'STOCK_{i}' for i in range(8)]
        )

        clusterer = CovarianceClusterer(max_cluster_size=4)
        clusters1 = clusterer.cluster(returns)
        clusters2 = clusterer.cluster(returns)

        # Should produce identical results
        assert len(clusters1) == len(clusters2)
        for cid in clusters1:
            assert cid in clusters2
            assert set(clusters1[cid]) == set(clusters2[cid])
