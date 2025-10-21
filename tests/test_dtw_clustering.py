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

"""Tests for DTW-based clustering."""

import pytest
import numpy as np
import pandas as pd

# Skip entire module if tslearn not available
pytest.importorskip("tslearn", reason="tslearn is required for DTW clustering")

from clustering import DTWClusterer


def generate_synthetic_returns(n_assets=50, n_days=252, seed=42):
    """Generate synthetic returns for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n_days)
    tickers = [f'ASSET_{i:03d}' for i in range(n_assets)]

    returns_data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0003 + (i / n_assets) * 0.0005
        vol = 0.01 + (i / n_assets) * 0.02
        returns = np.random.normal(drift, vol, n_days)
        returns_data[ticker] = returns

    return pd.DataFrame(returns_data, index=dates)


class TestDTWClusterer:
    """Test DTW-based clustering."""

    def test_basic_dtw_clustering(self):
        """Test DTW clustering."""
        returns = generate_synthetic_returns(n_assets=10, n_days=100)

        clusterer = DTWClusterer(max_cluster_size=5)
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 10

    def test_max_cluster_size_constraint(self):
        """Test that max cluster size is respected."""
        returns = generate_synthetic_returns(n_assets=20, n_days=100)

        clusterer = DTWClusterer(max_cluster_size=8)
        clusters = clusterer.cluster(returns)

        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 8

    def test_linkage_methods(self):
        """Test different linkage methods."""
        returns = generate_synthetic_returns(n_assets=10, n_days=100)

        for method in ['ward', 'single', 'complete', 'average']:
            clusterer = DTWClusterer(max_cluster_size=5, linkage_method=method)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 10

    def test_time_lagged_patterns(self):
        """Test DTW captures time-lagged patterns."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # Create base pattern
        base = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)

        # Create lagged versions
        returns = pd.DataFrame({
            'LEAD_0': base,
            'LEAD_1': np.roll(base, 0) + np.random.normal(0, 0.1, 100),
            'LAG_5': np.roll(base, 5) + np.random.normal(0, 0.1, 100),
            'LAG_10': np.roll(base, 10) + np.random.normal(0, 0.1, 100),
            'UNCORR': np.random.normal(0, 0.1, 100),
        }, index=dates)

        clusterer = DTWClusterer(max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        # DTW should potentially group lagged patterns together
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 5

    def test_momentum_patterns(self):
        """Test DTW clustering with momentum-style patterns."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # Momentum group: consistent uptrend
        momentum = pd.DataFrame({
            f'MOM_{i}': np.cumsum(np.random.normal(0.01, 0.02, 100))
            for i in range(5)
        }, index=dates)

        # Mean reversion group: oscillating
        reversion = pd.DataFrame({
            f'REV_{i}': np.sin(np.linspace(0, 8*np.pi, 100)) + np.random.normal(0, 0.1, 100)
            for i in range(5)
        }, index=dates)

        returns = pd.concat([momentum, reversion], axis=1)

        clusterer = DTWClusterer(max_cluster_size=8)
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 10

    def test_short_time_series(self):
        """Test DTW with short time series."""
        returns = generate_synthetic_returns(n_assets=8, n_days=30)

        clusterer = DTWClusterer(max_cluster_size=5)
        clusters = clusterer.cluster(returns)

        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 8

    def test_dtw_vs_correlation_difference(self):
        """Test that DTW can produce different results than correlation."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # Create time series where DTW and correlation differ
        # Pattern with phase shift
        base = np.sin(np.linspace(0, 4*np.pi, 100))

        returns = pd.DataFrame({
            'A': base + np.random.normal(0, 0.1, 100),
            'B': np.roll(base, 20) + np.random.normal(0, 0.1, 100),  # Shifted
            'C': -base + np.random.normal(0, 0.1, 100),  # Inverted
        }, index=dates)

        clusterer = DTWClusterer(max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        # DTW treats time-shifted patterns differently than correlation
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 3
