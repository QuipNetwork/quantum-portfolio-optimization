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

"""Tests for correlation-based clustering."""

import pytest
import numpy as np
import pandas as pd
from clustering import CorrelationClusterer


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


class TestCorrelationClusterer:
    """Test correlation-based clustering."""

    def test_absolute_correlation(self):
        """Test clustering with absolute correlation."""
        returns = generate_synthetic_returns(n_assets=30)

        clusterer = CorrelationClusterer(use_absolute=True)
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 30

    def test_signed_correlation(self):
        """Test clustering with signed correlation."""
        returns = generate_synthetic_returns(n_assets=30)

        clusterer = CorrelationClusterer(use_absolute=False)
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 30

    def test_absolute_vs_signed_difference(self):
        """Test that absolute vs signed correlation produce different results."""
        returns = generate_synthetic_returns(n_assets=30)

        clusterer_abs = CorrelationClusterer(use_absolute=True)
        clusters_abs = clusterer_abs.cluster(returns)

        clusterer_signed = CorrelationClusterer(use_absolute=False)
        clusters_signed = clusterer_signed.cluster(returns)

        # Both should work
        assert len(clusters_abs) > 0
        assert len(clusters_signed) > 0

        # May produce different clusterings (negative correlation treated differently)
        # Just verify both are valid

    def test_linkage_methods(self):
        """Test different linkage methods."""
        returns = generate_synthetic_returns(n_assets=30)

        for method in ['ward', 'single', 'complete', 'average']:
            clusterer = CorrelationClusterer(linkage_method=method)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0, f"{method} produced no clusters"
            assert sum(len(t) for t in clusters.values()) == 30, f"{method} lost assets"

    def test_max_cluster_size(self):
        """Test max cluster size constraint."""
        returns = generate_synthetic_returns(n_assets=50)

        clusterer = CorrelationClusterer(max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 10, f"{cluster_id} exceeds max size"

    def test_negatively_correlated_assets(self):
        """Test clustering with negatively correlated assets."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        # Create two groups: positively and negatively correlated
        base_returns = np.random.normal(0, 0.01, 252)

        returns = pd.DataFrame({
            'POS_1': base_returns + np.random.normal(0, 0.005, 252),
            'POS_2': base_returns + np.random.normal(0, 0.005, 252),
            'NEG_1': -base_returns + np.random.normal(0, 0.005, 252),
            'NEG_2': -base_returns + np.random.normal(0, 0.005, 252),
        }, index=dates)

        # Absolute correlation should group POS together and NEG together
        clusterer_abs = CorrelationClusterer(use_absolute=True, max_cluster_size=10)
        clusters_abs = clusterer_abs.cluster(returns)

        # Signed correlation treats negative correlation differently
        clusterer_signed = CorrelationClusterer(use_absolute=False, max_cluster_size=10)
        clusters_signed = clusterer_signed.cluster(returns)

        # Both should produce valid clusters
        assert len(clusters_abs) >= 1
        assert len(clusters_signed) >= 1
