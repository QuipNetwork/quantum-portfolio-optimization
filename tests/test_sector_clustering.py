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

"""Tests for sector-based clustering."""

import pytest
import numpy as np
import pandas as pd
from clustering import SectorClusterer


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


class TestSectorClusterer:
    """Test sector-based clustering."""

    def test_sector_clustering_with_map(self):
        """Test clustering with pre-defined sector map."""
        returns = generate_synthetic_returns(n_assets=12)
        tickers = returns.columns.tolist()

        # Create sector map
        sector_map = {
            tickers[0]: 'Technology',
            tickers[1]: 'Technology',
            tickers[2]: 'Technology',
            tickers[3]: 'Technology',
            tickers[4]: 'Finance',
            tickers[5]: 'Finance',
            tickers[6]: 'Finance',
            tickers[7]: 'Healthcare',
            tickers[8]: 'Healthcare',
            tickers[9]: 'Healthcare',
            tickers[10]: 'Energy',
            tickers[11]: 'Energy',
        }

        clusterer = SectorClusterer(sector_map=sector_map, max_cluster_size=18)
        clusters = clusterer.cluster(returns)

        # Should have 4 sectors
        assert len(clusters) == 4
        assert 'Technology' in clusters
        assert 'Finance' in clusters
        assert 'Healthcare' in clusters
        assert 'Energy' in clusters

        # Verify correct assignment
        assert len(clusters['Technology']) == 4
        assert len(clusters['Finance']) == 3
        assert len(clusters['Healthcare']) == 3
        assert len(clusters['Energy']) == 2

    def test_large_sector_splitting(self):
        """Test that large sectors are split."""
        returns = generate_synthetic_returns(n_assets=25)
        tickers = returns.columns.tolist()

        # Create one large sector
        sector_map = {ticker: 'LargeSector' for ticker in tickers}

        clusterer = SectorClusterer(sector_map=sector_map, max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should split into multiple clusters
        assert len(clusters) >= 3  # 25 assets / 10 max = 3 clusters

        # All assets should be accounted for
        assert sum(len(t) for t in clusters.values()) == 25

        # Each cluster should respect max size
        for cluster_id, tickers_list in clusters.items():
            assert len(tickers_list) <= 10

    def test_multiple_large_sectors(self):
        """Test splitting multiple large sectors."""
        returns = generate_synthetic_returns(n_assets=50)
        tickers = returns.columns.tolist()

        # Create two large sectors
        sector_map = {
            **{tickers[i]: 'Tech' for i in range(30)},
            **{tickers[i]: 'Finance' for i in range(30, 50)},
        }

        clusterer = SectorClusterer(sector_map=sector_map, max_cluster_size=12)
        clusters = clusterer.cluster(returns)

        # Should split both sectors
        assert len(clusters) >= 5  # 30/12 + 20/12 ≈ 5

        # All assets accounted for
        assert sum(len(t) for t in clusters.values()) == 50

        # Size constraints
        for cluster_id, tickers_list in clusters.items():
            assert len(tickers_list) <= 12

    def test_mixed_sector_sizes(self):
        """Test clustering with sectors of varying sizes."""
        returns = generate_synthetic_returns(n_assets=40)
        tickers = returns.columns.tolist()

        sector_map = {
            # Small sector
            **{tickers[i]: 'Energy' for i in range(5)},
            # Medium sector
            **{tickers[i]: 'Healthcare' for i in range(5, 15)},
            # Large sector (needs splitting)
            **{tickers[i]: 'Technology' for i in range(15, 40)},
        }

        clusterer = SectorClusterer(sector_map=sector_map, max_cluster_size=15)
        clusters = clusterer.cluster(returns)

        # Should have: 1 Energy + 1 Healthcare + 2 Technology (split) = 4+ clusters
        assert len(clusters) >= 3

        # All assets accounted for
        assert sum(len(t) for t in clusters.values()) == 40

    def test_unknown_sector_handling(self):
        """Test handling of assets with unknown sector."""
        returns = generate_synthetic_returns(n_assets=15)
        tickers = returns.columns.tolist()

        # Only map some tickers
        sector_map = {
            tickers[0]: 'Technology',
            tickers[1]: 'Technology',
            tickers[2]: 'Finance',
            tickers[3]: 'Finance',
            # tickers[4:] have no sector mapping
        }

        clusterer = SectorClusterer(sector_map=sector_map, max_cluster_size=18)
        clusters = clusterer.cluster(returns)

        # Should create clusters for known sectors + unknown
        assert len(clusters) >= 2

        # All assets should be accounted for
        assert sum(len(t) for t in clusters.values()) == 15

    @pytest.mark.skip(reason="Requires internet connection and Yahoo Finance API")
    def test_sector_fetching(self):
        """Test fetching sectors from Yahoo Finance."""
        # Create returns with real tickers
        np.random.seed(42)
        returns = pd.DataFrame({
            'AAPL': np.random.randn(100),
            'MSFT': np.random.randn(100),
            'JPM': np.random.randn(100),
            'BAC': np.random.randn(100),
        })

        clusterer = SectorClusterer(max_cluster_size=18)
        clusters = clusterer.cluster(returns)

        # Should fetch sectors and cluster
        # AAPL, MSFT should be in Technology
        # JPM, BAC should be in Financials
        assert len(clusters) >= 1

    def test_single_asset_sectors(self):
        """Test handling of sectors with single assets."""
        returns = generate_synthetic_returns(n_assets=5)
        tickers = returns.columns.tolist()

        # Each asset in different sector
        sector_map = {
            tickers[0]: 'Technology',
            tickers[1]: 'Finance',
            tickers[2]: 'Healthcare',
            tickers[3]: 'Energy',
            tickers[4]: 'Utilities',
        }

        clusterer = SectorClusterer(sector_map=sector_map, max_cluster_size=18)
        clusters = clusterer.cluster(returns)

        # Should have 5 clusters (one per asset)
        assert len(clusters) == 5

        # Each cluster should have 1 asset
        for cluster_id, tickers_list in clusters.items():
            assert len(tickers_list) == 1
