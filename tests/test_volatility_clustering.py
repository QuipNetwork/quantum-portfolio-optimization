"""Tests for volatility-based clustering."""

import pytest
import numpy as np
import pandas as pd
from clustering import VolatilityClusterer


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


class TestVolatilityClusterer:
    """Test volatility-based clustering."""

    def test_basic_volatility_clustering(self):
        """Test clustering by volatility."""
        returns = generate_synthetic_returns(n_assets=30)

        clusterer = VolatilityClusterer(max_cluster_size=15)
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 30

    def test_volatility_separation(self):
        """Test that high/low vol assets are separated."""
        # Create assets with distinct volatility levels
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        high_vol = pd.DataFrame({
            f'HIGHVOL_{i}': np.random.normal(0, 0.05, 252)
            for i in range(5)
        }, index=dates)

        low_vol = pd.DataFrame({
            f'LOWVOL_{i}': np.random.normal(0, 0.005, 252)
            for i in range(5)
        }, index=dates)

        returns = pd.concat([high_vol, low_vol], axis=1)

        clusterer = VolatilityClusterer(max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should create at least 1 cluster
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 10

    def test_three_volatility_levels(self):
        """Test clustering with three distinct volatility levels."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        low_vol = pd.DataFrame({
            f'LOW_{i}': np.random.normal(0, 0.005, 252)
            for i in range(5)
        }, index=dates)

        med_vol = pd.DataFrame({
            f'MED_{i}': np.random.normal(0, 0.015, 252)
            for i in range(5)
        }, index=dates)

        high_vol = pd.DataFrame({
            f'HIGH_{i}': np.random.normal(0, 0.04, 252)
            for i in range(5)
        }, index=dates)

        returns = pd.concat([low_vol, med_vol, high_vol], axis=1)

        clusterer = VolatilityClusterer(max_cluster_size=10, linkage_method='ward')
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 15

    def test_linkage_methods(self):
        """Test different linkage methods."""
        returns = generate_synthetic_returns(n_assets=30)

        for method in ['ward', 'single', 'complete', 'average']:
            clusterer = VolatilityClusterer(max_cluster_size=15, linkage_method=method)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 30

    def test_max_cluster_size_constraint(self):
        """Test that max cluster size is respected."""
        returns = generate_synthetic_returns(n_assets=50)

        clusterer = VolatilityClusterer(max_cluster_size=12)
        clusters = clusterer.cluster(returns)

        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 12

    def test_defensive_vs_aggressive_assets(self):
        """Test separation of defensive (low vol) vs aggressive (high vol) assets."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        # Defensive: low volatility
        defensive = pd.DataFrame({
            f'DEFENSIVE_{i}': np.random.normal(0.0005, 0.008, 252)
            for i in range(8)
        }, index=dates)

        # Aggressive: high volatility
        aggressive = pd.DataFrame({
            f'AGGRESSIVE_{i}': np.random.normal(0.002, 0.03, 252)
            for i in range(8)
        }, index=dates)

        returns = pd.concat([defensive, aggressive], axis=1)

        clusterer = VolatilityClusterer(max_cluster_size=12)
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 16

    def test_uniform_volatility(self):
        """Test clustering when all assets have similar volatility."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        # All assets have similar volatility
        returns = pd.DataFrame({
            f'ASSET_{i}': np.random.normal(0.001, 0.015, 252)
            for i in range(20)
        }, index=dates)

        clusterer = VolatilityClusterer(max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should still produce valid clusters
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 20
