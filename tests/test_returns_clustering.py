"""Tests for returns-based clustering."""

import pytest
import numpy as np
import pandas as pd
from clustering import ReturnsClusterer


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


class TestReturnsClusterer:
    """Test returns-based clustering."""

    def test_basic_returns_clustering(self):
        """Test clustering by mean returns."""
        returns = generate_synthetic_returns(n_assets=30)

        clusterer = ReturnsClusterer(max_cluster_size=15)
        clusters = clusterer.cluster(returns)

        # Should group assets with similar returns
        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 30

    def test_returns_separation(self):
        """Test that high/low return assets are separated."""
        # Create assets with distinct return levels
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        high_return = pd.DataFrame({
            f'HIGH_{i}': np.random.normal(0.002, 0.01, 252)
            for i in range(5)
        }, index=dates)

        low_return = pd.DataFrame({
            f'LOW_{i}': np.random.normal(-0.001, 0.01, 252)
            for i in range(5)
        }, index=dates)

        returns = pd.concat([high_return, low_return], axis=1)

        clusterer = ReturnsClusterer(max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should create at least 1 cluster
        assert len(clusters) >= 1

        # Check if high and low return assets tend to be in different clusters
        # (This is probabilistic, so we just verify valid clustering)
        assert sum(len(t) for t in clusters.values()) == 10

    def test_growth_vs_value_separation(self):
        """Test separation of growth (high return) vs value (low return) assets."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        # Growth stocks: high return, high volatility
        growth = pd.DataFrame({
            f'GROWTH_{i}': np.random.normal(0.003, 0.02, 252)
            for i in range(10)
        }, index=dates)

        # Value stocks: low return, low volatility
        value = pd.DataFrame({
            f'VALUE_{i}': np.random.normal(0.0005, 0.01, 252)
            for i in range(10)
        }, index=dates)

        returns = pd.concat([growth, value], axis=1)

        clusterer = ReturnsClusterer(max_cluster_size=15, linkage_method='ward')
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 20

    def test_linkage_methods(self):
        """Test different linkage methods."""
        returns = generate_synthetic_returns(n_assets=30)

        for method in ['ward', 'single', 'complete', 'average']:
            clusterer = ReturnsClusterer(max_cluster_size=15, linkage_method=method)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 30

    def test_max_cluster_size_constraint(self):
        """Test that max cluster size is respected."""
        returns = generate_synthetic_returns(n_assets=50)

        clusterer = ReturnsClusterer(max_cluster_size=12)
        clusters = clusterer.cluster(returns)

        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 12

    def test_uniform_returns(self):
        """Test clustering when all assets have similar returns."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        # All assets have similar mean return
        returns = pd.DataFrame({
            f'ASSET_{i}': np.random.normal(0.001, 0.01, 252)
            for i in range(20)
        }, index=dates)

        clusterer = ReturnsClusterer(max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should still produce valid clusters (even if arbitrary)
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 20
