"""Tests for hierarchical correlation clustering."""

import pytest
import numpy as np
import pandas as pd
from clustering import PortfolioClustering


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


class TestPortfolioClustering:
    """Test default hierarchical correlation clustering."""

    def test_basic_clustering(self):
        """Test basic clustering functionality."""
        returns = generate_synthetic_returns(n_assets=50)
        corr = returns.corr()

        clusterer = PortfolioClustering(max_cluster_size=18, n_bits=10)
        clusters = clusterer.cluster(corr)

        # Validate
        assert len(clusters) > 0
        assert all(isinstance(tickers, list) for tickers in clusters.values())
        assert sum(len(t) for t in clusters.values()) == 50  # All assets clustered

    def test_max_cluster_size_constraint(self):
        """Test that max_cluster_size is respected."""
        returns = generate_synthetic_returns(n_assets=50)
        corr = returns.corr()

        clusterer = PortfolioClustering(max_cluster_size=18, n_bits=10)
        clusters = clusterer.cluster(corr)

        # Check all clusters satisfy size constraint
        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 18, f"{cluster_id} has {len(tickers)} > 18 assets"

    def test_variable_constraint(self):
        """Test that variable count constraint is satisfied."""
        returns = generate_synthetic_returns(n_assets=50)
        corr = returns.corr()

        clusterer = PortfolioClustering(max_cluster_size=18, n_bits=10)
        clusters = clusterer.cluster(corr)

        # Validate degree constraint
        assert clusterer.validate_degree_constraint(clusters)

    def test_cluster_stats(self):
        """Test cluster statistics."""
        returns = generate_synthetic_returns(n_assets=50)
        corr = returns.corr()

        clusterer = PortfolioClustering(max_cluster_size=18, n_bits=10)
        clusters = clusterer.cluster(corr)

        stats = clusterer.get_cluster_stats(clusters)

        assert 'n_clusters' in stats
        assert 'max_cluster_size' in stats
        assert 'avg_cluster_size' in stats
        assert stats['total_assets'] == 50

    def test_small_portfolio(self):
        """Test clustering with small portfolio."""
        returns = generate_synthetic_returns(n_assets=10)
        corr = returns.corr()

        clusterer = PortfolioClustering(max_cluster_size=18)
        clusters = clusterer.cluster(corr)

        # Should have at least 1 cluster
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 10

    def test_large_portfolio(self):
        """Test clustering with large portfolio."""
        returns = generate_synthetic_returns(n_assets=100)
        corr = returns.corr()

        clusterer = PortfolioClustering(max_cluster_size=18)
        clusters = clusterer.cluster(corr)

        # Should create multiple clusters
        assert len(clusters) >= 6  # 100 / 18 ≈ 6
        assert sum(len(t) for t in clusters.values()) == 100

        # All should respect size constraint
        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 18
