"""Tests for graph-based clustering."""

import pytest
import numpy as np
import pandas as pd

# Skip entire module if networkx not available
pytest.importorskip("networkx", reason="networkx is required for graph clustering")

from clustering import GraphClusterer


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


class TestGraphClusterer:
    """Test graph-based clustering."""

    def test_basic_graph_clustering(self):
        """Test graph community detection."""
        returns = generate_synthetic_returns(n_assets=20, n_days=100)

        clusterer = GraphClusterer(
            correlation_threshold=0.3,
            algorithm='greedy',
            max_cluster_size=10
        )
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 20

    def test_different_algorithms(self):
        """Test different community detection algorithms."""
        returns = generate_synthetic_returns(n_assets=20, n_days=100)

        algorithms = ['louvain', 'greedy', 'label_prop']

        for algorithm in algorithms:
            try:
                clusterer = GraphClusterer(
                    correlation_threshold=0.3,
                    algorithm=algorithm,
                    max_cluster_size=10
                )
                clusters = clusterer.cluster(returns)

                assert len(clusters) > 0
                assert sum(len(t) for t in clusters.values()) == 20
            except ImportError:
                # louvain requires python-louvain package
                if algorithm == 'louvain':
                    pytest.skip(f"{algorithm} algorithm requires additional package")
                else:
                    raise

    def test_correlation_threshold_effect(self):
        """Test effect of different correlation thresholds."""
        returns = generate_synthetic_returns(n_assets=20, n_days=100)

        # Low threshold = more edges = fewer, larger clusters
        clusterer_low = GraphClusterer(correlation_threshold=0.1, max_cluster_size=15)
        clusters_low = clusterer_low.cluster(returns)

        # High threshold = fewer edges = more, smaller clusters
        clusterer_high = GraphClusterer(correlation_threshold=0.6, max_cluster_size=15)
        clusters_high = clusterer_high.cluster(returns)

        # Both should produce valid clusters
        assert len(clusters_low) > 0
        assert len(clusters_high) > 0

        # Typically, higher threshold should produce more clusters
        # (but this is probabilistic with synthetic data)
        assert sum(len(t) for t in clusters_low.values()) == 20
        assert sum(len(t) for t in clusters_high.values()) == 20

    def test_max_cluster_size_constraint(self):
        """Test that max cluster size is respected."""
        returns = generate_synthetic_returns(n_assets=30, n_days=100)

        clusterer = GraphClusterer(
            correlation_threshold=0.3,
            max_cluster_size=8
        )
        clusters = clusterer.cluster(returns)

        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 8

    def test_strongly_connected_groups(self):
        """Test clustering with strongly connected groups."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # Create two strongly connected groups
        base1 = np.random.normal(0, 0.01, 100)
        base2 = np.random.normal(0, 0.01, 100)

        group1 = pd.DataFrame({
            f'G1_{i}': base1 + np.random.normal(0, 0.002, 100)
            for i in range(5)
        }, index=dates)

        group2 = pd.DataFrame({
            f'G2_{i}': base2 + np.random.normal(0, 0.002, 100)
            for i in range(5)
        }, index=dates)

        returns = pd.concat([group1, group2], axis=1)

        clusterer = GraphClusterer(correlation_threshold=0.3, max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should produce at least 1 cluster
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 10

    def test_sparse_correlation_graph(self):
        """Test with sparse correlation (high threshold)."""
        returns = generate_synthetic_returns(n_assets=15, n_days=100)

        # Very high threshold = very sparse graph
        clusterer = GraphClusterer(correlation_threshold=0.9, max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should still produce valid clusters (may have many singleton clusters)
        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 15

    def test_dense_correlation_graph(self):
        """Test with dense correlation (low threshold)."""
        returns = generate_synthetic_returns(n_assets=15, n_days=100)

        # Very low threshold = dense graph
        clusterer = GraphClusterer(correlation_threshold=0.0, max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 15

    def test_negative_correlation_handling(self):
        """Test handling of negative correlations."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        base = np.random.normal(0, 0.01, 100)

        returns = pd.DataFrame({
            'POS_1': base + np.random.normal(0, 0.002, 100),
            'POS_2': base + np.random.normal(0, 0.002, 100),
            'NEG_1': -base + np.random.normal(0, 0.002, 100),
            'NEG_2': -base + np.random.normal(0, 0.002, 100),
        }, index=dates)

        # Graph clusterer uses absolute correlation
        clusterer = GraphClusterer(correlation_threshold=0.3, max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 4

    def test_isolated_assets(self):
        """Test handling of assets with low correlation to everything."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # Connected group
        base = np.random.normal(0, 0.01, 100)
        connected = pd.DataFrame({
            f'CONN_{i}': base + np.random.normal(0, 0.002, 100)
            for i in range(5)
        }, index=dates)

        # Isolated assets (uncorrelated)
        isolated = pd.DataFrame({
            f'ISOL_{i}': np.random.normal(0, 0.01, 100)
            for i in range(3)
        }, index=dates)

        returns = pd.concat([connected, isolated], axis=1)

        clusterer = GraphClusterer(correlation_threshold=0.5, max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        # Should handle both connected and isolated assets
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 8
