"""Comparison tests across all clustering methods."""

import pytest
import numpy as np
import pandas as pd
from clustering import (
    PortfolioClustering,
    CorrelationClusterer,
    ReturnsClusterer,
    VolatilityClusterer,
    SectorClusterer,
    FactorClusterer,
)


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


class TestClusteringComparison:
    """Compare all clustering methods on same dataset."""

    def test_all_methods_produce_valid_clusters(self):
        """Test that all methods produce valid clusters."""
        returns = generate_synthetic_returns(n_assets=30)

        methods = {
            'Hierarchical': PortfolioClustering(),
            'Correlation': CorrelationClusterer(),
            'Returns': ReturnsClusterer(),
            'Volatility': VolatilityClusterer(),
            'Factor': FactorClusterer(),
        }

        for name, clusterer in methods.items():
            if name == 'Hierarchical':
                clusters = clusterer.cluster(returns.corr())
            else:
                clusters = clusterer.cluster(returns)

            # All should produce valid clusters
            assert len(clusters) > 0, f"{name} produced no clusters"
            total_assets = sum(len(t) for t in clusters.values())
            assert total_assets == 30, f"{name} lost assets: {total_assets}/30"

            # Check size constraints
            for cluster_id, tickers in clusters.items():
                assert len(tickers) <= 18, f"{name} {cluster_id} too large"

    def test_all_methods_handle_large_portfolio(self):
        """Test all methods on larger portfolio."""
        returns = generate_synthetic_returns(n_assets=100)

        methods = {
            'Hierarchical': PortfolioClustering(max_cluster_size=18),
            'Correlation': CorrelationClusterer(max_cluster_size=18),
            'Returns': ReturnsClusterer(max_cluster_size=18),
            'Volatility': VolatilityClusterer(max_cluster_size=18),
            'Factor': FactorClusterer(max_cluster_size=18),
        }

        for name, clusterer in methods.items():
            if name == 'Hierarchical':
                clusters = clusterer.cluster(returns.corr())
            else:
                clusters = clusterer.cluster(returns)

            # Verify valid clustering
            assert len(clusters) >= 6, f"{name} should create multiple clusters"
            total_assets = sum(len(t) for t in clusters.values())
            assert total_assets == 100, f"{name} lost assets"

            # Check size constraints
            for cluster_id, tickers in clusters.items():
                assert len(tickers) <= 18, f"{name} {cluster_id} exceeds max size"

    def test_cluster_quality_metrics(self):
        """Compare cluster quality across methods."""
        returns = generate_synthetic_returns(n_assets=50, seed=123)

        methods = {
            'Hierarchical': PortfolioClustering(max_cluster_size=15),
            'Correlation': CorrelationClusterer(max_cluster_size=15),
            'Returns': ReturnsClusterer(max_cluster_size=15),
            'Volatility': VolatilityClusterer(max_cluster_size=15),
            'Factor': FactorClusterer(n_factors=5, max_cluster_size=15),
        }

        quality_metrics = {}

        for name, clusterer in methods.items():
            if name == 'Hierarchical':
                clusters = clusterer.cluster(returns.corr())
            else:
                clusters = clusterer.cluster(returns)

            # Compute intra-cluster correlation
            intra_corrs = []
            for cluster_id, tickers in clusters.items():
                if len(tickers) > 1:
                    cluster_returns = returns[tickers]
                    corr_matrix = cluster_returns.corr()
                    # Upper triangle excluding diagonal
                    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                    intra_corrs.extend(corr_matrix.values[mask])

            avg_intra_corr = np.mean(intra_corrs) if intra_corrs else 0
            quality_metrics[name] = {
                'n_clusters': len(clusters),
                'avg_intra_correlation': avg_intra_corr,
                'cluster_sizes': [len(t) for t in clusters.values()],
            }

        # All methods should have produced clusters
        for name, metrics in quality_metrics.items():
            assert metrics['n_clusters'] > 0, f"{name} produced no clusters"
            assert len(metrics['cluster_sizes']) == metrics['n_clusters']

    def test_deterministic_behavior(self):
        """Test that methods produce consistent results."""
        returns = generate_synthetic_returns(n_assets=30, seed=456)

        methods = {
            'Hierarchical': PortfolioClustering(),
            'Correlation': CorrelationClusterer(),
            'Returns': ReturnsClusterer(),
            'Volatility': VolatilityClusterer(),
            'Factor': FactorClusterer(n_factors=3),
        }

        for name, clusterer in methods.items():
            # Run twice
            if name == 'Hierarchical':
                clusters1 = clusterer.cluster(returns.corr())
                clusters2 = clusterer.cluster(returns.corr())
            else:
                clusters1 = clusterer.cluster(returns)
                clusters2 = clusterer.cluster(returns)

            # Should produce same number of clusters
            assert len(clusters1) == len(clusters2), f"{name} non-deterministic"

            # Should produce same total assets
            assert (sum(len(t) for t in clusters1.values()) ==
                   sum(len(t) for t in clusters2.values()))

    def test_methods_with_different_constraints(self):
        """Test all methods with varying constraints."""
        returns = generate_synthetic_returns(n_assets=40)

        for max_size in [8, 12, 18]:
            methods = {
                'Hierarchical': PortfolioClustering(max_cluster_size=max_size),
                'Correlation': CorrelationClusterer(max_cluster_size=max_size),
                'Returns': ReturnsClusterer(max_cluster_size=max_size),
                'Volatility': VolatilityClusterer(max_cluster_size=max_size),
                'Factor': FactorClusterer(max_cluster_size=max_size),
            }

            for name, clusterer in methods.items():
                if name == 'Hierarchical':
                    clusters = clusterer.cluster(returns.corr())
                else:
                    clusters = clusterer.cluster(returns)

                # Check constraint satisfaction
                for cluster_id, tickers in clusters.items():
                    assert len(tickers) <= max_size, \
                        f"{name} violated max_size={max_size}: {len(tickers)}"

    def test_small_portfolio_handling(self):
        """Test all methods handle small portfolios correctly."""
        returns = generate_synthetic_returns(n_assets=5)

        methods = {
            'Hierarchical': PortfolioClustering(max_cluster_size=18),
            'Correlation': CorrelationClusterer(max_cluster_size=18),
            'Returns': ReturnsClusterer(max_cluster_size=18),
            'Volatility': VolatilityClusterer(max_cluster_size=18),
            'Factor': FactorClusterer(max_cluster_size=18),
        }

        for name, clusterer in methods.items():
            if name == 'Hierarchical':
                clusters = clusterer.cluster(returns.corr())
            else:
                clusters = clusterer.cluster(returns)

            # Should handle small portfolio
            assert len(clusters) >= 1
            assert sum(len(t) for t in clusters.values()) == 5

    def test_clustering_statistics(self):
        """Test cluster statistics across methods."""
        returns = generate_synthetic_returns(n_assets=60)

        methods = {
            'Hierarchical': PortfolioClustering(max_cluster_size=15),
            'Correlation': CorrelationClusterer(max_cluster_size=15),
            'Returns': ReturnsClusterer(max_cluster_size=15),
            'Volatility': VolatilityClusterer(max_cluster_size=15),
            'Factor': FactorClusterer(max_cluster_size=15),
        }

        for name, clusterer in methods.items():
            if name == 'Hierarchical':
                clusters = clusterer.cluster(returns.corr())
            else:
                clusters = clusterer.cluster(returns)

            cluster_sizes = [len(t) for t in clusters.values()]

            # Basic statistics
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            avg_size = np.mean(cluster_sizes)

            assert min_size >= 1, f"{name} has empty cluster"
            assert max_size <= 15, f"{name} exceeds max size"
            assert avg_size > 0, f"{name} invalid average"

            # Variance should be reasonable
            std_size = np.std(cluster_sizes)
            assert std_size >= 0, f"{name} invalid std"


@pytest.mark.slow
class TestAdvancedClusterers:
    """Tests for advanced clusterers with optional dependencies."""

    def test_dtw_if_available(self):
        """Test DTW clustering if tslearn is available."""
        try:
            from clustering import DTWClusterer

            returns = generate_synthetic_returns(n_assets=15, n_days=100)
            clusterer = DTWClusterer(max_cluster_size=10)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 15

        except ImportError:
            pytest.skip("tslearn not available")

    def test_graph_if_available(self):
        """Test graph clustering if networkx is available."""
        try:
            from clustering import GraphClusterer

            returns = generate_synthetic_returns(n_assets=20, n_days=100)
            clusterer = GraphClusterer(correlation_threshold=0.3, max_cluster_size=10)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 20

        except ImportError:
            pytest.skip("networkx not available")
