"""Tests for factor-based clustering."""

import pytest
import numpy as np
import pandas as pd
from clustering import FactorClusterer


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


class TestFactorClusterer:
    """Test factor-based clustering."""

    def test_basic_factor_clustering(self):
        """Test clustering by factor exposures."""
        returns = generate_synthetic_returns(n_assets=30, n_days=252)

        clusterer = FactorClusterer(n_factors=3, max_cluster_size=15)
        clusters = clusterer.cluster(returns)

        assert len(clusters) > 0
        assert sum(len(t) for t in clusters.values()) == 30

    def test_factor_exposures_shape(self):
        """Test factor exposure calculation."""
        returns = generate_synthetic_returns(n_assets=30, n_days=252)

        clusterer = FactorClusterer(n_factors=5)
        exposures = clusterer.get_factor_exposures(returns)

        # Should have 30 assets × 5 factors
        assert exposures.shape == (30, 5)
        assert list(exposures.columns) == [f'Factor_{i+1}' for i in range(5)]
        assert len(exposures.index) == 30

    def test_explained_variance(self):
        """Test explained variance calculation."""
        returns = generate_synthetic_returns(n_assets=30, n_days=252)

        clusterer = FactorClusterer(n_factors=5)
        variance = clusterer.get_explained_variance(returns)

        # Should have 5 factors
        assert len(variance) == 5

        # Should sum to <= 1.0
        assert variance.sum() <= 1.0

        # First factor should explain most variance
        assert variance.iloc[0] >= variance.iloc[1]

        # All values should be positive
        assert all(variance >= 0)

    def test_different_n_factors(self):
        """Test clustering with different numbers of factors."""
        returns = generate_synthetic_returns(n_assets=30, n_days=252)

        for n_factors in [2, 3, 5, 8]:
            clusterer = FactorClusterer(n_factors=n_factors, max_cluster_size=15)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 30

    def test_max_cluster_size_constraint(self):
        """Test that max cluster size is respected."""
        returns = generate_synthetic_returns(n_assets=50, n_days=252)

        clusterer = FactorClusterer(n_factors=5, max_cluster_size=12)
        clusters = clusterer.cluster(returns)

        for cluster_id, tickers in clusters.items():
            assert len(tickers) <= 12

    def test_linkage_methods(self):
        """Test different linkage methods."""
        returns = generate_synthetic_returns(n_assets=30, n_days=252)

        for method in ['ward', 'single', 'complete', 'average']:
            clusterer = FactorClusterer(n_factors=3, linkage_method=method, max_cluster_size=15)
            clusters = clusterer.cluster(returns)

            assert len(clusters) > 0
            assert sum(len(t) for t in clusters.values()) == 30

    def test_factor_clustering_with_structure(self):
        """Test factor clustering with known factor structure."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)

        # Create returns with underlying factor structure
        # Factor 1: Market factor
        market = np.random.normal(0.001, 0.02, 252)
        # Factor 2: Size factor
        size = np.random.normal(0, 0.01, 252)
        # Factor 3: Value factor
        value = np.random.normal(0, 0.01, 252)

        returns_data = {}

        # Group 1: High market exposure
        for i in range(10):
            returns_data[f'MARKET_{i}'] = (
                0.8 * market +
                0.1 * size +
                0.1 * value +
                np.random.normal(0, 0.005, 252)
            )

        # Group 2: High size exposure
        for i in range(10):
            returns_data[f'SIZE_{i}'] = (
                0.3 * market +
                0.6 * size +
                0.1 * value +
                np.random.normal(0, 0.005, 252)
            )

        # Group 3: High value exposure
        for i in range(10):
            returns_data[f'VALUE_{i}'] = (
                0.3 * market +
                0.1 * size +
                0.6 * value +
                np.random.normal(0, 0.005, 252)
            )

        returns = pd.DataFrame(returns_data, index=dates)

        clusterer = FactorClusterer(n_factors=3, max_cluster_size=15)
        clusters = clusterer.cluster(returns)

        # Should produce valid clusters
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 30

    def test_factor_exposures_consistency(self):
        """Test that factor exposures are consistent across calls."""
        returns = generate_synthetic_returns(n_assets=20, n_days=252, seed=123)

        clusterer = FactorClusterer(n_factors=3)

        # Get exposures twice
        exposures1 = clusterer.get_factor_exposures(returns)
        exposures2 = clusterer.get_factor_exposures(returns)

        # Should be identical (PCA is deterministic)
        pd.testing.assert_frame_equal(exposures1, exposures2)

    def test_minimum_samples_for_factors(self):
        """Test that we need sufficient samples for PCA."""
        # Very few samples
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=10)
        returns = pd.DataFrame({
            f'ASSET_{i}': np.random.randn(10)
            for i in range(20)
        }, index=dates)

        # Should still work but may not be meaningful
        clusterer = FactorClusterer(n_factors=3, max_cluster_size=10)
        clusters = clusterer.cluster(returns)

        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 20

    def test_more_factors_than_assets(self):
        """Test behavior when requesting more factors than assets."""
        returns = generate_synthetic_returns(n_assets=5, n_days=252)

        # Request 10 factors but only 5 assets
        # PCA should cap at min(n_assets, n_samples)
        clusterer = FactorClusterer(n_factors=10, max_cluster_size=18)
        clusters = clusterer.cluster(returns)

        # Should still produce valid clusters
        assert len(clusters) >= 1
        assert sum(len(t) for t in clusters.values()) == 5
