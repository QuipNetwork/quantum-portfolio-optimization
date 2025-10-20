"""Factor-based clustering using PCA."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from .base import BaseClusterer


class FactorClusterer(BaseClusterer):
    """
    Cluster assets by common factor exposures.

    Uses PCA to extract latent factors, then clusters based on factor loadings.
    Captures fundamental drivers of returns.
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 n_factors: int = 5):
        """
        Initialize factor-based clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            n_factors: Number of principal components to extract (default 5)
        """
        super().__init__(max_cluster_size, n_bits, linkage_method)
        self.n_factors = n_factors

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute distance matrix based on factor loadings.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Condensed distance array
        """
        from sklearn.decomposition import PCA

        # Cap n_factors at number of assets
        n_assets = returns.shape[1]
        n_factors = min(self.n_factors, n_assets)

        # Extract factors using PCA
        pca = PCA(n_components=n_factors)

        # Fit on transposed returns (N assets × T time periods)
        factor_loadings = pca.fit_transform(returns.T)

        # Distance between factor loadings
        distance = pdist(factor_loadings, metric='euclidean')

        return distance

    def get_factor_exposures(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Get factor exposures for each asset.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            DataFrame of factor loadings (N assets × n_factors)
        """
        from sklearn.decomposition import PCA

        # Cap n_factors at number of assets
        n_assets = returns.shape[1]
        n_factors = min(self.n_factors, n_assets)

        pca = PCA(n_components=n_factors)
        factor_loadings = pca.fit_transform(returns.T)

        return pd.DataFrame(
            factor_loadings,
            index=returns.columns,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )

    def get_explained_variance(self, returns: pd.DataFrame) -> pd.Series:
        """
        Get explained variance ratio for each factor.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Series of explained variance ratios
        """
        from sklearn.decomposition import PCA

        # Cap n_factors at number of assets
        n_assets = returns.shape[1]
        n_factors = min(self.n_factors, n_assets)

        pca = PCA(n_components=n_factors)
        pca.fit(returns.T)

        return pd.Series(
            pca.explained_variance_ratio_,
            index=[f'Factor_{i+1}' for i in range(n_factors)]
        )
