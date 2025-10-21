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
                 n_factors: int = 5,
                 target_cluster_size: int = None):
        """
        Initialize factor-based clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            n_factors: Number of principal components to extract (default 5)
        """
        super().__init__(max_cluster_size, n_bits, linkage_method, target_cluster_size=target_cluster_size)
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
        from sklearn.preprocessing import StandardScaler

        # Cap n_factors at number of assets
        n_assets = returns.shape[1]
        n_factors = min(self.n_factors, n_assets)

        # Standardize returns to prevent numerical instability in PCA
        # StandardScaler normalizes each feature (time series) to mean=0, std=1
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)

        # Extract factors using PCA
        pca = PCA(n_components=n_factors, svd_solver='full')

        # Fit on transposed returns (N assets × T time periods)
        factor_loadings = pca.fit_transform(returns_scaled.T)

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
        from sklearn.preprocessing import StandardScaler

        # Cap n_factors at number of assets
        n_assets = returns.shape[1]
        n_factors = min(self.n_factors, n_assets)

        # Standardize returns to prevent numerical instability
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)

        pca = PCA(n_components=n_factors, svd_solver='full')
        factor_loadings = pca.fit_transform(returns_scaled.T)

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
        from sklearn.preprocessing import StandardScaler

        # Cap n_factors at number of assets
        n_assets = returns.shape[1]
        n_factors = min(self.n_factors, n_assets)

        # Standardize returns to prevent numerical instability
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)

        pca = PCA(n_components=n_factors, svd_solver='full')
        pca.fit(returns_scaled.T)

        return pd.Series(
            pca.explained_variance_ratio_,
            index=[f'Factor_{i+1}' for i in range(n_factors)]
        )
