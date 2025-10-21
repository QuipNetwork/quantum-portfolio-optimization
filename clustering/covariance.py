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

"""Covariance-based clustering."""

import numpy as np
import pandas as pd
from typing import Dict
from .base import BaseClusterer


class CovarianceClusterer(BaseClusterer):
    """
    Cluster assets by covariance structure.

    Unlike correlation clustering which normalizes by volatility,
    covariance clustering preserves the scale information and groups
    assets based on their absolute covariation patterns.

    This is useful when you want to cluster assets with similar
    volatility profiles AND correlation patterns together.

    Distance metrics available:
    - 'euclidean': Euclidean distance in covariance space
    - 'frobenius': Frobenius norm distance (similar to euclidean but normalized)
    - 'spectral': Distance based on eigenvalue structure
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 distance_metric: str = 'euclidean',
                 normalize: bool = False,
                 target_cluster_size: int = None):
        """
        Initialize covariance clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            distance_metric: 'euclidean', 'frobenius', or 'spectral'
            normalize: If True, normalize distances to [0, 1] range
        """
        super().__init__(max_cluster_size, n_bits, linkage_method, target_cluster_size=target_cluster_size)
        self.distance_metric = distance_metric
        self.normalize = normalize

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute covariance-based distance matrix.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Distance matrix (N × N)
        """
        # Compute covariance matrix
        cov = returns.cov().values
        n_assets = cov.shape[0]

        if self.distance_metric == 'euclidean':
            distance = self._euclidean_distance(cov, n_assets)
        elif self.distance_metric == 'frobenius':
            distance = self._frobenius_distance(cov, n_assets)
        elif self.distance_metric == 'spectral':
            distance = self._spectral_distance(cov, n_assets)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Normalize if requested
        if self.normalize:
            max_dist = distance.max()
            if max_dist > 0:
                distance = distance / max_dist

        # Ensure distance is symmetric and zero-diagonal
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2

        return distance

    def _euclidean_distance(self, cov: np.ndarray, n_assets: int) -> np.ndarray:
        """
        Compute Euclidean distance between covariance vectors.

        For each pair of assets i and j, compute the distance as:
        d(i,j) = sqrt(sum_k (cov[i,k] - cov[j,k])^2)

        This measures how differently asset i and j covary with all other assets.
        """
        distance = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Distance = norm of difference between covariance vectors
                diff = cov[i, :] - cov[j, :]
                dist = np.sqrt(np.sum(diff**2))
                distance[i, j] = dist
                distance[j, i] = dist

        return distance

    def _frobenius_distance(self, cov: np.ndarray, n_assets: int) -> np.ndarray:
        """
        Compute Frobenius norm distance between covariance patterns.

        Similar to Euclidean but considers the full covariance structure.
        For assets i and j, uses the L2 norm of their covariance vector difference.
        """
        distance = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Frobenius norm simplifies to L2 norm for vectors
                diff = cov[i, :] - cov[j, :]
                dist = np.linalg.norm(diff)  # Default is L2 norm
                distance[i, j] = dist
                distance[j, i] = dist

        return distance

    def _spectral_distance(self, cov: np.ndarray, n_assets: int) -> np.ndarray:
        """
        Compute distance based on eigenvalue/eigenvector structure.

        This captures the directional similarity of how assets contribute
        to portfolio variance through their principal components.
        """
        distance = np.zeros((n_assets, n_assets))

        # Compute eigendecomposition of covariance matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # For each pair of assets, compare their loadings on principal components
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    # Weight by eigenvalues (importance of each component)
                    weights = np.abs(eigenvalues)
                    weights = weights / weights.sum() if weights.sum() > 0 else weights

                    # Distance = weighted difference in eigenvector loadings
                    diff = eigenvectors[i, :] - eigenvectors[j, :]
                    dist = np.sqrt(np.sum(weights * diff**2))

                    distance[i, j] = dist
                    distance[j, i] = dist

        except np.linalg.LinAlgError:
            # Fall back to euclidean if eigendecomposition fails
            return self._euclidean_distance(cov, n_assets)

        return distance

    def get_covariance_summary(self, returns: pd.DataFrame,
                              clusters: Dict) -> pd.DataFrame:
        """
        Get summary statistics of covariance structure for each cluster.

        Args:
            returns: Returns DataFrame
            clusters: Cluster dictionary

        Returns:
            DataFrame with covariance statistics per cluster
        """
        cov = returns.cov()
        summary = []

        for cluster_id, tickers in clusters.items():
            if len(tickers) > 1:
                cluster_cov = cov.loc[tickers, tickers]

                # Extract off-diagonal elements (actual covariances)
                mask = ~np.eye(len(tickers), dtype=bool)
                cov_values = cluster_cov.values[mask]

                summary.append({
                    'cluster': cluster_id,
                    'n_assets': len(tickers),
                    'avg_covariance': np.mean(cov_values),
                    'std_covariance': np.std(cov_values),
                    'min_covariance': np.min(cov_values),
                    'max_covariance': np.max(cov_values),
                    'avg_variance': np.mean(np.diag(cluster_cov)),
                })
            else:
                # Single asset cluster
                variance = cov.loc[tickers[0], tickers[0]]
                summary.append({
                    'cluster': cluster_id,
                    'n_assets': 1,
                    'avg_covariance': 0.0,
                    'std_covariance': 0.0,
                    'min_covariance': 0.0,
                    'max_covariance': 0.0,
                    'avg_variance': variance,
                })

        return pd.DataFrame(summary)
