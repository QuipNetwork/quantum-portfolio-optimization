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

"""Base class for all clustering methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class BaseClusterer(ABC):
    """Abstract base class for portfolio clustering."""

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 target_cluster_size: int = None):
        """
        Initialize base clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster (for Zephyr degree constraint)
            n_bits: Bits per weight variable (for discretization)
            linkage_method: Hierarchical linkage method ('ward', 'single', 'complete', 'average')
            target_cluster_size: Target average cluster size (default: max_cluster_size // 2)
                                If specified, algorithm aims for this average size to avoid too many small clusters
        """
        self.max_cluster_size = max_cluster_size
        self.target_cluster_size = target_cluster_size if target_cluster_size is not None else max_cluster_size // 2
        self.n_bits = n_bits
        self.max_variables = max_cluster_size * n_bits
        self.linkage_method = linkage_method

    @abstractmethod
    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute distance matrix for clustering.

        Must be implemented by subclasses.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Distance matrix (N × N) or condensed distance array
        """
        pass

    def cluster(self, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Cluster assets using hierarchical clustering.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dictionary mapping cluster IDs to lists of ticker symbols
            {cluster_id: [ticker1, ticker2, ...]}
        """
        # Compute distance matrix (method-specific)
        distance = self.compute_distance_matrix(returns)

        # Ensure it's a condensed distance array for linkage
        if distance.ndim == 2:
            # Square matrix -> condensed
            distance = squareform(distance)

        # Hierarchical clustering
        linkage_matrix = linkage(distance, method=self.linkage_method)

        # Cut dendrogram to satisfy cluster size constraint
        clusters = self._cut_dendrogram(
            linkage_matrix,
            returns.columns.tolist()
        )

        return clusters

    def _cut_dendrogram(self,
                       Z: np.ndarray,
                       labels: List[str]) -> Dict[str, List[str]]:
        """
        Cut dendrogram to enforce cluster size constraints.

        Uses iterative approach: find optimal number of clusters that satisfies
        both min and max cluster size constraints.

        Args:
            Z: Linkage matrix from scipy
            labels: Asset labels (ticker symbols)

        Returns:
            Dictionary of clusters
        """
        n = len(labels)

        # Start with reasonable number of clusters
        n_clusters = max(1, n // self.max_cluster_size)

        best_result = None
        best_violation_score = float('inf')

        # Iteratively adjust to find best configuration
        for _ in range(100):  # Safety limit
            cluster_ids = fcluster(Z, n_clusters, criterion='maxclust')

            # Check cluster sizes
            unique_ids, counts = np.unique(cluster_ids, return_counts=True)

            # Count constraint violations and distance from target
            max_violations = np.sum(counts > self.max_cluster_size)
            
            # Calculate how far from target average cluster size
            avg_size = np.mean(counts)
            target_penalty = abs(avg_size - self.target_cluster_size)
            
            # Total score: violations + distance from target (weighted lower)
            violation_score = max_violations * 1000 + target_penalty

            # Track best solution
            if violation_score < best_violation_score:
                best_violation_score = violation_score
                best_result = cluster_ids.copy()

            # Check if we found a valid solution
            if np.all(counts <= self.max_cluster_size) and abs(avg_size - self.target_cluster_size) < 1:
                # All clusters satisfy both constraints
                break

            # Adjust based on violations and target
            if max_violations > 0:
                # Some clusters too large - need more granular clusters
                n_clusters += 1
            elif avg_size > self.target_cluster_size:
                # Clusters too large on average - need more clusters
                n_clusters += 1
            elif avg_size < self.target_cluster_size and n_clusters > 1:
                # Clusters too small on average - need fewer clusters
                n_clusters -= 1
            else:
                # Close enough to target
                break

            if n_clusters >= n:
                # Degenerate case: each asset is its own cluster
                break

        # Use best result found
        cluster_ids = best_result if best_result is not None else cluster_ids

        # Build cluster dictionary
        clusters = {}
        for idx, cluster_id in enumerate(cluster_ids):
            cid = f"cluster_{cluster_id}"
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(labels[idx])

        return clusters

    def validate_degree_constraint(self,
                                   clusters: Dict[str, List[str]]) -> bool:
        """
        Validate that all clusters satisfy the variable count constraint.

        Args:
            clusters: Dictionary of clusters

        Returns:
            True if all clusters are valid, False otherwise
        """
        for cluster_id, tickers in clusters.items():
            n_vars = len(tickers) * self.n_bits
            if n_vars > self.max_variables:
                return False

        return True

    def get_cluster_stats(self, clusters: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Get statistics about the clustering.

        Args:
            clusters: Dictionary of clusters

        Returns:
            Statistics dictionary
        """
        cluster_sizes = [len(tickers) for tickers in clusters.values()]
        cluster_vars = [len(tickers) * self.n_bits for tickers in clusters.values()]

        return {
            'n_clusters': len(clusters),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_variables': max(cluster_vars) if cluster_vars else 0,
            'total_assets': sum(cluster_sizes)
        }
