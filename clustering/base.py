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
                 linkage_method: str = 'ward'):
        """
        Initialize base clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster (for Zephyr degree constraint)
            n_bits: Bits per weight variable (for discretization)
            linkage_method: Hierarchical linkage method ('ward', 'single', 'complete', 'average')
        """
        self.max_cluster_size = max_cluster_size
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
        Cut dendrogram to enforce cluster size constraint.

        Uses iterative approach: start with minimal clusters,
        increase until all clusters satisfy max_cluster_size.

        Args:
            Z: Linkage matrix from scipy
            labels: Asset labels (ticker symbols)

        Returns:
            Dictionary of clusters
        """
        n = len(labels)

        # Start with reasonable number of clusters
        n_clusters = max(1, n // self.max_cluster_size)

        # Iteratively adjust until constraint is satisfied
        for _ in range(100):  # Safety limit
            cluster_ids = fcluster(Z, n_clusters, criterion='maxclust')

            # Check cluster sizes
            unique_ids, counts = np.unique(cluster_ids, return_counts=True)

            if np.all(counts <= self.max_cluster_size):
                # All clusters satisfy constraint
                break

            # Need more granular clusters
            n_clusters += 1

            if n_clusters >= n:
                # Degenerate case: each asset is its own cluster
                break

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
