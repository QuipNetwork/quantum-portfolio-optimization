"""Asset clustering for quantum portfolio optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.cluster.hierarchy import linkage, fcluster


class PortfolioClustering:
    """Cluster assets to satisfy D-Wave topology constraints."""

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10):
        """
        Initialize clustering algorithm.

        Args:
            max_cluster_size: Maximum assets per cluster (for Zephyr degree constraint)
            n_bits: Bits per weight variable (for discretization)

        Note:
            With n_bits=10, each asset becomes 10 binary variables.
            To keep total variables per cluster ≤180, we limit to ~18 assets.
        """
        self.max_cluster_size = max_cluster_size
        self.n_bits = n_bits
        self.max_variables = max_cluster_size * n_bits

    def cluster(self, corr_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Hierarchical clustering on correlation matrix.

        Args:
            corr_matrix: Correlation matrix (N × N)

        Returns:
            Dictionary mapping cluster IDs to lists of ticker symbols
            {cluster_id: [ticker1, ticker2, ...]}
        """
        # Distance metric: 1 - |correlation|
        # (highly correlated assets are "close")
        distance = 1 - corr_matrix.abs().values

        # Ensure distance is symmetric and zero-diagonal
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2

        # Convert to condensed distance matrix for linkage
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance)

        # Hierarchical clustering with Ward linkage
        linkage_matrix = linkage(condensed_dist, method='ward')

        # Cut dendrogram to satisfy cluster size constraint
        clusters = self._cut_dendrogram(
            linkage_matrix,
            corr_matrix.index.tolist()
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
