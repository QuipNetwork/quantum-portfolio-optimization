"""Asset clustering for quantum portfolio optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.cluster.hierarchy import linkage, fcluster


class PortfolioClustering:
    """Cluster assets to satisfy D-Wave topology constraints."""

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 target_cluster_size: int = None):
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
        self.target_cluster_size = target_cluster_size if target_cluster_size is not None else max_cluster_size // 2
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
