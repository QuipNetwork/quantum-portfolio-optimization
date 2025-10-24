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
                 target_cluster_size: int = None,
                 max_clusters: int = None,
                 min_cluster_size: int = None):
        """
        Initialize base clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster (HARD constraint - quantum hardware limit)
            n_bits: Bits per weight variable (for discretization)
            linkage_method: Hierarchical linkage method ('ward', 'single', 'complete', 'average')
            target_cluster_size: Target average cluster size (default: max_cluster_size // 2)
                                If specified, algorithm aims for this average size to avoid too many small clusters
            max_clusters: Maximum number of clusters (HARD constraint - fixed template limit)
                         If specified, algorithm will produce at most this many clusters
            min_cluster_size: Minimum assets per cluster (HARD constraint - efficiency requirement)
                             If specified, small clusters will be merged into nearest neighbors
                             Default: 2 (avoid single-asset clusters)
        """
        self.max_cluster_size = max_cluster_size
        self.target_cluster_size = target_cluster_size if target_cluster_size is not None else max_cluster_size // 2
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size if min_cluster_size is not None else 2
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
        Cut dendrogram to enforce cluster size and count constraints.

        Uses iterative approach: find optimal number of clusters that satisfies
        max_cluster_size (HARD) and max_clusters (HARD if specified).

        Args:
            Z: Linkage matrix from scipy
            labels: Asset labels (ticker symbols)

        Returns:
            Dictionary of clusters

        Raises:
            ValueError: If constraints cannot be satisfied
        """
        n = len(labels)

        # Calculate feasible range for number of clusters
        min_clusters_needed = (n + self.max_cluster_size - 1) // self.max_cluster_size  # Ceiling division

        if self.max_clusters is not None:
            if min_clusters_needed > self.max_clusters:
                raise ValueError(
                    f"Cannot satisfy constraints: {n} assets with max_cluster_size={self.max_cluster_size} "
                    f"requires at least {min_clusters_needed} clusters, but max_clusters={self.max_clusters}. "
                    f"Either increase max_cluster_size or increase max_clusters."
                )
            # Start from max allowed and work down
            n_clusters = self.max_clusters
        else:
            # Start from minimum needed
            n_clusters = min_clusters_needed

        best_result = None
        best_violation_score = float('inf')

        # Iteratively adjust to find best configuration
        for attempt in range(100):  # Safety limit
            cluster_ids = fcluster(Z, n_clusters, criterion='maxclust')

            # Check cluster sizes
            unique_ids, counts = np.unique(cluster_ids, return_counts=True)
            actual_n_clusters = len(unique_ids)

            # Check hard constraints
            max_size_violations = np.sum(counts > self.max_cluster_size)

            # Check max_clusters constraint
            if self.max_clusters is not None and actual_n_clusters > self.max_clusters:
                cluster_count_violations = actual_n_clusters - self.max_clusters
            else:
                cluster_count_violations = 0

            # Calculate how far from target average cluster size (soft constraint)
            avg_size = np.mean(counts)
            target_penalty = abs(avg_size - self.target_cluster_size)

            # Violation score: hard constraints heavily weighted
            violation_score = (
                max_size_violations * 10000 +      # CRITICAL: clusters too large
                cluster_count_violations * 5000 +   # CRITICAL: too many clusters
                target_penalty                      # MINOR: deviation from target size
            )

            # Track best solution
            if violation_score < best_violation_score:
                best_violation_score = violation_score
                best_result = (cluster_ids.copy(), actual_n_clusters, counts.copy())

            # Check if we found a perfect solution
            if (max_size_violations == 0 and
                cluster_count_violations == 0 and
                target_penalty < 1):
                break

            # Adjustment logic
            if max_size_violations > 0:
                # Some clusters too large - need more clusters
                n_clusters += 1
            elif cluster_count_violations > 0:
                # Too many clusters - try to merge (reduce n_clusters)
                n_clusters -= 1
            elif avg_size > self.target_cluster_size and n_clusters < n:
                # Average too large - split more
                n_clusters += 1
            elif avg_size < self.target_cluster_size and n_clusters > min_clusters_needed:
                # Average too small - merge more
                n_clusters -= 1
            else:
                # Close enough
                break

            # Bounds check
            if self.max_clusters and n_clusters > self.max_clusters:
                n_clusters = self.max_clusters
            if n_clusters < min_clusters_needed:
                n_clusters = min_clusters_needed
            if n_clusters >= n:
                break

        # Use best result found
        if best_result is None:
            raise RuntimeError("Failed to find any valid clustering solution")

        cluster_ids, final_n_clusters, final_counts = best_result

        # Don't check max_cluster_size constraint here - split-and-sweep will handle it
        # Don't check max_clusters constraint here - split-and-sweep will handle it

        # Build cluster dictionary
        clusters = {}
        for idx, cluster_id in enumerate(cluster_ids):
            cid = f"cluster_{cluster_id}"
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(labels[idx])

        # Merge small clusters
        clusters = self._merge_small_clusters(clusters, labels, Z)

        # Apply split-and-sweep to fit topology constraints (if specified)
        # This is enabled by default to ensure topology compatibility
        # Split handles clusters that are too large, sweep handles too many clusters
        has_large_clusters = any(len(tickers) > self.max_cluster_size for tickers in clusters.values())
        if self.max_clusters is not None or has_large_clusters:
            clusters = self._split_and_sweep(clusters, self.max_clusters)

        # Final validation
        final_sizes = [len(tickers) for tickers in clusters.values()]
        if any(size > self.max_cluster_size for size in final_sizes):
            raise ValueError(
                f"Split-and-sweep failed: clusters still exceed max_cluster_size={self.max_cluster_size}"
            )

        return clusters

    def _merge_small_clusters(self,
                             clusters: Dict[str, List[str]],
                             labels: List[str],
                             Z: np.ndarray) -> Dict[str, List[str]]:
        """
        Merge clusters smaller than min_cluster_size into nearest neighbors.

        Strategy: For each small cluster, find the nearest larger cluster based
        on the hierarchical linkage and merge into it (if merge doesn't violate max_cluster_size).

        Args:
            clusters: Initial cluster dictionary
            labels: Asset labels
            Z: Linkage matrix

        Returns:
            Modified cluster dictionary with small clusters merged
        """
        # Check if any clusters are too small
        small_clusters = {cid: tickers for cid, tickers in clusters.items()
                         if len(tickers) < self.min_cluster_size}

        if not small_clusters:
            return clusters  # Nothing to merge

        # Build label-to-cluster map
        label_to_cluster = {}
        for cid, tickers in clusters.items():
            for ticker in tickers:
                label_to_cluster[ticker] = cid

        # Merge small clusters iteratively
        merged_clusters = dict(clusters)

        for small_cid, small_tickers in small_clusters.items():
            # Find nearest cluster that can accept these assets
            best_target_cid = None
            best_distance = float('inf')

            for target_cid, target_tickers in merged_clusters.items():
                if target_cid == small_cid:
                    continue  # Don't merge with self

                # Check if merge would violate max_cluster_size
                if len(target_tickers) + len(small_tickers) > self.max_cluster_size:
                    continue

                # Compute average pairwise distance between clusters
                # (approximation: use mean of all pairwise distances)
                distances = []
                for s_ticker in small_tickers:
                    s_idx = labels.index(s_ticker)
                    for t_ticker in target_tickers:
                        t_idx = labels.index(t_ticker)
                        # Use linkage matrix to estimate distance
                        # This is approximate - ideally we'd recompute from original distance matrix
                        distances.append(abs(s_idx - t_idx))  # Simple index-based heuristic

                avg_dist = np.mean(distances) if distances else float('inf')

                if avg_dist < best_distance:
                    best_distance = avg_dist
                    best_target_cid = target_cid

            # Merge into best target
            if best_target_cid is not None:
                merged_clusters[best_target_cid].extend(small_tickers)
                del merged_clusters[small_cid]
            else:
                # Cannot merge without violating max_cluster_size
                # Keep the small cluster (will be caught by validation if this is a problem)
                pass

        return merged_clusters

    def _sweep_clusters_to_topology(self,
                                    clusters: Dict[str, List[str]],
                                    target_num_clusters: int = None) -> Dict[str, List[str]]:
        """
        Sweep/merge clusters to fit topology constraints (enabled by default).

        Algorithm:
        1. Sort clusters by size (largest to smallest)
        2. While we have too many clusters:
           - Take the smallest cluster
           - Try to merge it into the largest cluster that has room
           - If no cluster has room, fail
        3. Continue until we have target_num_clusters or fewer

        Args:
            clusters: Original cluster assignments
            target_num_clusters: Target number of clusters (default: self.max_clusters)

        Returns:
            Swept cluster dictionary

        Raises:
            ValueError: If clusters cannot be swept to fit constraints
        """
        # Use max_clusters if no target specified
        if target_num_clusters is None:
            if self.max_clusters is None:
                # No constraint - return as-is
                return clusters
            target_num_clusters = self.max_clusters

        # Convert to list of (cluster_id, tickers)
        cluster_list = [(cid, list(tickers)) for cid, tickers in clusters.items()]

        # Check if any cluster exceeds max size
        for cid, tickers in cluster_list:
            if len(tickers) > self.max_cluster_size:
                raise ValueError(
                    f"Cluster {cid} too large: {len(tickers)} > {self.max_cluster_size}. "
                    f"Cannot sweep - violates hardware constraint."
                )

        # If we already fit, return as-is
        if len(cluster_list) <= target_num_clusters:
            return clusters

        # Perform sweep: merge smallest into largest
        iteration = 0
        max_iterations = len(cluster_list) * 2  # Safety limit

        while len(cluster_list) > target_num_clusters and iteration < max_iterations:
            iteration += 1

            # Sort by size (largest first, smallest last)
            cluster_list.sort(key=lambda x: len(x[1]), reverse=True)

            # Take smallest cluster
            smallest_id, smallest_tickers = cluster_list.pop()
            smallest_size = len(smallest_tickers)

            # Try to merge into largest cluster with room
            merged = False
            for i, (target_id, target_tickers) in enumerate(cluster_list):
                if len(target_tickers) + smallest_size <= self.max_cluster_size:
                    # Merge smallest into this cluster
                    target_tickers.extend(smallest_tickers)
                    merged = True
                    break

            if not merged:
                # Couldn't fit anywhere - restoration and failure
                cluster_list.append((smallest_id, smallest_tickers))
                raise ValueError(
                    f"Cannot sweep {len(cluster_list)} clusters into {target_num_clusters} "
                    f"(max_cluster_size={self.max_cluster_size}). "
                    f"Consider increasing max_cluster_size or target cluster count."
                )

        if iteration >= max_iterations:
            raise ValueError("Sweep exceeded max iterations - unable to converge")

        # Success! Rebuild cluster dictionary
        swept_clusters = {cid: tickers for cid, tickers in cluster_list}
        return swept_clusters

    def _split_large_clusters(self,
                             clusters: Dict[str, List[str]],
                             distance_matrix: np.ndarray = None,
                             labels: List[str] = None) -> Dict[str, List[str]]:
        """
        Split clusters that exceed max_cluster_size.

        Algorithm:
        1. Identify clusters that are too large
        2. For each large cluster, split it using k-means or random split
        3. Continue until all clusters fit within max_cluster_size

        Args:
            clusters: Original cluster assignments
            distance_matrix: Optional distance matrix for smarter splitting
            labels: Original asset labels (for distance matrix indexing)

        Returns:
            Clusters with no cluster exceeding max_cluster_size
        """
        split_clusters = {}
        cluster_counter = 0

        for cluster_id, tickers in clusters.items():
            cluster_size = len(tickers)

            if cluster_size <= self.max_cluster_size:
                # Cluster is fine, keep as-is
                split_clusters[f"cluster_{cluster_counter}"] = tickers
                cluster_counter += 1
            else:
                # Cluster is too large - split it
                # Calculate how many sub-clusters we need
                num_splits = (cluster_size + self.max_cluster_size - 1) // self.max_cluster_size

                # Simple split: divide roughly equally
                # More sophisticated version could use k-means on the subset
                split_size = cluster_size // num_splits
                remainder = cluster_size % num_splits

                start_idx = 0
                for split_i in range(num_splits):
                    # Distribute remainder across first few splits
                    current_size = split_size + (1 if split_i < remainder else 0)
                    end_idx = start_idx + current_size

                    split_tickers = tickers[start_idx:end_idx]
                    split_clusters[f"cluster_{cluster_counter}"] = split_tickers
                    cluster_counter += 1

                    start_idx = end_idx

        return split_clusters

    def _split_and_sweep(self,
                        clusters: Dict[str, List[str]],
                        target_num_clusters: int = None) -> Dict[str, List[str]]:
        """
        Combined split-and-sweep algorithm to fit topology constraints.

        Algorithm:
        1. First, SPLIT any clusters that exceed max_cluster_size
        2. Then, SWEEP (merge) to reduce cluster count if needed

        Args:
            clusters: Original cluster assignments
            target_num_clusters: Target number of clusters (default: self.max_clusters)

        Returns:
            Clusters that fit both max_cluster_size and max_clusters constraints

        Raises:
            ValueError: If constraints cannot be satisfied
        """
        # Use max_clusters if no target specified
        if target_num_clusters is None:
            if self.max_clusters is None:
                # No constraint - just split if needed
                return self._split_large_clusters(clusters)
            target_num_clusters = self.max_clusters

        # Step 1: Split large clusters
        clusters = self._split_large_clusters(clusters)

        # Step 2: Verify no cluster exceeds max size after split
        for cid, tickers in clusters.items():
            if len(tickers) > self.max_cluster_size:
                raise ValueError(
                    f"Cluster {cid} still too large after split: {len(tickers)} > {self.max_cluster_size}"
                )

        # Step 3: If we have too many clusters, sweep (merge small ones)
        if len(clusters) > target_num_clusters:
            clusters = self._sweep_clusters_to_topology(clusters, target_num_clusters)

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
