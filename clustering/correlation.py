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

"""Correlation-based clustering."""

import numpy as np
import pandas as pd
from typing import Dict, List
from .base import BaseClusterer


class CorrelationClusterer(BaseClusterer):
    """
    Cluster assets by correlation similarity.

    Distance metric: 1 - |correlation|
    Highly correlated assets are "close" and clustered together.
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 use_absolute: bool = True,
                 target_cluster_size: int = None,
                 max_clusters: int = None,
                 min_cluster_size: int = None):
        """
        Initialize correlation clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            use_absolute: If True, use |correlation| (ignores direction).
                         If False, negative correlation = far apart.
            target_cluster_size: Target average cluster size
            max_clusters: Maximum number of clusters (hard constraint)
        """
        super().__init__(max_cluster_size, n_bits, linkage_method,
                        target_cluster_size=target_cluster_size,
                        max_clusters=max_clusters,
                        min_cluster_size=min_cluster_size)
        self.use_absolute = use_absolute

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute correlation-based distance matrix.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Distance matrix (N × N)
        """
        # Compute correlation matrix
        corr = returns.corr()

        # Replace NaN values with 0 (treat as uncorrelated if insufficient data)
        corr = corr.fillna(0)

        # Distance: 1 - correlation
        if self.use_absolute:
            distance = 1 - corr.abs().values
        else:
            distance = 1 - corr.values

        # Ensure distance is symmetric and zero-diagonal
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2

        # Clip to [0, 2] (for negative correlations without abs)
        distance = np.clip(distance, 0, 2)

        return distance

    def cluster(self, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Cluster assets using correlation. Tries hierarchical first, then falls back to greedy best-fit.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dictionary mapping cluster IDs to lists of ticker symbols
        """
        # Try hierarchical clustering first
        try:
            return super().cluster(returns)
        except ValueError as e:
            # If hierarchical fails due to constraints, use greedy best-fit
            if "Cannot satisfy" in str(e):
                return self._greedy_best_fit_cluster(returns)
            else:
                raise

    def _greedy_best_fit_cluster(self, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Greedy best-fit clustering that respects all constraints.

        Strategy:
        1. Compute pairwise correlations
        2. For each asset, find the cluster with highest average correlation
        3. Add to that cluster if it doesn't violate constraints
        4. Otherwise, start a new cluster

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dictionary of clusters
        """
        tickers = returns.columns.tolist()
        n_assets = len(tickers)

        # Compute correlation matrix
        corr_matrix = returns.corr().values

        # Start with first asset in first cluster
        clusters = {"cluster_1": [tickers[0]]}
        cluster_counter = 1

        # Process remaining assets in order of decreasing average correlation
        # (most correlated assets first - they're easiest to place)
        avg_correlations = []
        for i in range(1, n_assets):
            avg_corr = np.mean(np.abs(corr_matrix[i, :i]))  # Avg with previous assets
            avg_correlations.append((avg_corr, tickers[i]))

        avg_correlations.sort(reverse=True)  # Highest correlation first

        for _, ticker in avg_correlations:
            ticker_idx = tickers.index(ticker)

            # Find best cluster for this asset (highest average correlation)
            best_cluster_id = None
            best_avg_corr = -1

            for cluster_id, cluster_tickers in clusters.items():
                # Check if adding would violate max_cluster_size
                if len(cluster_tickers) >= self.max_cluster_size:
                    continue

                # Compute average correlation with cluster members
                cluster_indices = [tickers.index(t) for t in cluster_tickers]
                if self.use_absolute:
                    avg_corr = np.mean([np.abs(corr_matrix[ticker_idx, idx]) for idx in cluster_indices])
                else:
                    avg_corr = np.mean([corr_matrix[ticker_idx, idx] for idx in cluster_indices])

                if avg_corr > best_avg_corr:
                    best_avg_corr = avg_corr
                    best_cluster_id = cluster_id

            # Add to best cluster or create new one
            if best_cluster_id is not None:
                clusters[best_cluster_id].append(ticker)
            else:
                # No suitable cluster - create new one
                cluster_counter += 1
                new_cluster_id = f"cluster_{cluster_counter}"
                clusters[new_cluster_id] = [ticker]

                # Check if we've exceeded max_clusters
                if self.max_clusters is not None and len(clusters) > self.max_clusters:
                    raise ValueError(
                        f"Cannot satisfy max_clusters={self.max_clusters} with greedy best-fit. "
                        f"Created {len(clusters)} clusters."
                    )

        # Validate min_cluster_size by merging small clusters
        clusters = self._merge_small_greedy(clusters, corr_matrix, tickers)

        return clusters

    def _merge_small_greedy(self,
                           clusters: Dict[str, List[str]],
                           corr_matrix: np.ndarray,
                           tickers: List[str]) -> Dict[str, List[str]]:
        """
        Merge small clusters greedily based on correlation.

        Args:
            clusters: Initial clusters
            corr_matrix: Correlation matrix
            tickers: List of all tickers

        Returns:
            Clusters with small ones merged
        """
        merged = dict(clusters)

        while True:
            # Find small clusters
            small = [(cid, ctickers) for cid, ctickers in merged.items()
                    if len(ctickers) < self.min_cluster_size]

            if not small:
                break  # No more small clusters

            # Take the smallest cluster
            small.sort(key=lambda x: len(x[1]))
            small_cid, small_tickers = small[0]

            # Find best target cluster (highest avg correlation that can accept them)
            best_target = None
            best_corr = -np.inf

            for target_cid, target_tickers in merged.items():
                if target_cid == small_cid:
                    continue

                # Check capacity
                if len(target_tickers) + len(small_tickers) > self.max_cluster_size:
                    continue

                # Compute avg correlation between clusters
                avg_corr = 0
                count = 0
                for s_ticker in small_tickers:
                    s_idx = tickers.index(s_ticker)
                    for t_ticker in target_tickers:
                        t_idx = tickers.index(t_ticker)
                        if self.use_absolute:
                            avg_corr += np.abs(corr_matrix[s_idx, t_idx])
                        else:
                            avg_corr += corr_matrix[s_idx, t_idx]
                        count += 1

                if count > 0:
                    avg_corr /= count

                if avg_corr > best_corr:
                    best_corr = avg_corr
                    best_target = target_cid

            # Merge or fail
            if best_target is not None:
                merged[best_target].extend(small_tickers)
                del merged[small_cid]
            else:
                raise ValueError(
                    f"Cannot satisfy min_cluster_size={self.min_cluster_size}. "
                    f"Cluster '{small_cid}' has {len(small_tickers)} assets but cannot be merged."
                )

        return merged
