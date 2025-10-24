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

"""Uniform clustering: uniformly assign assets to clusters (baseline/random)."""

import numpy as np
import pandas as pd
from typing import Dict, List
from .base import BaseClusterer


class UniformClusterer(BaseClusterer):
    """
    Uniformly assign assets to clusters.

    This is a baseline/control clustering method that distributes assets
    uniformly across clusters without considering any relationships between them.
    Useful as a null hypothesis or baseline for comparing other clustering methods.
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 target_cluster_size: int = None,
                 max_clusters: int = None,
                 min_cluster_size: int = None,
                 random_state: int = None):
        """
        Initialize uniform clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster (HARD constraint)
            n_bits: Bits per weight variable
            linkage_method: Ignored (kept for API compatibility)
            target_cluster_size: Target average cluster size
            max_clusters: Maximum number of clusters (HARD constraint)
            min_cluster_size: Minimum assets per cluster (HARD constraint)
            random_state: Random seed for reproducibility
        """
        super().__init__(max_cluster_size, n_bits, linkage_method,
                        target_cluster_size=target_cluster_size,
                        max_clusters=max_clusters,
                        min_cluster_size=min_cluster_size)
        self.random_state = random_state

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Not used for uniform clustering.

        Raises:
            NotImplementedError: Uniform clustering doesn't use distance matrix
        """
        raise NotImplementedError("Uniform clustering doesn't use distance matrix")

    def cluster(self, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Uniformly assign assets to clusters.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dictionary mapping cluster IDs to lists of ticker symbols

        Raises:
            ValueError: If constraints cannot be satisfied
        """
        tickers = returns.columns.tolist()
        n_assets = len(tickers)

        # Determine number of clusters
        if self.max_clusters is not None:
            # Use max_clusters if specified
            n_clusters = self.max_clusters
        else:
            # Calculate optimal number based on target_cluster_size
            n_clusters = max(1, int(np.ceil(n_assets / self.target_cluster_size)))

        # Validate constraints
        min_clusters_needed = int(np.ceil(n_assets / self.max_cluster_size))
        if n_clusters < min_clusters_needed:
            raise ValueError(
                f"Cannot satisfy constraints: {n_assets} assets with max_cluster_size={self.max_cluster_size} "
                f"requires at least {min_clusters_needed} clusters, but n_clusters={n_clusters}."
            )

        max_clusters_allowed = n_assets // self.min_cluster_size
        if n_clusters > max_clusters_allowed:
            raise ValueError(
                f"Cannot satisfy constraints: {n_assets} assets with min_cluster_size={self.min_cluster_size} "
                f"allows at most {max_clusters_allowed} clusters, but n_clusters={n_clusters}."
            )

        # Shuffle assets randomly
        rng = np.random.RandomState(self.random_state)
        shuffled_tickers = rng.permutation(tickers).tolist()

        # Distribute uniformly across clusters
        clusters = {}
        for i, ticker in enumerate(shuffled_tickers):
            cluster_id = f"cluster_{i % n_clusters + 1}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(ticker)

        # Validate final constraints
        cluster_sizes = [len(tickers) for tickers in clusters.values()]
        if max(cluster_sizes) > self.max_cluster_size:
            raise ValueError(
                f"Cannot satisfy max_cluster_size={self.max_cluster_size}. "
                f"Largest cluster has {max(cluster_sizes)} assets."
            )

        if min(cluster_sizes) < self.min_cluster_size:
            raise ValueError(
                f"Cannot satisfy min_cluster_size={self.min_cluster_size}. "
                f"Smallest cluster has {min(cluster_sizes)} assets."
            )

        return clusters
