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

"""Graph-based clustering using correlation networks."""

import numpy as np
import pandas as pd
from typing import Dict, List
from .base import BaseClusterer


class GraphClusterer(BaseClusterer):
    """
    Cluster assets using graph community detection.

    Builds a correlation network and finds communities using modularity optimization.
    Captures natural market structures.

    Requires: networkx library (pip install networkx)
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 correlation_threshold: float = 0.5,
                 algorithm: str = 'louvain',
                 target_cluster_size: int = None,
                 max_clusters: int = None,
                 min_cluster_size: int = None):
        """
        Initialize graph-based clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            correlation_threshold: Minimum |correlation| to create edge (default 0.5)
            algorithm: Community detection algorithm ('louvain', 'greedy', 'label_prop')
        """
        super().__init__(max_cluster_size, n_bits, linkage_method='ward', target_cluster_size=target_cluster_size,
                        max_clusters=max_clusters,
                        min_cluster_size=min_cluster_size)
        self.correlation_threshold = correlation_threshold
        self.algorithm = algorithm

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Not used for graph clustering (we override cluster() method).

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dummy distance matrix
        """
        raise NotImplementedError("Graph clustering doesn't use distance matrix")

    def cluster(self, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Cluster assets using graph community detection.

        Overrides base class method.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dictionary mapping cluster IDs to ticker lists
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx not installed. "
                "Install with: pip install networkx\n"
                "Note: Graph clustering is optional. Use CorrelationClusterer for basic usage."
            )

        # Compute correlation matrix
        corr = returns.corr()
        tickers = returns.columns.tolist()

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(tickers)

        # Add edges for correlations above threshold
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                corr_val = abs(corr.iloc[i, j])

                if corr_val >= self.correlation_threshold:
                    G.add_edge(tickers[i], tickers[j], weight=corr_val)

        # Detect communities
        if self.algorithm == 'louvain':
            communities = self._louvain_communities(G)
        elif self.algorithm == 'greedy':
            communities = self._greedy_communities(G)
        elif self.algorithm == 'label_prop':
            communities = self._label_propagation(G)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Convert to cluster dictionary and enforce size constraints
        clusters = self._communities_to_clusters(communities)

        return clusters

    def _louvain_communities(self, G):
        """Louvain community detection (best modularity)."""
        try:
            import community as community_louvain  # python-louvain package
            partition = community_louvain.best_partition(G)

            # Group nodes by community
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)

            return list(communities.values())

        except ImportError:
            # Fallback to networkx greedy
            print("Warning: python-louvain not installed, using greedy modularity")
            return self._greedy_communities(G)

    def _greedy_communities(self, G):
        """Greedy modularity maximization."""
        import networkx as nx
        from networkx.algorithms import community

        communities = community.greedy_modularity_communities(G)
        return [list(c) for c in communities]

    def _label_propagation(self, G):
        """Label propagation algorithm."""
        import networkx as nx
        from networkx.algorithms import community

        communities = community.label_propagation_communities(G)
        return [list(c) for c in communities]

    def _communities_to_clusters(self, communities: List[List[str]]) -> Dict[str, List[str]]:
        """
        Convert communities to clusters, enforcing size constraints.

        Args:
            communities: List of communities (list of ticker lists)

        Returns:
            Dictionary of clusters
        """
        clusters = {}
        cluster_counter = 1

        for community in communities:
            if len(community) <= self.max_cluster_size:
                # Community fits in one cluster
                cluster_id = f"cluster_{cluster_counter}"
                clusters[cluster_id] = community
                cluster_counter += 1
            else:
                # Split large community
                n_splits = int(np.ceil(len(community) / self.max_cluster_size))

                for i in range(n_splits):
                    start_idx = i * self.max_cluster_size
                    end_idx = min((i + 1) * self.max_cluster_size, len(community))

                    cluster_id = f"cluster_{cluster_counter}"
                    clusters[cluster_id] = community[start_idx:end_idx]
                    cluster_counter += 1

        return clusters
