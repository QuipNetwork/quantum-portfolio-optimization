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

"""Aggregation strategies for combining cluster solutions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from enum import Enum


class AggregationStrategy(Enum):
    """Available aggregation strategies."""
    CONCATENATE = 'concatenate'
    PROPORTIONAL = 'proportional'
    UNIFORM = 'uniform'


class ClusterAggregator:
    """Aggregate portfolio weights from multiple clusters."""

    def __init__(self, strategy: str = 'concatenate'):
        """
        Initialize aggregator.

        Args:
            strategy: 'concatenate', 'proportional', or 'uniform'

        Strategy descriptions:
            - concatenate: Combine weights directly, then normalize (simple, no reweighting)
            - proportional: Weight by cluster size (larger clusters get more budget)
            - uniform: Equal budget to each cluster (ignores size differences)
        """
        if strategy not in [s.value for s in AggregationStrategy]:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Choose from: {[s.value for s in AggregationStrategy]}")
        self.strategy = strategy

    def aggregate(self,
                 cluster_weights: Dict[str, pd.Series],
                 clusters: Dict[str, List[str]],
                 target_budget: float = 1.0,
                 cardinality: Optional[int] = None) -> pd.Series:
        """
        Aggregate cluster solutions into global portfolio.

        Args:
            cluster_weights: {cluster_id: Series of unnormalized weights}
            clusters: {cluster_id: [tickers]} (for metadata)
            target_budget: Target sum of weights (default 1.0)
            cardinality: Maximum number of non-zero assets (optional)

        Returns:
            pd.Series with normalized global portfolio weights
        """
        if self.strategy == 'concatenate':
            global_weights = self._concatenate(cluster_weights, target_budget)

        elif self.strategy == 'proportional':
            global_weights = self._proportional(cluster_weights, clusters, target_budget)

        elif self.strategy == 'uniform':
            global_weights = self._uniform(cluster_weights, clusters, target_budget)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Apply cardinality constraint if specified
        if cardinality is not None:
            global_weights = self._apply_cardinality(global_weights, cardinality)

        # Final normalization to ensure exact budget
        if global_weights.sum() > 0:
            global_weights = (global_weights / global_weights.sum()) * target_budget

        return global_weights

    def _concatenate(self,
                    cluster_weights: Dict[str, pd.Series],
                    target_budget: float) -> pd.Series:
        """
        Concatenate + Normalize strategy.

        Simply combines all cluster weights and normalizes to target budget.
        No inter-cluster reweighting.

        Args:
            cluster_weights: {cluster_id: weights}
            target_budget: Target sum

        Returns:
            Normalized weights
        """
        # Combine all weights
        all_weights = {}
        for weights in cluster_weights.values():
            all_weights.update(weights.to_dict())

        weights_series = pd.Series(all_weights)

        # Normalize
        total = weights_series.sum()
        if total > 0:
            weights_series = (weights_series / total) * target_budget

        return weights_series

    def _proportional(self,
                     cluster_weights: Dict[str, pd.Series],
                     clusters: Dict[str, List[str]],
                     target_budget: float) -> pd.Series:
        """
        Proportional strategy (size-weighted).

        Allocates budget proportionally to cluster sizes.
        Larger clusters get more budget.

        Args:
            cluster_weights: {cluster_id: weights}
            clusters: {cluster_id: tickers}
            target_budget: Target sum

        Returns:
            Weighted and normalized portfolio
        """
        # Calculate cluster sizes
        cluster_sizes = {cid: len(tickers) for cid, tickers in clusters.items()}
        total_assets = sum(cluster_sizes.values())

        # Allocate budget proportionally
        all_weights = {}

        for cluster_id, weights in cluster_weights.items():
            # Budget for this cluster = (cluster_size / total_assets) * target_budget
            cluster_budget = (cluster_sizes[cluster_id] / total_assets) * target_budget

            # Normalize cluster weights to cluster budget
            cluster_sum = weights.sum()
            if cluster_sum > 0:
                scaled_weights = (weights / cluster_sum) * cluster_budget
                all_weights.update(scaled_weights.to_dict())
            else:
                all_weights.update(weights.to_dict())

        return pd.Series(all_weights)

    def _uniform(self,
                cluster_weights: Dict[str, pd.Series],
                clusters: Dict[str, List[str]],
                target_budget: float) -> pd.Series:
        """
        Uniform strategy (equal budget per cluster).

        Allocates equal budget to each cluster regardless of size.
        Good when clusters represent different strategies or sectors.

        Args:
            cluster_weights: {cluster_id: weights}
            clusters: {cluster_id: tickers}
            target_budget: Target sum

        Returns:
            Uniformly weighted portfolio
        """
        n_clusters = len(cluster_weights)
        budget_per_cluster = target_budget / n_clusters

        all_weights = {}

        for cluster_id, weights in cluster_weights.items():
            # Normalize cluster to equal budget
            cluster_sum = weights.sum()
            if cluster_sum > 0:
                scaled_weights = (weights / cluster_sum) * budget_per_cluster
                all_weights.update(scaled_weights.to_dict())
            else:
                all_weights.update(weights.to_dict())

        return pd.Series(all_weights)

    def _apply_cardinality(self,
                          weights: pd.Series,
                          cardinality: int) -> pd.Series:
        """
        Apply cardinality constraint by keeping top-k assets.

        Args:
            weights: Portfolio weights
            cardinality: Maximum number of non-zero assets

        Returns:
            Sparse portfolio with at most k assets
        """
        if cardinality <= 0 or cardinality >= len(weights):
            return weights

        # Get top k assets by weight
        top_k_assets = weights.nlargest(cardinality).index

        # Zero out all other assets
        sparse_weights = pd.Series(0.0, index=weights.index)
        sparse_weights[top_k_assets] = weights[top_k_assets]

        return sparse_weights

    def get_allocation_info(self,
                          cluster_weights: Dict[str, pd.Series],
                          clusters: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Get information about budget allocation across clusters.

        Args:
            cluster_weights: {cluster_id: weights}
            clusters: {cluster_id: tickers}

        Returns:
            DataFrame with cluster statistics
        """
        info = []

        for cluster_id, weights in cluster_weights.items():
            tickers = clusters[cluster_id]

            info.append({
                'cluster_id': cluster_id,
                'n_assets': len(tickers),
                'n_nonzero': (weights > 0).sum(),
                'total_weight': weights.sum(),
                'mean_weight': weights.mean(),
                'max_weight': weights.max(),
                'sparsity': (weights == 0).sum() / len(weights)
            })

        return pd.DataFrame(info)

    def compare_strategies(self,
                          cluster_weights: Dict[str, pd.Series],
                          clusters: Dict[str, List[str]],
                          target_budget: float = 1.0) -> Dict[str, pd.Series]:
        """
        Compare all aggregation strategies side-by-side.

        Args:
            cluster_weights: {cluster_id: weights}
            clusters: {cluster_id: tickers}
            target_budget: Target sum

        Returns:
            {strategy_name: global_weights}
        """
        results = {}

        for strategy in AggregationStrategy:
            aggregator = ClusterAggregator(strategy.value)
            weights = aggregator.aggregate(cluster_weights, clusters, target_budget)
            results[strategy.value] = weights

        return results

    def get_cluster_contributions(self,
                                 global_weights: pd.Series,
                                 clusters: Dict[str, List[str]]) -> pd.Series:
        """
        Calculate how much each cluster contributes to global portfolio.

        Args:
            global_weights: Final portfolio weights
            clusters: {cluster_id: tickers}

        Returns:
            pd.Series with total weight per cluster
        """
        contributions = {}

        for cluster_id, tickers in clusters.items():
            cluster_total = sum(global_weights.get(ticker, 0) for ticker in tickers)
            contributions[cluster_id] = cluster_total

        return pd.Series(contributions)
