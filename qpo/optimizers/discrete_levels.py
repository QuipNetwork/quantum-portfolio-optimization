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

"""Discrete levels optimizer wrapper for backtesting compatibility."""

import time
import logging
import numpy as np
import pandas as pd
from typing import Optional, Any, Dict, List, Tuple
import dimod

from qpo.qubo.discrete_levels import DiscreteLevelFormulator, CachedEmbeddingManager
from qpo.qubo.aggregation import ClusterAggregator
from qpo.utils.constants import (
    TRADING_DAYS_PER_YEAR,
    DEFAULT_N_LEVELS,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_THERMOMETER_PENALTY,
    DEFAULT_MAX_CLUSTER_SIZE
)
from clustering import SectorClusterer

logger = logging.getLogger(__name__)


class DiscreteLevelsOptimizer:
    """
    Portfolio optimizer using discrete weight levels with sector clustering.

    Compatible with the Backtester framework for comparison with classical
    and other quantum optimizers.
    """

    def __init__(self,
                 n_levels: int = DEFAULT_N_LEVELS,
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA,
                 thermometer_penalty: float = DEFAULT_THERMOMETER_PENALTY,
                 solver_type: str = 'simulated',
                 num_reads: Optional[int] = None,
                 num_sweeps: Optional[int] = None,
                 annealing_time: Optional[float] = None,
                 max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
                 portfolio_info_csv: str = "portfolio-info.csv",
                 clusterer: Optional[Any] = None,
                 expected_assets_per_cluster: Optional[int] = None,
                 expected_n_clusters: Optional[int] = None,
                 auto_select_template: bool = True,
                 l1_sparsity_penalty: float = 1.0,
                 use_thermometer_cutoff: bool = False):
        """
        Initialize discrete levels optimizer.

        Args:
            n_levels: Number of discrete weight levels (default 6)
            alpha: Return coefficient (default 10.0)
            beta: Risk coefficient (default 2.0)
            thermometer_penalty: Penalty for invalid thermometer encoding
            solver_type: 'simulated' or 'qpu'
            num_reads: Number of annealing samples (default: 256 for SA, 10 for QPU)
            num_sweeps: Sweeps for simulated annealing (default 256, ignored for QPU)
            annealing_time: Annealing time in microseconds for QPU (default: 20, min: 0.5, max: 2000)
            max_cluster_size: Maximum assets per cluster (19 for native embedding)
            portfolio_info_csv: Path to portfolio info for sector data
            clusterer: Custom clusterer (default: SectorClusterer)
            expected_assets_per_cluster: For QPU with templates, pad clusters to this size
            expected_n_clusters: For QPU with templates, pad to this many clusters
            auto_select_template: Automatically select best template for portfolio size
            l1_sparsity_penalty: L1 regularization penalty (λ > 0 promotes sparsity, try 0.01-1.0)
                                Adds +λ·||w||₁ to objective → fewer holdings
            use_thermometer_cutoff: If True, keep only assets with maximum thermometer level
        """
        self.n_levels = n_levels
        self.alpha = alpha
        self.beta = beta
        self.thermometer_penalty = thermometer_penalty
        self.solver_type = solver_type
        self.l1_sparsity_penalty = l1_sparsity_penalty
        self.use_thermometer_cutoff = use_thermometer_cutoff

        # Set num_reads based on solver type if not specified
        if num_reads is None:
            self.num_reads = 10 if solver_type == 'qpu' else 512
        else:
            self.num_reads = num_reads

        # Set num_sweeps based on solver type if not specified
        if num_sweeps is None:
            self.num_sweeps = 512 if solver_type == 'simulated' else 0
        else:
            self.num_sweeps = num_sweeps

        # Set annealing time (QPU only, in microseconds)
        self.annealing_time = annealing_time if annealing_time is not None else 5.0

        self.max_cluster_size = max_cluster_size
        self.portfolio_info_csv = portfolio_info_csv
        self.expected_assets_per_cluster = expected_assets_per_cluster
        self.expected_n_clusters = expected_n_clusters
        self.auto_select_template = auto_select_template

        # Create formulator
        self.formulator = DiscreteLevelFormulator(
            n_levels=n_levels,
            alpha=alpha,
            beta=beta,
            thermometer_penalty=thermometer_penalty,
            l1_sparsity_penalty=l1_sparsity_penalty
        )

        # Create embedding manager
        self.embedding_mgr = CachedEmbeddingManager()

        # Create clusterer
        if clusterer is None:
            from clustering import CorrelationClusterer
            self.clusterer = CorrelationClusterer(
                target_cluster_size=max_cluster_size
            )
        else:
            self.clusterer = clusterer

        # Aggregator
        self.aggregator = ClusterAggregator(strategy='concatenate')

        # QPU sampler (reuse connection across clusters)
        self._qpu_sampler = None

    def _pad_clusters_to_size(self, clusters: dict, target_size_per_cluster: int, mu: pd.Series, Sigma: pd.DataFrame, target_n_clusters: int = None):
        """
        Pad clusters with dummy assets to reach target size.

        Dummy assets have zero returns and zero covariance, so they'll get zero weight.
        This allows using pre-computed templates that expect a specific cluster size.

        Args:
            clusters: Dict of cluster_id -> list of tickers
            target_size_per_cluster: Target number of assets per cluster
            mu: Returns series (will be extended with zeros for dummy assets)
            Sigma: Covariance matrix (will be extended with zeros for dummy assets)
            target_n_clusters: If specified, add empty clusters to reach this count

        Returns:
            padded_clusters, padded_mu, padded_Sigma, real_tickers
        """
        padded_clusters = {}
        real_tickers = set(mu.index)
        dummy_counter = 0

        # Pad existing clusters to target size
        for cluster_id, tickers in clusters.items():
            n_real = len(tickers)
            n_padding = max(0, target_size_per_cluster - n_real)

            padded_tickers = list(tickers)

            # Add dummy assets to reach target size
            for i in range(n_padding):
                dummy_ticker = f"_DUMMY_{dummy_counter}"
                padded_tickers.append(dummy_ticker)
                dummy_counter += 1

            padded_clusters[cluster_id] = padded_tickers

        # Add empty clusters if needed to reach target count
        if target_n_clusters is not None:
            current_n_clusters = len(padded_clusters)
            if current_n_clusters < target_n_clusters:
                for i in range(target_n_clusters - current_n_clusters):
                    # Create empty cluster with only dummy assets
                    empty_cluster_id = f"_EMPTY_CLUSTER_{i}"
                    empty_cluster = []

                    for j in range(target_size_per_cluster):
                        dummy_ticker = f"_DUMMY_{dummy_counter}"
                        empty_cluster.append(dummy_ticker)
                        dummy_counter += 1

                    padded_clusters[empty_cluster_id] = empty_cluster

        # Extend mu with small negative returns for dummy assets
        # This ensures they won't be selected AND prevents all-zero BQMs
        all_tickers = []
        for tickers in padded_clusters.values():
            all_tickers.extend(tickers)

        dummy_tickers = [t for t in all_tickers if t.startswith('_DUMMY_')]

        padded_mu = mu.copy()
        for dummy in dummy_tickers:
            # Small negative return: actively discouraged but prevents zero BQM
            padded_mu[dummy] = -1e-6  # Tiny penalty

        # Extend Sigma with zeros for dummy assets
        # Build dummy covariance matrix efficiently (avoid fragmentation)
        if dummy_tickers:
            # Create dummy rows/columns all at once
            n_dummies = len(dummy_tickers)
            n_existing = len(Sigma)

            # Create zero matrix for dummy cross-covariances
            dummy_block = pd.DataFrame(
                0.0,
                index=dummy_tickers,
                columns=list(Sigma.columns) + dummy_tickers
            )

            # Create dummy-to-existing covariances (all zeros)
            existing_to_dummy = pd.DataFrame(
                0.0,
                index=Sigma.index,
                columns=dummy_tickers
            )

            # Set tiny variance on diagonal for dummy assets
            for dummy in dummy_tickers:
                dummy_block.loc[dummy, dummy] = 1e-10

            # Concatenate efficiently
            padded_Sigma = pd.concat([
                pd.concat([Sigma, existing_to_dummy], axis=1),
                dummy_block
            ], axis=0)
        else:
            padded_Sigma = Sigma.copy()

        return padded_clusters, padded_mu, padded_Sigma, real_tickers

    def _relabel_bqm_for_template(
        self,
        unified_bqm: dimod.BinaryQuadraticModel,
        cluster_info: List[Dict]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Relabel BQM variables from ticker-based names to template format.

        BQM uses: TICKER_LEVEL (e.g., "ZTS_5") + CLUSTER_ID_LEVEL for meta-cluster
        Template uses: C{cluster_idx}_ASSET_{asset_idx}_{level} + META_{cluster_idx}_{level}

        Args:
            unified_bqm: BQM with ticker-based variable names
            cluster_info: List of cluster metadata dicts

        Returns:
            (relabeling dict, reverse_relabeling dict)
        """
        # Build mapping: ticker → (cluster_idx, asset_idx_within_cluster)
        # and cluster_id → cluster_idx
        ticker_to_location = {}
        cluster_id_to_idx = {}

        # Process asset clusters (exclude META_CLUSTER)
        asset_clusters = [c for c in cluster_info if c['id'] != 'META_CLUSTER']
        for cluster_idx, cluster_data in enumerate(asset_clusters):
            cluster_id = cluster_data['id']
            tickers = cluster_data['tickers']
            cluster_id_to_idx[cluster_id] = cluster_idx
            for asset_idx, ticker in enumerate(tickers):
                ticker_to_location[ticker] = (cluster_idx, asset_idx)

        # Relabel BQM variables
        relabeling = {}
        for var in unified_bqm.variables:
            # Debug CSCO specifically
            if var.startswith('CSCO'):
                logger.debug(f"DEBUG: Processing variable {var}")
                logger.debug(f"  CSCO in ticker_to_location: {'CSCO' in ticker_to_location}")
                if 'CSCO' in ticker_to_location:
                    logger.debug(f"  CSCO location: {ticker_to_location['CSCO']}")
                logger.debug(f"  cluster_ids: {list(cluster_id_to_idx.keys())}")

            # Check if it's a meta-cluster variable (cluster_id with level)
            is_meta = False
            for cluster_id in cluster_id_to_idx.keys():
                if var.startswith(str(cluster_id) + '_') and var.split('_')[-1].isdigit():
                    # Found meta-cluster variable: cluster_id_level → META_cluster_idx_level
                    level = var.split('_')[-1]
                    cluster_idx = cluster_id_to_idx[cluster_id]
                    template_var = f"META_{cluster_idx}_{level}"
                    relabeling[var] = template_var
                    is_meta = True

                    if var.startswith('CSCO'):
                        logger.debug(f"  DEBUG: CSCO matched as meta-cluster! cluster_id={cluster_id}")
                    break

            if not is_meta:
                # Asset variable: TICKER_LEVEL → C{cluster_idx}_ASSET_{asset_idx}_{level}
                # Important: Tickers can have underscores (e.g., _DUMMY_37), so we must
                # check all possible tickers in our mapping to find the right split point
                mapped = False
                for ticker in ticker_to_location.keys():
                    if var.startswith(ticker + '_'):
                        # Found matching ticker, extract level
                        level_str = var[len(ticker) + 1:]  # +1 for underscore
                        if level_str.isdigit():
                            cluster_idx, asset_idx = ticker_to_location[ticker]
                            template_var = f"C{cluster_idx}_ASSET_{asset_idx}_{level_str}"
                            relabeling[var] = template_var
                            mapped = True

                            if var.startswith('CSCO'):
                                logger.debug(f"  DEBUG: CSCO mapped as asset! ticker={ticker}, template_var={template_var}")
                            break

                if not mapped:
                    # Keep as is if not found
                    if var.startswith('CSCO'):
                        logger.debug(f"  DEBUG: CSCO NOT MAPPED - keeping as {var}")
                    relabeling[var] = var

        # Create reverse mapping for decoding solution
        reverse_relabeling = {v: k for k, v in relabeling.items()}

        return relabeling, reverse_relabeling

    def _validate_and_fix_bqm_topology(self, bqm: dimod.BinaryQuadraticModel, cluster_info: list, embedding: dict) -> dimod.BinaryQuadraticModel:
        """
        Ensure BQM topology matches template embedding.

        The runtime BQM may have more edges than the template (due to different covariance
        structures). Remove edges that can't be embedded by the template.

        Args:
            bqm: The BQM to validate
            cluster_info: List of cluster metadata (with tickers)
            embedding: The template embedding dict

        Returns:
            BQM with topology matching template
        """
        embedded_vars = set(embedding.keys())

        # Build set of all possible edges between embedded variables
        # (Template can embed any edge between its variables)
        embeddable_edges = set()
        embedded_var_list = list(embedded_vars)
        for i, u in enumerate(embedded_var_list):
            for v in embedded_var_list[i+1:]:
                embeddable_edges.add(tuple(sorted([u, v])))

        # Remove edges not in template
        edges_to_remove = [edge for edge in bqm.quadratic if edge not in embeddable_edges]
        for edge in edges_to_remove:
            bqm.remove_interaction(edge[0], edge[1])

        # Remove variables not in embedding
        vars_to_remove = set(bqm.variables) - embedded_vars
        for var in vars_to_remove:
            bqm.remove_variable(var)

        # Add missing variables with zero bias
        missing_vars = embedded_vars - set(bqm.variables)
        for var in missing_vars:
            bqm.add_variable(var, 0.0)

        return bqm

    def _combine_cluster_bqms(self, clusters, mu, Sigma):
        """
        Combine multiple independent cluster BQMs into one large BQM.

        This allows solving all clusters in a single QPU call.

        Note: We build the combined BQM directly from h/Q dicts instead of using
        .update() to avoid numerical precision issues that can cause edges to be dropped.
        """
        import dimod

        # Build combined h and Q dictionaries directly
        h_combined = {}
        Q_combined = {}
        cluster_info = []

        # Handle both dict and list formats
        if isinstance(clusters, dict):
            cluster_items = clusters.items()
        else:
            cluster_items = enumerate(clusters)

        for cluster_id, cluster_tickers in cluster_items:
            cluster_tickers = [t for t in cluster_tickers if t in mu.index]
            if len(cluster_tickers) == 0:
                continue

            cluster_mu = pd.Series(mu[cluster_tickers])
            cluster_Sigma = pd.DataFrame(Sigma.loc[cluster_tickers, cluster_tickers])

            # Create BQM for this cluster
            # Post-processing normalizes weights, so budget constraint is unnecessary
            bqm = self.formulator.formulate_cluster(
                cluster_tickers, cluster_mu, cluster_Sigma
            )

            # Manually merge linear terms (h)
            for var, coeff in bqm.linear.items():
                h_combined[var] = h_combined.get(var, 0.0) + coeff

            # Manually merge quadratic terms (Q)
            for edge, coeff in bqm.quadratic.items():
                Q_combined[edge] = Q_combined.get(edge, 0.0) + coeff

            cluster_info.append({
                'id': cluster_id,
                'tickers': cluster_tickers,
                'variables': list(bqm.variables)
            })

        # Build combined BQM from merged dictionaries
        combined_bqm = dimod.BinaryQuadraticModel(h_combined, Q_combined, 0.0, dimod.BINARY)

        return combined_bqm, cluster_info

    def _compute_cluster_statistics(self, clusters, mu, Sigma):
        """
        Compute cluster-level expected returns and covariances.

        This treats each cluster as a "mega-asset" with properties derived
        from equal-weighted constituent assets.

        Args:
            clusters: Dict/list of asset clusters
            mu: Asset expected returns
            Sigma: Asset covariance matrix

        Returns:
            cluster_ids, cluster_mu, cluster_Sigma
        """
        # Handle both dict and list formats
        if isinstance(clusters, dict):
            cluster_items = list(clusters.items())
        else:
            cluster_items = list(enumerate(clusters))

        cluster_ids = []
        cluster_tickers_map = {}

        for cluster_id, cluster_tickers in cluster_items:
            cluster_tickers = [t for t in cluster_tickers if t in mu.index]
            if len(cluster_tickers) > 0:
                cluster_ids.append(cluster_id)
                cluster_tickers_map[cluster_id] = cluster_tickers

        # Compute cluster expected returns (equal-weighted within cluster)
        cluster_mu = pd.Series(index=cluster_ids, dtype=float)
        for cluster_id in cluster_ids:
            tickers = cluster_tickers_map[cluster_id]
            # Equal weight within cluster for meta-cluster calculation
            cluster_mu[cluster_id] = mu[tickers].mean()

        # Compute cluster covariance matrix
        cluster_Sigma = pd.DataFrame(0.0, index=cluster_ids, columns=cluster_ids)

        for i, cluster_i in enumerate(cluster_ids):
            tickers_i = cluster_tickers_map[cluster_i]
            n_i = len(tickers_i)

            for j, cluster_j in enumerate(cluster_ids):
                tickers_j = cluster_tickers_map[cluster_j]
                n_j = len(tickers_j)

                # Equal-weighted covariance between clusters
                # cov(C_i, C_j) = (1/n_i) * (1/n_j) * sum_{t_i in C_i} sum_{t_j in C_j} Sigma[t_i, t_j]
                cluster_sigma_block = Sigma.loc[tickers_i, tickers_j]
                cov_ij = cluster_sigma_block.values.mean()

                cluster_Sigma.loc[cluster_i, cluster_j] = cov_ij

        return cluster_ids, cluster_mu, cluster_Sigma, cluster_tickers_map

    def _apply_meta_cluster_weights(self, cluster_results: dict, meta_cluster_weights: pd.Series, real_tickers: set) -> pd.Series:
        """
        Apply meta-cluster weights to scale asset weights with magnitude preservation.

        Option B: Preserve the magnitude of meta-cluster weights while normalizing.

        Instead of flattening all clusters to equal proportions (sum=1.0), we:
        1. Compute normalized weights (sum=1.0)
        2. Scale by original magnitude (n_clusters)
        3. Top clusters get multiplier > 1.0, receiving more than proportional share

        This amplifies the solver's preference signal while maintaining valid weights.

        Args:
            cluster_results: Dict of cluster_id -> {'weights': pd.Series, 'tickers': list}
            meta_cluster_weights: Unnormalized meta-cluster weights from solver
            real_tickers: Set of real asset tickers (excludes dummy padding assets)

        Returns:
            Final portfolio weights (pd.Series, normalized to sum=1.0)
        """
        # Magnitude preservation: normalize, then rescale by n_clusters
        # This allows top cluster to receive > 1/n_clusters share
        n_clusters = len(meta_cluster_weights)
        original_sum = meta_cluster_weights.sum()

        if original_sum > 0:
            # Normalize to sum=1.0, then scale by n_clusters to preserve magnitude
            cluster_allocations = (meta_cluster_weights / original_sum) * n_clusters
        else:
            # Fallback to equal weight across clusters
            cluster_allocations = pd.Series(
                1.0,  # Each cluster gets weight=1.0 (will normalize at portfolio level)
                index=meta_cluster_weights.index
            )

        # Convert to dict for lookup
        cluster_allocations_dict = cluster_allocations.to_dict()

        # Apply meta-cluster weights to asset weights
        all_weights = {}
        for cluster_id, result in cluster_results.items():
            cluster_alloc = cluster_allocations_dict[cluster_id]
            cluster_weights_unnorm = result['weights']  # pd.Series
            cluster_weight_sum = cluster_weights_unnorm.sum()

            if cluster_weight_sum > 0:
                # Scale asset weights by cluster allocation from meta-cluster
                cluster_weights_norm = cluster_weights_unnorm / cluster_weight_sum
                for ticker in cluster_weights_norm.index:
                    all_weights[ticker] = cluster_weights_unnorm[ticker] * cluster_alloc
            else:
                # Equal weight within cluster, scaled by cluster allocation
                n_tickers = len(result['tickers'])
                for ticker in result['tickers']:
                    all_weights[ticker] = cluster_alloc / n_tickers

        # Filter out dummy assets (used for template padding)
        real_weights = {k: v for k, v in all_weights.items() if k in real_tickers}

        # Normalize to sum = 1.0 (should already be close, but ensure exact)
        total_weight = sum(real_weights.values())
        if total_weight > 0:
            final_weights = pd.Series(
                {k: v/total_weight for k, v in real_weights.items()}
            )
        else:
            # Fallback to equal weight (only over real assets)
            original_tickers = list(real_tickers)
            final_weights = pd.Series(1.0 / len(original_tickers), index=original_tickers)

        # Ensure all real tickers are present
        for ticker in real_tickers:
            if ticker not in final_weights:
                final_weights[ticker] = 0.0

        return final_weights

    def _apply_return_adjustment(self, weights: pd.Series, returns: pd.DataFrame,
                                 tickers: list) -> pd.Series:
        """
        Adjust weights by annualized returns over training period.

        Applied to ALL clusters (asset clusters and meta-cluster).

        New approach: w' = (% of cluster) * (YTD return as percentage)
        Example: If asset has 21.9% of cluster and +24.17% YTD return:
                 score = 21.9 * 24.17 = 529.323

        Args:
            weights: Raw weights from optimizer
            returns: Training period returns DataFrame
            tickers: List of tickers corresponding to weights

        Returns:
            Return-adjusted weights
        """
        # Calculate annualized returns over training period (YTD)
        annualized_returns = pd.Series(index=weights.index, dtype=float)
        for ticker in weights.index:
            if ticker in returns.columns:
                # Annualized return for single asset: daily_mean * TRADING_DAYS_PER_YEAR
                annualized_returns[ticker] = returns[ticker].mean() * TRADING_DAYS_PER_YEAR
            elif ticker in tickers:
                # For clusters: average returns across cluster members
                cluster_tickers = tickers
                valid_tickers = [t for t in cluster_tickers if t in returns.columns]
                if len(valid_tickers) > 0:
                    annualized_returns[ticker] = (
                        returns[valid_tickers].mean(axis=1).mean() * TRADING_DAYS_PER_YEAR
                    )
                else:
                    annualized_returns[ticker] = 0.0
            else:
                annualized_returns[ticker] = 0.0

        # Convert weights to percentages (0-100 scale)
        weight_pct = weights * 100

        # Convert returns to percentage points (0.2417 -> 24.17)
        return_pct = annualized_returns * 100

        # Score = weight_percentage * return_percentage
        # Example: 21.9% weight * 24.17% return = 529.323
        scores = weight_pct * return_pct

        # Handle negative returns gracefully (use absolute value to keep relative ranking)
        # This prevents negative scores from flipping the ordering
        scores_abs = scores.abs()

        # Normalize scores to weights
        if scores_abs.sum() > 0:
            final_weights = scores_abs / scores_abs.sum()
        else:
            # Fallback: use original weights
            final_weights = weights / weights.sum() if weights.sum() > 0 else weights

        return final_weights

    def _apply_thermometer_cutoff(self, weights: pd.Series) -> pd.Series:
        """
        Keep only assets with maximum (or near-maximum) thermometer level.

        The QUBO solver sets more bits for "better" assets. This filtering
        keeps only the highest-confidence picks based on thermometer encoding.

        Args:
            weights: Portfolio weights from optimizer

        Returns:
            Filtered weights (only max-level assets, renormalized)
        """
        if not self.use_thermometer_cutoff:
            return weights

        # Get max weight (represents maximum thermometer level)
        max_weight = weights.max()

        # Keep only assets at or near max level (within 10% tolerance for numerical precision)
        threshold = max_weight * 0.9
        high_confidence = weights[weights >= threshold]

        # Renormalize
        if high_confidence.sum() > 0:
            filtered_weights = high_confidence / high_confidence.sum()
        else:
            # Fallback: keep original weights
            filtered_weights = weights

        # Expand to include zeros for all original assets
        final_weights = pd.Series(0.0, index=weights.index)
        final_weights[filtered_weights.index] = filtered_weights

        return final_weights

    def _compute_dynamic_meta_beta(self, returns: pd.DataFrame) -> float:
        """
        Compute dynamic risk aversion for meta-cluster based on market conditions.

        Strategy: Inversely proportional to market returns
        - Bull market (high returns) → lower beta (less risk-averse, take more risk)
        - Bear market (low/negative returns) → higher beta (more risk-averse, preserve capital)

        Args:
            returns: Training period returns

        Returns:
            Adjusted beta for meta-cluster
        """
        # Compute market return over training period (annualized)
        market_return = returns.mean().mean() * TRADING_DAYS_PER_YEAR

        # Base beta
        base_beta = self.beta

        # Dynamic adjustment: beta_meta = base_beta * (1 + k / (1 + market_return))
        # When market_return is high (e.g., 0.5 = 50%), beta decreases (more aggressive)
        # When market_return is low/negative, beta increases (more defensive)

        # Scaling factor (controls sensitivity)
        k = 2.0  # Typical: 2x adjustment range

        # Compute adjusted beta
        if market_return > -0.5:  # Avoid division issues in extreme bear markets
            beta_meta = base_beta * (1 + k / (1 + market_return))
        else:
            beta_meta = base_beta * 5.0  # Extreme defensive in severe bear market

        # Clamp to reasonable range
        beta_meta = max(base_beta * 0.5, min(beta_meta, base_beta * 5.0))

        return beta_meta

    def _create_combined_bqm(
        self,
        clusters: Dict[str, List[str]],
        mu: pd.Series,
        Sigma: pd.DataFrame,
        cluster_ids: List[str],
        cluster_mu: np.ndarray,
        cluster_Sigma: np.ndarray,
        returns: pd.DataFrame
    ) -> Tuple[Any, List[Dict]]:
        """
        Create combined BQM for all asset clusters + meta-cluster.

        Args:
            clusters: Dictionary of cluster_id -> ticker list
            mu: Mean returns (padded)
            Sigma: Covariance matrix (padded)
            cluster_ids: List of cluster IDs
            cluster_mu: Cluster-level mean returns
            cluster_Sigma: Cluster-level covariance matrix
            returns: Returns DataFrame for dynamic beta calculation

        Returns:
            Tuple of (combined_bqm, cluster_info)
        """
        # 1. Create BQMs for all asset clusters
        combined_bqm, cluster_info = self._combine_cluster_bqms(clusters, mu, Sigma)

        # 2. Create meta-cluster BQM with dynamic risk aversion
        meta_cluster_tickers = list(cluster_ids)
        meta_beta = self._compute_dynamic_meta_beta(returns)

        # Create temporary formulator with adjusted beta for meta-cluster
        from qpo.qubo.discrete_levels import DiscreteLevelFormulator
        meta_formulator = DiscreteLevelFormulator(
            n_levels=self.n_levels,
            alpha=self.alpha,
            beta=meta_beta,
            thermometer_penalty=self.thermometer_penalty,
            l1_sparsity_penalty=self.l1_sparsity_penalty
        )

        meta_bqm = meta_formulator.formulate_cluster(
            meta_cluster_tickers, cluster_mu, cluster_Sigma
        )

        # 3. Combine asset clusters + meta-cluster BQMs
        # Manually merge to avoid numerical precision issues with .update()
        for var, coeff in meta_bqm.linear.items():
            combined_bqm.add_variable(var, coeff)
        for edge, coeff in meta_bqm.quadratic.items():
            combined_bqm.add_interaction(edge[0], edge[1], coeff)

        # Add meta-cluster info
        cluster_info.append({
            'id': 'META_CLUSTER',
            'tickers': meta_cluster_tickers,
            'variables': list(meta_bqm.variables)
        })

        return combined_bqm, cluster_info

    def _solve_with_fixed_embedding(
        self,
        combined_bqm: Any,
        cluster_info: List[Dict]
    ) -> Tuple[Dict, float, float, float]:
        """
        Solve combined BQM using FixedEmbeddingComposite (works for both QPU and SA).

        Args:
            combined_bqm: Combined BQM for all clusters
            cluster_info: List of cluster metadata dicts

        Returns:
            Tuple of (best_sample, runtime, qpu_access_time, network_time)
        """
        # Get base sampler and handle embedding based on solver type
        if self.solver_type == 'qpu':
            # Initialize QPU sampler if needed
            if self._qpu_sampler is None:
                from dwave.system import DWaveSampler
                import os
                solver_name = os.environ.get('DWAVE_API_SOLVER', 'Advantage2_system1.6')
                self._qpu_sampler = DWaveSampler(solver=solver_name)
            base_sampler = self._qpu_sampler

            # Use FixedEmbeddingComposite with templates for QPU
            if self.expected_assets_per_cluster is not None:
                from dwave.system import FixedEmbeddingComposite
                # Load template embedding with expected dimensions
                combined_embedding = self.embedding_mgr.get_full_portfolio_embedding(
                    cluster_info,
                    self.n_levels,
                    self.expected_assets_per_cluster,
                    self.expected_n_clusters
                )

                # Relabel BQM variables from ticker names to template format
                # (e.g., CSCO_1 -> C0_ASSET_5_1, cluster_1_6 -> META_1_6)
                relabeling, reverse_relabeling = self._relabel_bqm_for_template(
                    combined_bqm, cluster_info
                )
                combined_bqm = combined_bqm.relabel_variables(relabeling, inplace=False)

                # Store reverse relabeling for decoding solution later
                self._reverse_relabeling = reverse_relabeling

                # Pad BQM with missing template variables (set to 0)
                # The template expects all variables even if not used by actual portfolio
                template_vars = set(combined_embedding.keys())
                bqm_vars = set(combined_bqm.variables)
                missing_vars = template_vars - bqm_vars

                if missing_vars:
                    # Add missing variables with zero bias (no contribution to objective)
                    for var in missing_vars:
                        combined_bqm.add_variable(var, 0.0)

                # Remove any extra variables not in template (shouldn't happen but be safe)
                extra_vars = bqm_vars - template_vars
                if extra_vars:
                    for var in extra_vars:
                        combined_bqm.remove_variable(var)

                # Final validation
                if set(combined_bqm.variables) != template_vars:
                    raise ValueError(
                        f"BQM variable mismatch after padding!\n"
                        f"Expected: {len(template_vars)} variables\n"
                        f"Got: {len(combined_bqm.variables)} variables"
                    )

                sampler = FixedEmbeddingComposite(base_sampler, combined_embedding)
            else:
                # Without templates, use auto-embedding for QPU
                from dwave.system import EmbeddingComposite
                sampler = EmbeddingComposite(base_sampler)

        else:  # simulated annealing
            from dwave.samplers import SimulatedAnnealingSampler
            base_sampler = SimulatedAnnealingSampler()

            # For SA with templates, relabel and validate BQM structure
            # SA is unstructured and doesn't need qubit mapping, but we must ensure
            # the problem structure is identical to what QPU would solve
            if self.expected_assets_per_cluster is not None:
                # Load template to validate BQM structure matches expected dimensions
                combined_embedding = self.embedding_mgr.get_full_portfolio_embedding(
                    cluster_info,
                    self.n_levels,
                    self.expected_assets_per_cluster,
                    self.expected_n_clusters
                )

                # Relabel BQM variables from ticker names to template format
                # (e.g., ZTS_5 -> C0_ASSET_5_5, cluster_1_6 -> META_1_6)
                relabeling, reverse_relabeling = self._relabel_bqm_for_template(
                    combined_bqm, cluster_info
                )
                combined_bqm = combined_bqm.relabel_variables(relabeling, inplace=False)

                # Store reverse relabeling for decoding solution later
                self._reverse_relabeling = reverse_relabeling

                # Pad BQM with missing template variables (set to 0)
                # The template expects all variables even if not used by actual portfolio
                template_vars = set(combined_embedding.keys())
                bqm_vars = set(combined_bqm.variables)
                missing_vars = template_vars - bqm_vars

                if missing_vars:
                    # Add missing variables with zero bias (no contribution to objective)
                    for var in missing_vars:
                        combined_bqm.add_variable(var, 0.0)

                # Remove any extra variables not in template (shouldn't happen but be safe)
                extra_vars = bqm_vars - template_vars
                if extra_vars:
                    for var in extra_vars:
                        combined_bqm.remove_variable(var)

                # Final validation
                if set(combined_bqm.variables) != template_vars:
                    raise ValueError(
                        f"BQM variable mismatch after padding!\n"
                        f"Expected: {len(template_vars)} variables\n"
                        f"Got: {len(combined_bqm.variables)} variables"
                    )

            # Use base sampler directly (SA doesn't need embedding)
            sampler = base_sampler

        # Solve
        start_solve = time.time()
        if self.solver_type == 'qpu':
            response = sampler.sample(
                combined_bqm,
                num_reads=10,
                annealing_time=self.annealing_time
            )
        else:
            response = sampler.sample(
                combined_bqm,
                num_reads=self.num_reads,
                num_sweeps=self.num_sweeps
            )
        solve_time = time.time() - start_solve

        # Extract timing info (QPU only)
        qpu_access_time = 0.0
        network_time = 0.0
        if self.solver_type == 'qpu' and hasattr(response, 'info'):
            qpu_access_time = response.info.get('timing', {}).get('qpu_access_time', 0.0) / 1_000_000.0  # Convert µs to seconds
            network_time = solve_time - qpu_access_time  # Approximate network time

        return response.first.sample, solve_time, qpu_access_time, network_time

    def _decode_cluster_results(
        self,
        best_sample: Dict,
        cluster_info: List[Dict],
        returns: pd.DataFrame
    ) -> Tuple[Dict, pd.Series]:
        """
        Decode cluster results from BQM solution.

        Args:
            best_sample: Best sample from solver
            cluster_info: List of cluster metadata
            returns: Returns DataFrame for return adjustment

        Returns:
            Tuple of (cluster_results, meta_cluster_weights)
        """
        cluster_results = {}
        meta_cluster_weights = None

        for cluster_data in cluster_info:
            cluster_id = cluster_data['id']
            cluster_tickers = cluster_data['tickers']

            # Decode this cluster's portion of the solution
            cluster_weights = self.formulator.decode_solution(
                best_sample, cluster_tickers
            )

            # Apply return adjustment to ALL clusters
            return_adjusted_weights = self._apply_return_adjustment(
                cluster_weights, returns, cluster_tickers
            )

            if cluster_id == 'META_CLUSTER':
                # Meta-cluster: save weights for cluster allocation
                meta_cluster_weights = return_adjusted_weights
            else:
                # Asset clusters: apply thermometer cutoff per-cluster (if enabled)
                if self.use_thermometer_cutoff:
                    filtered_weights = self._apply_thermometer_cutoff(return_adjusted_weights)
                else:
                    filtered_weights = return_adjusted_weights

                cluster_results[cluster_id] = {
                    'weights': filtered_weights,
                    'tickers': cluster_tickers
                }

        return cluster_results, meta_cluster_weights

    def _prepare_clusters_and_statistics(
        self,
        returns: pd.DataFrame,
        mu_original: pd.Series,
        Sigma_original: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], pd.Series, pd.DataFrame, set, List[str], np.ndarray, np.ndarray, Dict]:
        """
        Prepare clusters with padding and compute cluster-level statistics.

        Args:
            returns: Returns DataFrame
            mu_original: Original mean returns (annualized)
            Sigma_original: Original covariance matrix (annualized)

        Returns:
            Tuple of:
            - clusters: Padded clusters dict
            - mu: Padded mean returns
            - Sigma: Padded covariance matrix
            - real_tickers: Set of actual (non-dummy) tickers
            - cluster_ids: List of cluster IDs
            - cluster_mu: Cluster-level mean returns
            - cluster_Sigma: Cluster-level covariance matrix
            - cluster_tickers_map: Mapping of cluster_id -> tickers
        """
        tickers = list(mu_original.index)
        clusters = self.clusterer.cluster(returns)

        # Auto-select template if enabled
        self._auto_select_template(clusters)

        # Prepare for padding
        real_tickers = set(tickers)
        mu = mu_original
        Sigma = Sigma_original

        # Pad clusters to match fixed template if specified
        if self.expected_assets_per_cluster is not None:
            clusters, mu, Sigma, real_tickers = self._pad_clusters_to_size(
                clusters, self.expected_assets_per_cluster, mu_original, Sigma_original,
                target_n_clusters=self.expected_n_clusters
            )

        # Compute cluster statistics for meta-cluster
        cluster_ids, cluster_mu, cluster_Sigma, cluster_tickers_map = \
            self._compute_cluster_statistics(clusters, mu, Sigma)

        return clusters, mu, Sigma, real_tickers, cluster_ids, cluster_mu, cluster_Sigma, cluster_tickers_map

    def _auto_select_template(self, clusters: Dict[str, List[str]]) -> None:
        """
        Auto-select the best-fitting template based on total portfolio size.

        Strategy:
        1. Calculate total assets needed (sum of all cluster sizes)
        2. Find templates where num_clusters × assets_per_cluster ≥ total_assets
        3. Prefer templates with the largest 'l' (levels) that fits
        4. Among same 'l', pick template with least waste

        Updates self.expected_n_clusters, self.expected_assets_per_cluster, and self.n_levels.
        Raises FileNotFoundError if no suitable template exists.

        Args:
            clusters: Dictionary mapping cluster_id -> list of ticker symbols
        """
        if not self.auto_select_template or self.expected_assets_per_cluster is not None:
            return  # Already manually specified or auto-select disabled

        # Calculate total assets needed
        total_assets = sum(len(tickers) for tickers in clusters.values())
        n_clusters = len(clusters)

        # Find all available templates (any level)
        import glob
        from pathlib import Path
        template_dir = Path(__file__).parent.parent.parent / 'embeddings' / 'templates'
        all_templates = glob.glob(str(template_dir / 'portfolio_*c_*a_*l.json'))

        if not all_templates:
            raise FileNotFoundError(
                f"No templates found in {template_dir}.\n"
                f"Portfolio requires: {total_assets} total assets across {n_clusters} clusters\n"
                f"Solutions:\n"
                f"  1. Generate templates\n"
                f"  2. Set auto_select_template=False to disable template usage"
            )

        # Parse all templates and group by level
        templates_by_level = {}  # level -> list of (clusters, assets, capacity, waste, fname)

        for template_path in all_templates:
            fname = Path(template_path).stem  # e.g., "portfolio_12c_16a_6l"
            parts = fname.split('_')
            if len(parts) >= 3:
                template_clusters = int(parts[1].rstrip('c'))
                template_assets = int(parts[2].rstrip('a'))
                template_levels = int(parts[3].rstrip('l'))

                # Calculate capacity and waste
                capacity = template_clusters * template_assets

                # Check if this template can fit our portfolio
                if capacity >= total_assets:
                    waste = capacity - total_assets

                    if template_levels not in templates_by_level:
                        templates_by_level[template_levels] = []

                    templates_by_level[template_levels].append(
                        (template_clusters, template_assets, capacity, waste, fname)
                    )

        if not templates_by_level:
            # No suitable templates found
            available_str = "\n  ".join(
                f"{Path(p).stem}" for p in sorted(all_templates)[:10]
            )
            raise FileNotFoundError(
                f"No suitable template found for portfolio size.\n"
                f"Portfolio requires: {total_assets} total assets across {n_clusters} clusters\n"
                f"Available templates (showing first 10):\n  {available_str}\n"
                f"Solutions:\n"
                f"  1. Reduce max_cluster_size to create smaller clusters\n"
                f"  2. Generate a larger template\n"
                f"  3. Set auto_select_template=False to disable template usage"
            )

        # Select template with largest 'l' (prefer more levels), then least waste
        best_template = None
        best_level = 0
        best_waste = float('inf')

        # Iterate through levels in descending order (largest 'l' first)
        for level in sorted(templates_by_level.keys(), reverse=True):
            candidates = templates_by_level[level]
            # Sort by waste (ascending) to find best candidate for this level
            candidates.sort(key=lambda x: x[3])

            # Pick the one with least waste for this level
            template_clusters, template_assets, capacity, waste, fname = candidates[0]

            # Always prefer larger 'l', regardless of waste
            # (Larger l = more granular weight selection)
            if level > best_level:
                best_template = (template_clusters, template_assets, level, fname)
                best_level = level
                best_waste = waste
            elif level == best_level and waste < best_waste:
                # Same level, pick the one with less waste
                best_template = (template_clusters, template_assets, level, fname)
                best_waste = waste

        if best_template:
            self.expected_n_clusters, self.expected_assets_per_cluster, actual_levels, fname = best_template

            # Update n_levels if we had to use a different one
            if actual_levels != self.n_levels:
                print(f"NOTE: Requested {self.n_levels} levels, but using {actual_levels} levels "
                      f"(largest available for portfolio size).")
                self.n_levels = actual_levels

                # Update formulator with new n_levels
                from qpo.qubo.discrete_levels import DiscreteLevelFormulator
                self.formulator = DiscreteLevelFormulator(
                    n_levels=actual_levels,
                    alpha=self.alpha,
                    beta=self.beta,
                    thermometer_penalty=self.thermometer_penalty,
                    l1_sparsity_penalty=self.l1_sparsity_penalty
                )

            capacity = self.expected_n_clusters * self.expected_assets_per_cluster
            print(f"Auto-selected template: {self.expected_n_clusters}c × "
                  f"{self.expected_assets_per_cluster}a × {actual_levels}l "
                  f"(capacity: {capacity}, needed: {total_assets})")

    def optimize(self, returns: pd.DataFrame) -> dict:
        """
        Optimize portfolio weights using quantum annealing with clustering.

        This method orchestrates the full optimization pipeline:
        1. Prepare clusters with padding and statistics
        2. Create combined BQM for all clusters + meta-cluster
        3. Solve using FixedEmbeddingComposite (QPU or SA)
        4. Decode and aggregate results

        Args:
            returns: Returns DataFrame (dates × tickers)

        Returns:
            Dictionary with 'weights' and 'metrics'
        """
        # Compute mu and Sigma from returns (preprocessing, not timed)
        mu_original = returns.mean() * TRADING_DAYS_PER_YEAR
        Sigma_original = returns.cov() * TRADING_DAYS_PER_YEAR

        # Prepare clusters with padding and statistics
        (clusters, mu, Sigma, real_tickers, cluster_ids,
         cluster_mu, cluster_Sigma, cluster_tickers_map) = \
            self._prepare_clusters_and_statistics(returns, mu_original, Sigma_original)

        # Start timing (only measure actual solving)
        start_time = time.time()

        # Create combined BQM for all asset clusters + meta-cluster
        combined_bqm, cluster_info = self._create_combined_bqm(
            clusters, mu, Sigma, cluster_ids, cluster_mu, cluster_Sigma, returns
        )

        # Solve using FixedEmbeddingComposite (works for both QPU and SA)
        best_sample, solve_time, qpu_access_time, network_time = \
            self._solve_with_fixed_embedding(combined_bqm, cluster_info)

        # Decode cluster results
        cluster_results, meta_cluster_weights = self._decode_cluster_results(
            best_sample, cluster_info, returns
        )

        # Apply meta-cluster weights to get final portfolio weights
        final_weights = self._apply_meta_cluster_weights(
            cluster_results, meta_cluster_weights, real_tickers
        )

        # Compute metrics using original mu/Sigma (before padding)
        aligned_weights = final_weights.reindex(mu_original.index, fill_value=0.0)

        portfolio_return = np.dot(aligned_weights.values, mu_original.values)
        portfolio_risk = np.sqrt(
            np.dot(aligned_weights.values, np.dot(Sigma_original.values, aligned_weights.values))
        )
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

        runtime = time.time() - start_time

        # Build metrics dict
        metrics = {
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe': sharpe,
            'n_clusters': len(cluster_results),
            'runtime': runtime,
            'solver_only_runtime': solve_time
        }

        # Add QPU-specific timing if available
        if self.solver_type == 'qpu' and qpu_access_time > 0:
            metrics['qpu_access_time'] = qpu_access_time  # In seconds
            metrics['network_latency'] = network_time  # In seconds
            metrics['solver_only_runtime'] = qpu_access_time  # In seconds

        return {
            'weights': final_weights,
            'metrics': metrics
        }


class DiscreteLevelsOptimizerWrapper:
    """Wrapper to make DiscreteLevelsOptimizer compatible with Backtester."""

    def __init__(self, optimizer: DiscreteLevelsOptimizer):
        self.optimizer = optimizer

    def optimize(self, returns: pd.DataFrame) -> dict:
        """
        Optimize and return result dict.

        Args:
            returns: Returns DataFrame

        Returns:
            Dictionary with 'weights' and 'metrics'
        """
        return self.optimizer.optimize(returns)

