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
import numpy as np
import pandas as pd
from typing import Optional, Any
import dimod

from qpo.qubo.discrete_levels import DiscreteLevelFormulator, CachedEmbeddingManager
from qpo.qubo.aggregation import ClusterAggregator
from clustering import SectorClusterer


class DiscreteLevelsOptimizer:
    """
    Portfolio optimizer using discrete weight levels with sector clustering.

    Compatible with the Backtester framework for comparison with classical
    and other quantum optimizers.
    """

    def __init__(self,
                 n_levels: int = 6,  # Match available templates (most are 6 levels)
                 alpha: float = 10,
                 beta: float = 2,
                 thermometer_penalty: float = 10.0,
                 solver_type: str = 'simulated',
                 num_reads: Optional[int] = None,
                 num_sweeps: Optional[int] = None,
                 max_cluster_size: int = 19,
                 portfolio_info_csv: str = "portfolio-info.csv",
                 clusterer: Optional[Any] = None,
                 expected_assets_per_cluster: Optional[int] = None,
                 expected_n_clusters: Optional[int] = None,
                 auto_select_template: bool = True,
                 k_spread: float = 0.0,
                 l1_sparsity_penalty: float = 0.0,
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
            max_cluster_size: Maximum assets per cluster (19 for native embedding)
            portfolio_info_csv: Path to portfolio info for sector data
            clusterer: Custom clusterer (default: SectorClusterer)
            expected_assets_per_cluster: For QPU with templates, pad clusters to this size
            expected_n_clusters: For QPU with templates, pad to this many clusters
            auto_select_template: Automatically select best template for portfolio size
            k_spread: Concentration parameter (0.0=no spread, 1.0-3.0=moderate spread)
            l1_sparsity_penalty: L1 regularization penalty (λ > 0 promotes sparsity, try 0.01-1.0)
                                Adds +λ·||w||₁ to objective → fewer holdings
            use_thermometer_cutoff: If True, keep only assets with maximum thermometer level
        """
        self.n_levels = n_levels
        self.alpha = alpha
        self.beta = beta
        self.thermometer_penalty = thermometer_penalty
        self.solver_type = solver_type
        self.k_spread = k_spread
        self.l1_sparsity_penalty = l1_sparsity_penalty
        self.use_thermometer_cutoff = use_thermometer_cutoff

        # Set num_reads based on solver type if not specified
        if num_reads is None:
            self.num_reads = 10 if solver_type == 'qpu' else 256
        else:
            self.num_reads = num_reads

        # Set num_sweeps based on solver type if not specified
        if num_sweeps is None:
            self.num_sweeps = 256 if solver_type == 'simulated' else 0
        else:
            self.num_sweeps = num_sweeps
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

        # Extend mu with zeros for dummy assets
        all_tickers = []
        for tickers in padded_clusters.values():
            all_tickers.extend(tickers)

        dummy_tickers = [t for t in all_tickers if t.startswith('_DUMMY_')]

        padded_mu = mu.copy()
        for dummy in dummy_tickers:
            padded_mu[dummy] = 0.0  # Zero expected return

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
                # Annualized return for single asset: daily_mean * 252
                annualized_returns[ticker] = returns[ticker].mean() * 252
            elif ticker in tickers:
                # For clusters: average returns across cluster members
                cluster_tickers = tickers
                valid_tickers = [t for t in cluster_tickers if t in returns.columns]
                if len(valid_tickers) > 0:
                    annualized_returns[ticker] = (
                        returns[valid_tickers].mean(axis=1).mean() * 252
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

    def _apply_k_spread(self, weights: pd.Series) -> pd.Series:
        """
        Apply k-based spread to weights using power function.

        Applied ONLY to meta-cluster weights.

        Args:
            weights: Return-adjusted weights

        Returns:
            Spread-adjusted weights
        """
        if self.k_spread <= 0:
            return weights

        # Apply k-based spread: w' = w^(1+k)
        spread_power = 1 + self.k_spread
        spread_weights = weights ** spread_power

        # Normalize
        if spread_weights.sum() > 0:
            final_weights = spread_weights / spread_weights.sum()
        else:
            final_weights = weights

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
        market_return = returns.mean().mean() * 252

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

    def optimize(self, returns: pd.DataFrame) -> dict:
        """
        Optimize portfolio weights.

        Args:
            returns: Returns DataFrame (dates × tickers)

        Returns:
            Dictionary with 'weights' and 'metrics'
        """
        # Compute mu and Sigma from returns (preprocessing, not timed)
        mu_original = returns.mean() * 252  # Annualized returns
        Sigma_original = returns.cov() * 252  # Annualized covariance

        # Cluster assets (preprocessing, not timed)
        tickers = list(mu_original.index)
        clusters = self.clusterer.cluster(returns)

        # Auto-select template if enabled and not manually specified
        if self.auto_select_template and self.expected_assets_per_cluster is None:
            # Find best matching template
            n_actual_clusters = len(clusters)
            max_actual_assets = max(len(tickers) for tickers in clusters.values())

            # Select first available template that fits (can expand with smarter selection)
            # For now, use a simple heuristic: pick smallest template that can fit
            import glob
            from pathlib import Path
            template_dir = Path(__file__).parent.parent.parent / 'embeddings' / 'templates'
            available_templates = glob.glob(str(template_dir / f'portfolio_*c_*a_{self.n_levels}l.json'))

            if available_templates:
                # Parse template dimensions
                best_template = None
                best_waste = float('inf')

                for template_path in available_templates:
                    fname = Path(template_path).stem  # e.g., "portfolio_12c_16a_6l"
                    parts = fname.split('_')
                    if len(parts) >= 3:
                        template_clusters = int(parts[1].rstrip('c'))
                        template_assets = int(parts[2].rstrip('a'))

                        # Check if this template can fit our portfolio
                        if template_clusters >= n_actual_clusters and template_assets >= max_actual_assets:
                            # Calculate "waste" (excess capacity)
                            waste = (template_clusters - n_actual_clusters) * template_assets + \
                                    (template_assets - max_actual_assets) * n_actual_clusters

                            if waste < best_waste:
                                best_waste = waste
                                best_template = (template_clusters, template_assets)

                if best_template:
                    self.expected_n_clusters, self.expected_assets_per_cluster = best_template
                    print(f"Auto-selected template: {self.expected_n_clusters}c × {self.expected_assets_per_cluster}a × {self.n_levels}l")

        # Pad clusters to match fixed template (required for both QPU and SA)
        # Templates are mandatory - we always need to match a pre-computed embedding
        real_tickers = set(tickers)
        mu = mu_original
        Sigma = Sigma_original

        if self.expected_assets_per_cluster is not None:
            # Pad to specified template shape
            clusters, mu, Sigma, real_tickers = self._pad_clusters_to_size(
                clusters, self.expected_assets_per_cluster, mu_original, Sigma_original,
                target_n_clusters=self.expected_n_clusters
            )

        # Compute cluster statistics for meta-cluster (preprocessing, not timed)
        cluster_ids, cluster_mu, cluster_Sigma, cluster_tickers_map = \
            self._compute_cluster_statistics(clusters, mu, Sigma)

        # Initialize QPU sampler if needed (setup, not timed)
        if self.solver_type == 'qpu' and self._qpu_sampler is None:
            from dwave.system import DWaveSampler
            import os
            solver_name = os.environ.get('DWAVE_API_SOLVER', 'Advantage2_system1.6')
            self._qpu_sampler = DWaveSampler(solver=solver_name)

        # Start timing (only measure actual solving)
        start_time = time.time()

        # Track QPU-specific timing
        total_qpu_access_time = 0.0  # microseconds
        total_network_time = 0.0  # seconds

        # Solve clusters + meta-cluster
        cluster_results = {}
        meta_cluster_weights = None

        if self.solver_type == 'qpu':
            # QPU: Combine all asset clusters + meta-cluster into one BQM
            from dwave.system import FixedEmbeddingComposite

            # 1. Create BQMs for all asset clusters
            combined_bqm, cluster_info = self._combine_cluster_bqms(clusters, mu, Sigma)

            # 2. Create meta-cluster BQM with dynamic risk aversion
            meta_cluster_tickers = list(cluster_ids)

            # Compute dynamic beta for meta-cluster (inversely proportional to market conditions)
            meta_beta = self._compute_dynamic_meta_beta(returns)

            # Create temporary formulator with adjusted beta for meta-cluster
            from qpo.qubo.discrete_levels import DiscreteLevelFormulator
            meta_formulator = DiscreteLevelFormulator(
                n_levels=self.n_levels,
                alpha=self.alpha,
                beta=meta_beta,  # Dynamic beta based on market conditions
                thermometer_penalty=self.thermometer_penalty,
                l1_sparsity_penalty=self.l1_sparsity_penalty
            )

            meta_bqm = meta_formulator.formulate_cluster(
                meta_cluster_tickers, cluster_mu, cluster_Sigma
            )

            # 3. Combine asset clusters + meta-cluster BQMs
            # (They are independent, no couplings between them)
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

            # Load template embedding
            combined_embedding = self.embedding_mgr.get_full_portfolio_embedding(cluster_info, self.n_levels)

            # Validate and fix BQM topology to match template
            combined_bqm = self._validate_and_fix_bqm_topology(combined_bqm, cluster_info, combined_embedding)

            # Create sampler with fixed embedding
            sampler = FixedEmbeddingComposite(self._qpu_sampler, combined_embedding)

            # Sample on QPU (single call for all asset clusters + meta-cluster)
            qpu_start = time.time()
            response = sampler.sample(
                combined_bqm,
                num_reads=self.num_reads,
                annealing_time=5
            )
            qpu_elapsed = time.time() - qpu_start

            # Extract QPU timing
            timing = response.info.get('timing', {})
            qpu_total_time = timing.get('qpu_access_time', 0)
            qpu_sampling_time = timing.get('qpu_sampling_time', 0)

            total_qpu_access_time = qpu_sampling_time
            total_network_time = max(0, qpu_elapsed - (qpu_total_time / 1e6))

            # Decode solution for asset clusters and meta-cluster
            best_sample = response.first.sample

            for cluster_data in cluster_info:
                cluster_id = cluster_data['id']
                cluster_tickers = cluster_data['tickers']

                # Decode this cluster's portion of the solution
                cluster_weights = self.formulator.decode_solution(
                    best_sample, cluster_tickers
                )

                # Step 1: Apply return adjustment to ALL clusters
                return_adjusted_weights = self._apply_return_adjustment(
                    cluster_weights, returns, cluster_tickers
                )

                if cluster_id == 'META_CLUSTER':
                    # Step 2: Apply k-spread ONLY to meta-cluster
                    meta_cluster_weights = self._apply_k_spread(return_adjusted_weights)
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

            # Apply meta-cluster weights to get final portfolio weights
            final_weights = self._apply_meta_cluster_weights(
                cluster_results, meta_cluster_weights, real_tickers
            )

        elif self.solver_type == 'simulated':
            # For SA: Combine all asset clusters + meta-cluster into one BQM and solve together

            # 1. Create BQMs for all asset clusters
            combined_bqm, cluster_info = self._combine_cluster_bqms(clusters, mu, Sigma)

            # 2. Create meta-cluster BQM with dynamic risk aversion
            meta_cluster_tickers = list(cluster_ids)

            # Compute dynamic beta for meta-cluster (inversely proportional to market conditions)
            meta_beta = self._compute_dynamic_meta_beta(returns)

            # Create temporary formulator with adjusted beta for meta-cluster
            from qpo.qubo.discrete_levels import DiscreteLevelFormulator
            meta_formulator = DiscreteLevelFormulator(
                n_levels=self.n_levels,
                alpha=self.alpha,
                beta=meta_beta,  # Dynamic beta based on market conditions
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

            # Solve combined BQM with SA (using neal for consistency)
            from neal import SimulatedAnnealingSampler
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(
                combined_bqm,
                num_reads=self.num_reads,
                num_sweeps=self.num_sweeps
            )

            # Decode solution for asset clusters and meta-cluster
            best_sample = response.first.sample

            for cluster_data in cluster_info:
                cluster_id = cluster_data['id']
                cluster_tickers = cluster_data['tickers']

                # Decode this cluster's portion of the solution
                cluster_weights = self.formulator.decode_solution(
                    best_sample, cluster_tickers
                )

                # Step 1: Apply return adjustment to ALL clusters
                return_adjusted_weights = self._apply_return_adjustment(
                    cluster_weights, returns, cluster_tickers
                )

                if cluster_id == 'META_CLUSTER':
                    # Step 2: Apply k-spread ONLY to meta-cluster
                    meta_cluster_weights = self._apply_k_spread(return_adjusted_weights)
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

            # Apply meta-cluster weights to get final portfolio weights
            final_weights = self._apply_meta_cluster_weights(
                cluster_results, meta_cluster_weights, real_tickers
            )

        # Thermometer cutoff is now applied per-cluster (above), not on final weights

        # Compute metrics using original mu/Sigma (before padding)
        # Align weights with original data
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
            'solver_only_runtime': runtime  # For comparison
        }

        # Add QPU-specific timing if available
        if self.solver_type == 'qpu' and total_qpu_access_time > 0:
            metrics['qpu_access_time'] = total_qpu_access_time / 1e6  # Convert to seconds
            metrics['network_latency'] = total_network_time
            metrics['solver_only_runtime'] = total_qpu_access_time / 1e6  # Actual QPU time

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
