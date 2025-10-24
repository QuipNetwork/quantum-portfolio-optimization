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
                 alpha: float = 5.0,
                 beta: float = 2.5,
                 budget_penalty: float = 200.0,  # Strong penalty for budget constraint
                 thermometer_penalty: float = 10.0,  # Relaxed penalty for thermometer encoding
                 solver_type: str = 'simulated',
                 num_reads: int = 20,
                 num_sweeps: int = 256,
                 max_cluster_size: int = 19,
                 portfolio_info_csv: str = "portfolio-info.csv",
                 clusterer: Optional[Any] = None,
                 expected_assets_per_cluster: Optional[int] = None,
                 expected_n_clusters: Optional[int] = None,
                 auto_select_template: bool = True):
        """
        Initialize discrete levels optimizer.

        Args:
            n_levels: Number of discrete weight levels (default 4)
            alpha: Return coefficient (default 5.0, aggressive)
            beta: Risk coefficient (default 2.5, ratio=0.5)
            thermometer_penalty: Penalty for invalid thermometer encoding
            solver_type: 'simulated' or 'qpu'
            num_reads: Number of annealing samples
            num_sweeps: Sweeps for simulated annealing (default 256, normalized)
            max_cluster_size: Maximum assets per cluster (19 for native embedding)
            portfolio_info_csv: Path to portfolio info for sector data
            clusterer: Custom clusterer (default: SectorClusterer)
            expected_assets_per_cluster: For QPU with templates, pad clusters to this size
            expected_n_clusters: For QPU with templates, pad to this many clusters
        """
        self.n_levels = n_levels
        self.alpha = alpha
        self.beta = beta
        self.budget_penalty = budget_penalty
        self.thermometer_penalty = thermometer_penalty
        self.solver_type = solver_type
        self.num_reads = num_reads
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
            budget_penalty=budget_penalty,
            thermometer_penalty=thermometer_penalty
        )

        # Create embedding manager
        self.embedding_mgr = CachedEmbeddingManager()

        # Create clusterer
        if clusterer is None:
            self.clusterer = SectorClusterer(
                max_cluster_size=max_cluster_size,
                target_cluster_size=max_cluster_size // 2,
                sector_map=portfolio_info_csv
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
        padded_Sigma = Sigma.copy()

        for dummy in dummy_tickers:
            # Add zero covariance row/column
            padded_Sigma.loc[dummy, :] = 0.0
            padded_Sigma.loc[:, dummy] = 0.0
            padded_Sigma.loc[dummy, dummy] = 1e-10  # Tiny variance to avoid singularity

        return padded_clusters, padded_mu, padded_Sigma, real_tickers

    def _combine_cluster_bqms(self, clusters, mu, Sigma):
        """
        Combine multiple independent cluster BQMs into one large BQM.

        This allows solving all clusters in a single QPU call.
        """
        import dimod

        combined_bqm = dimod.BinaryQuadraticModel('BINARY')
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
            bqm = self.formulator.formulate_cluster(
                cluster_tickers, cluster_mu, cluster_Sigma
            )

            # Add to combined BQM (clusters are independent, no couplings between them)
            combined_bqm.update(bqm)

            cluster_info.append({
                'id': cluster_id,
                'tickers': cluster_tickers,
                'variables': list(bqm.variables)
            })

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

            # 2. Create meta-cluster BQM
            meta_cluster_tickers = list(cluster_ids)
            meta_bqm = self.formulator.formulate_cluster(
                meta_cluster_tickers, cluster_mu, cluster_Sigma
            )

            # 3. Combine asset clusters + meta-cluster BQMs
            # (They are independent, no couplings between them)
            combined_bqm.update(meta_bqm)

            # Add meta-cluster info
            cluster_info.append({
                'id': 'META_CLUSTER',
                'tickers': meta_cluster_tickers,
                'variables': list(meta_bqm.variables)
            })

            # Use full portfolio template (pre-computed for N asset clusters + 1 meta-cluster)
            combined_embedding = self.embedding_mgr.get_full_portfolio_embedding(cluster_info, self.n_levels)

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

                if cluster_id == 'META_CLUSTER':
                    # Store meta-cluster weights separately
                    meta_cluster_weights = cluster_weights
                else:
                    # Store asset cluster weights
                    cluster_results[cluster_id] = {
                        'weights': cluster_weights,
                        'tickers': cluster_tickers
                    }

        elif self.solver_type == 'simulated':
            # For SA: Combine all asset clusters + meta-cluster into one BQM and solve together

            # 1. Create BQMs for all asset clusters
            combined_bqm, cluster_info = self._combine_cluster_bqms(clusters, mu, Sigma)

            # 2. Create meta-cluster BQM
            meta_cluster_tickers = list(cluster_ids)
            meta_bqm = self.formulator.formulate_cluster(
                meta_cluster_tickers, cluster_mu, cluster_Sigma
            )

            # 3. Combine asset clusters + meta-cluster BQMs
            combined_bqm.update(meta_bqm)

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

                if cluster_id == 'META_CLUSTER':
                    # Store meta-cluster weights separately
                    meta_cluster_weights = cluster_weights
                else:
                    # Store asset cluster weights
                    cluster_results[cluster_id] = {
                        'weights': cluster_weights,
                        'tickers': cluster_tickers
                    }

        # CLASSICAL APPLICATION OF META-CLUSTER WEIGHTS
        # Meta-cluster weights were solved simultaneously with asset clusters above
        # Now apply meta-cluster weights to scale asset weights

        # Convert meta-cluster weights to dict for lookup
        cluster_allocations = meta_cluster_weights.to_dict()

        # Apply meta-cluster weights to asset weights
        all_weights = {}
        for cluster_id, result in cluster_results.items():
            cluster_alloc = cluster_allocations[cluster_id]
            cluster_weights_unnorm = result['weights']  # pd.Series
            cluster_weight_sum = cluster_weights_unnorm.sum()

            if cluster_weight_sum > 0:
                # Scale asset weights by cluster allocation from meta-cluster
                cluster_weights_norm = cluster_weights_unnorm / cluster_weight_sum
                for ticker in cluster_weights_norm.index:
                    all_weights[ticker] = cluster_weights_norm[ticker] * cluster_alloc
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
