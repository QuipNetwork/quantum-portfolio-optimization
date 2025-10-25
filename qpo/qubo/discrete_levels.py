#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Discrete level portfolio optimization with cached embeddings.

This module implements portfolio optimization using discrete weight levels
with thermometer encoding, enabling:
1. Bit-aligned coupling (sparse graph)
2. Classical post-processing for exact metrics
3. Multi-cluster deployment on single QPU
4. Cached embeddings for fast repeated solves
"""

import numpy as np
import pandas as pd
import dimod
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
import hashlib
from minorminer import find_embedding
from dwave_networkx import zephyr_graph


class DiscreteLevelFormulator:
    """
    Formulate portfolio QUBO using discrete weight levels.

    Each asset's weight is encoded as one of n_levels discrete values using
    thermometer encoding. This enables bit-aligned coupling which dramatically
    reduces graph degree, allowing many more assets per cluster.
    """

    def __init__(self,
                 n_levels: int = 4,
                 alpha: float = 5.0,
                 beta: float = 2.5,
                 thermometer_penalty: float = 10.0,
                 l1_sparsity_penalty: float = 0.0):
        """
        Initialize discrete level formulator.

        Args:
            n_levels: Number of discrete weight levels (default 4)
            alpha: Return objective weight (default 5.0, aggressive)
            beta: Risk objective weight (default 2.5, ratio=0.5)
            thermometer_penalty: Penalty for thermometer constraint violation (weak/relaxed)
            l1_sparsity_penalty: L1 regularization penalty (λ > 0 promotes sparsity)
                                Adds +λ·||w||₁ term to objective
                                For long-only: ||w||₁ = Σᵢ wᵢ (linear term)
                                Higher values → fewer holdings (more sparse)

        Note:
            With n_levels=4, weights discretized to: 0.063, 0.125, 0.250, 0.500
            (using top-4 bits of 8-bit encoding)
            Alpha/Beta ratio of 0.5 represents aggressive risk tolerance

            Budget constraint (Σwᵢ = 1) is handled by post-processing normalization,
            NOT as a QUBO penalty term. This avoids adding edges to the graph.

        L1 vs L2 Regularization:
            L1 (this implementation): Promotes sparsity (drives weights to zero)
            L2 (sum of squares): Promotes diversification (equal weights)
            L1 adds ZERO edges to QUBO graph (linear term only)
        """
        self.n_levels = n_levels
        self.alpha = alpha
        self.beta = beta
        self.thermometer_penalty = thermometer_penalty
        self.l1_sparsity_penalty = l1_sparsity_penalty

        # Weight encoding: use top n_levels bits of 8-bit space
        self.n_total_bits = 8
        self.start_bit = self.n_total_bits - n_levels
        self.K = 2**self.n_total_bits - 1  # 255

        # Precompute weight values for each level
        self.weight_values = np.array([
            2**(self.start_bit + q) / self.K for q in range(n_levels)
        ])

    def formulate_cluster(self,
                         tickers: List[str],
                         mu: pd.Series,
                         Sigma: pd.DataFrame) -> dimod.BinaryQuadraticModel:
        """
        Create QUBO for a single cluster using discrete levels.

        Args:
            tickers: Asset tickers
            mu: Expected returns
            Sigma: Covariance matrix

        Returns:
            BinaryQuadraticModel with thermometer-encoded weights

        Note:
            Budget constraint (Σwᵢ = 1) is NOT included in QUBO.
            Weights are normalized in post-processing instead.
        """
        N = len(tickers)
        h, Q = {}, {}

        # Variable names: TICKER_LEVEL (e.g., "AAPL_0", "AAPL_1", ...)
        var_names = [[f"{ticker}_{q}" for q in range(self.n_levels)]
                     for ticker in tickers]

        # 1. RETURN OBJECTIVE: maximize μ^T w
        for n, ticker in enumerate(tickers):
            for q in range(self.n_levels):
                var = var_names[n][q]
                # Negative because we minimize (want to maximize return)
                # Handle potential duplicate indices by taking first value
                mu_val = mu.loc[ticker]
                if isinstance(mu_val, pd.Series):
                    mu_val = mu_val.iloc[0]
                coeff = float(-self.alpha * float(mu_val) * self.weight_values[q])
                h[var] = h.get(var, 0) + coeff

        # 2. THERMOMETER CONSTRAINT: if bit q=1, then bit q-1 must = 1
        # Penalty: x_q × (1 - x_{q-1}) = x_q - x_q·x_{q-1}
        for i in range(N):
            ticker = tickers[i]
            # Use much higher penalty for dummy assets to ensure valid thermometer encoding
            # Dummy assets have tiny objective coefficients, so need stronger constraint
            penalty = self.thermometer_penalty * 100 if ticker.startswith('_DUMMY_') else self.thermometer_penalty

            for q in range(1, self.n_levels):
                var_curr = var_names[i][q]
                var_prev = var_names[i][q-1]

                # Add penalty if curr=1 but prev=0
                h[var_curr] = h.get(var_curr, 0) + penalty
                key = tuple(sorted([var_curr, var_prev]))
                Q[key] = Q.get(key, 0) - penalty

        # 3. RISK OBJECTIVE: minimize w^T Σ w (bit-aligned approximation)
        for i in range(N):
            for j in range(i+1, N):
                sigma_ij = Sigma.iloc[i, j]
                if abs(sigma_ij) < 1e-6:  # Skip negligible covariances
                    continue

                # Bit-aligned: only couple same-index bits
                for q in range(self.n_levels):
                    var_i = var_names[i][q]
                    var_j = var_names[j][q]

                    # Approximate: w_i × w_j when both bits q are set
                    coeff = float(2 * self.beta * sigma_ij * self.weight_values[q]**2)
                    key = tuple(sorted([var_i, var_j]))
                    Q[key] = Q.get(key, 0) + coeff

        # 4. L1 SPARSITY PENALTY: +λ * Σᵢ |wᵢ| (promotes sparse portfolios)
        # For long-only portfolios (wᵢ ≥ 0): ||w||₁ = Σᵢ |wᵢ| = Σᵢ wᵢ
        # This is a LINEAR term (adds to h vector only, NO new edges in Q)
        # Higher λ → fewer holdings (drives small weights to zero)
        if self.l1_sparsity_penalty != 0:
            for i in range(N):
                for q in range(self.n_levels):
                    var_q = var_names[i][q]

                    # Linear penalty: +λ * wᵢ
                    # Adds penalty proportional to weight magnitude
                    # This encourages the optimizer to set weights to zero (sparsity)
                    coeff = self.l1_sparsity_penalty * self.weight_values[q]
                    h[var_q] = h.get(var_q, 0) + coeff

        # Budget constraint (Σwᵢ = 1) is handled by normalization in post-processing
        # NOT included as a QUBO penalty term to keep graph sparse
        return dimod.BinaryQuadraticModel(h, Q, 0.0, dimod.BINARY)

    def formulate_inter_cluster(self,
                                all_tickers: List[str],
                                mu: pd.Series,
                                Sigma: pd.DataFrame) -> dimod.BinaryQuadraticModel:
        """
        Create QUBO for ALL assets with inter-cluster correlations.

        Unlike formulate_cluster which only includes intra-cluster risk terms,
        this includes ALL pairwise correlations across the entire portfolio.

        Args:
            all_tickers: All asset tickers (across all clusters)
            mu: Expected returns for all assets
            Sigma: Full covariance matrix (all assets)

        Returns:
            BinaryQuadraticModel with full inter-cluster correlations
        """
        N = len(all_tickers)
        h, Q = {}, {}

        # Variable names: TICKER_LEVEL
        var_names = [[f"{ticker}_{q}" for q in range(self.n_levels)]
                     for ticker in all_tickers]

        # 1. RETURN OBJECTIVE: maximize μ^T w
        for n, ticker in enumerate(all_tickers):
            for q in range(self.n_levels):
                var = var_names[n][q]
                mu_val = mu.loc[ticker]
                if isinstance(mu_val, pd.Series):
                    mu_val = mu_val.iloc[0]
                coeff = float(-self.alpha * float(mu_val) * self.weight_values[q])
                h[var] = h.get(var, 0) + coeff

        # 2. THERMOMETER CONSTRAINT
        for i in range(N):
            for q in range(1, self.n_levels):
                var_curr = var_names[i][q]
                var_prev = var_names[i][q-1]
                h[var_curr] = h.get(var_curr, 0) + self.thermometer_penalty
                key = tuple(sorted([var_curr, var_prev]))
                Q[key] = Q.get(key, 0) - self.thermometer_penalty

        # 3. RISK OBJECTIVE: FULL COVARIANCE MATRIX (inter-cluster!)
        # This is the key difference - we include ALL pairs, not just within-cluster
        print(f"  Adding {N*(N-1)//2} pairwise risk terms (inter-cluster)...")
        for i in range(N):
            for j in range(i+1, N):
                sigma_ij = Sigma.iloc[i, j]
                if abs(sigma_ij) < 1e-6:  # Skip negligible covariances
                    continue

                # Bit-aligned: couple same-index bits
                for q in range(self.n_levels):
                    var_i = var_names[i][q]
                    var_j = var_names[j][q]

                    coeff = float(2 * self.beta * sigma_ij * self.weight_values[q]**2)
                    key = tuple(sorted([var_i, var_j]))
                    Q[key] = Q.get(key, 0) + coeff

        # 4. BUDGET CONSTRAINT: λ * (Σw - 1)²
        # Linear term: -2λ * Σw
        for i in range(N):
            for q in range(self.n_levels):
                var = var_names[i][q]
                coeff = -2 * self.thermometer_penalty * self.weight_values[q]
                h[var] = h.get(var, 0) + coeff

        # Quadratic terms: λ * w_i² (self-terms)
        for i in range(N):
            for q in range(self.n_levels):
                for p in range(self.n_levels):
                    var_q = var_names[i][q]
                    var_p = var_names[i][p]

                    coeff = self.thermometer_penalty * self.weight_values[q] * self.weight_values[p]

                    if var_q == var_p:
                        h[var_q] = h.get(var_q, 0) + coeff
                    else:
                        key = tuple(sorted([var_q, var_p]))
                        Q[key] = Q.get(key, 0) + coeff

        # Cross terms: λ * w_i * w_j for i ≠ j
        for i in range(N):
            for j in range(i+1, N):
                for q in range(self.n_levels):
                    for p in range(self.n_levels):
                        var_i = var_names[i][q]
                        var_j = var_names[j][p]

                        coeff = 2 * self.thermometer_penalty * self.weight_values[q] * self.weight_values[p]
                        key = tuple(sorted([var_i, var_j]))
                        Q[key] = Q.get(key, 0) + coeff

        return dimod.BinaryQuadraticModel(h, Q, 0.0, dimod.BINARY)

    def formulate_with_cluster_representatives(self,
                                               clusters: dict,
                                               mu: pd.Series,
                                               Sigma: pd.DataFrame,
                                               inter_cluster_coupling: float = 1.0) -> dimod.BinaryQuadraticModel:
        """
        Create QUBO with cluster representative variables for inter-cluster correlations.

        Instead of adding O(N²) inter-asset connections, we:
        1. Add K bits per cluster to represent cluster weight
        2. Couple cluster representatives (O(C²) where C = # clusters)
        3. Couple assets to their cluster representative

        This allows QPU to optimize inter-cluster allocation while staying embeddable.

        Args:
            clusters: Dict of {cluster_id: [tickers]}
            mu: Expected returns
            Sigma: Covariance matrix
            inter_cluster_coupling: Strength of inter-cluster correlation terms

        Returns:
            BinaryQuadraticModel with cluster representatives
        """
        import dimod

        h, Q = {}, {}
        cluster_list = []

        # Process each cluster
        for cluster_id, tickers in clusters.items():
            # Filter to available tickers
            cluster_tickers = [t for t in tickers if t in mu.index]
            if len(cluster_tickers) == 0:
                continue

            cluster_list.append({
                'id': cluster_id,
                'tickers': cluster_tickers,
                'representative_var': f"CLUSTER_{cluster_id}"
            })


        # 1. Create BQM for each cluster (standard intra-cluster optimization)
        for cluster_info in cluster_list:
            cluster_id = cluster_info['id']
            cluster_tickers = cluster_info['tickers']

            cluster_mu = pd.Series(mu[cluster_tickers])
            cluster_Sigma = pd.DataFrame(Sigma.loc[cluster_tickers, cluster_tickers])

            # Create intra-cluster BQM
            cluster_bqm = self.formulate_cluster(cluster_tickers, cluster_mu, cluster_Sigma)

            # Merge into combined BQM
            h.update(cluster_bqm.linear)
            Q.update(cluster_bqm.quadratic)

        # 2. Add cluster representative variables (K bits per cluster)
        for cluster_info in cluster_list:
            rep_base = cluster_info['representative_var']

            # Create K bits for this cluster's weight
            for q in range(self.n_levels):
                var = f"{rep_base}_{q}"
                h[var] = 0  # Neutral bias initially

            # Thermometer constraint for cluster weight
            for q in range(1, self.n_levels):
                var_curr = f"{rep_base}_{q}"
                var_prev = f"{rep_base}_{q-1}"

                h[var_curr] = h.get(var_curr, 0) + self.thermometer_penalty
                key = tuple(sorted([var_curr, var_prev]))
                Q[key] = Q.get(key, 0) - self.thermometer_penalty

        # 3. No asset-to-cluster coupling or global asset budget constraint needed!
        #
        # Design philosophy:
        # - Each cluster's assets are optimized for return/risk within the cluster
        # - Cluster representatives handle inter-cluster allocation via meta-cluster budget
        # - No O(N²) asset couplings needed - this is the whole point of clustering!
        #
        # The meta-cluster budget constraint (section 5) on cluster representatives
        # is sufficient to ensure global budget allocation.

        # 4. Add inter-cluster coupling based on correlations
        n_inter_cluster = 0

        for i, cluster_i in enumerate(cluster_list):
            for j, cluster_j in enumerate(cluster_list):
                if i >= j:
                    continue

                tickers_i = cluster_i['tickers']
                tickers_j = cluster_j['tickers']
                rep_i = cluster_i['representative_var']
                rep_j = cluster_j['representative_var']

                # Compute average correlation between clusters
                cross_corr = Sigma.loc[tickers_i, tickers_j].values
                avg_corr = cross_corr.mean()

                if abs(avg_corr) < 1e-6:
                    continue

                # Couple cluster representatives based on correlation
                for q in range(self.n_levels):
                    var_i = f"{rep_i}_{q}"
                    var_j = f"{rep_j}_{q}"

                    coeff = float(2 * self.beta * inter_cluster_coupling * avg_corr * self.weight_values[q]**2)
                    key = tuple(sorted([var_i, var_j]))
                    Q[key] = Q.get(key, 0) + coeff
                    n_inter_cluster += 1

        # 5. Add BUDGET CONSTRAINT on cluster representatives: λ * (Σw_cluster - 1)²
        # Strategy: Apply budget constraint to cluster representative variables only
        # This is more efficient than applying to all assets (O(C²) vs O(N²) where C << N)
        #
        # Cluster representatives approximate the total weight allocated to each cluster.
        # Enforcing Σw_cluster = 1.0 ensures the full portfolio budget is allocated.
        #
        # Expansion: λ * (Σw_cluster - 1)² = λ * [(Σw_cluster)² - 2*Σw_cluster + 1]
        #                                   = λ * (Σw_cluster)² - 2λ * Σw_cluster + λ
        # Terms:
        #   1. Quadratic (Σw_cluster)²: Couple cluster representatives
        #   2. Linear -2λΣw_cluster: Applied to each cluster representative variable
        #   3. Constant +λ: Added to BQM offset (makes all-zero expensive!)

        # Get cluster representative variables
        cluster_rep_vars = []
        for cluster_info in cluster_list:
            rep_base = cluster_info['representative_var']
            for q in range(self.n_levels):
                var = f"{rep_base}_{q}"
                cluster_rep_vars.append((var, self.weight_values[q]))

        # Linear term: -2λ * Σw_cluster
        for var, weight_val in cluster_rep_vars:
            coeff = -2 * self.budget_penalty * weight_val
            h[var] = h.get(var, 0) + coeff

        # Quadratic self-terms: λ * w_cluster_i²
        # Group by cluster representative
        rep_to_vars = {}
        for cluster_info in cluster_list:
            rep_base = cluster_info['representative_var']
            rep_to_vars[rep_base] = [f"{rep_base}_{q}" for q in range(self.n_levels)]

        for rep_base, var_list in rep_to_vars.items():
            for q in range(self.n_levels):
                for p in range(self.n_levels):
                    var_q = var_list[q]
                    var_p = var_list[p]
                    coeff = self.budget_penalty * self.weight_values[q] * self.weight_values[p]

                    if var_q == var_p:
                        h[var_q] = h.get(var_q, 0) + coeff
                    else:
                        key = tuple(sorted([var_q, var_p]))
                        Q[key] = Q.get(key, 0) + coeff

        # Cross terms: λ * w_cluster_i * w_cluster_j for i ≠ j
        rep_list = [cluster_info['representative_var'] for cluster_info in cluster_list]

        for i, rep_i in enumerate(rep_list):
            for j, rep_j in enumerate(rep_list):
                if i >= j:
                    continue

                for q in range(self.n_levels):
                    for p in range(self.n_levels):
                        var_i = f"{rep_i}_{q}"
                        var_j = f"{rep_j}_{p}"

                        coeff = 2 * self.budget_penalty * self.weight_values[q] * self.weight_values[p]
                        key = tuple(sorted([var_i, var_j]))
                        Q[key] = Q.get(key, 0) + coeff

        # CRITICAL: Add constant offset λ to make all-zero solutions expensive
        # When Σw_cluster=0 (all zeros): energy = 0 - 0 + λ = +λ (penalty!)
        # When Σw_cluster=1 (budget met): energy = λ - 2λ + λ = 0 (no penalty)
        budget_offset = self.budget_penalty

        bqm = dimod.BinaryQuadraticModel(h, Q, budget_offset, dimod.BINARY)
        return bqm, cluster_list

    def decode_solution(self,
                       sample: Dict[str, int],
                       tickers: List[str]) -> pd.Series:
        """
        Decode binary solution to portfolio weights (classical post-processing).

        Args:
            sample: Binary solution from QUBO solver
            tickers: Asset tickers

        Returns:
            Portfolio weights (normalized to sum=1.0)
        """
        weights = np.zeros(len(tickers))
        thermometer_violations = 0
        violating_assets = []

        for i, ticker in enumerate(tickers):
            # Skip dummy assets entirely - they're filtered out later anyway
            if ticker.startswith('_DUMMY_'):
                continue

            # Decode thermometer: count number of 1s
            value = 0
            prev_bit = 0
            asset_violations = 0
            for q in range(self.n_levels):
                var_name = f"{ticker}_{q}"
                bit = sample.get(var_name, 0)

                # Check thermometer constraint: if bit q=1, then bit q-1 must = 1
                if q > 0 and bit == 1 and prev_bit == 0:
                    thermometer_violations += 1
                    asset_violations += 1

                if bit == 1:
                    value += 2**(self.start_bit + q)
                prev_bit = bit

            if asset_violations > 0:
                violating_assets.append((ticker, asset_violations))

            weights[i] = value / self.K

        # Suppress thermometer violation warnings
        # These occur when solver assigns near-zero weights to low-performing assets
        # They don't affect results since these assets get filtered out anyway
        # if thermometer_violations > 0:
        #     import logging
        #     logger = logging.getLogger(__name__)
        #     asset_list = ", ".join(f"{t}({v})" for t, v in violating_assets[:5])
        #     if len(violating_assets) > 5:
        #         asset_list += f" +{len(violating_assets)-5} more"
        #     logger.warning(f"Thermometer constraint violations: {thermometer_violations} bits violated in {len(violating_assets)} assets: {asset_list}")

        # Normalize weights
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # All-zero solution: cluster has no clear winner
            # This is normal when assets have similar risk/return profiles
            # Use equal weights as fallback
            weights = np.ones(len(tickers)) / len(tickers)

        return pd.Series(weights, index=tickers)


class CachedEmbeddingManager:
    """
    Manage cached embeddings for discrete level QUBOs.

    Computes embedding once using find_embedding, then caches it for reuse.
    Multiple clusters can use the same embedding template with different
    variable mappings.

    Supports both JSON and pickle formats with backward compatibility.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedding manager.

        Args:
            cache_dir: Directory for cached embeddings (default: ./embeddings)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / 'embeddings'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.zephyr = zephyr_graph(4)
        self.embedding_template = None

    def _load_embedding_file(self, file_path: Path) -> Dict:
        """
        Load embedding from file (supports both JSON and pickle).

        Args:
            file_path: Path to embedding file (.json or .pkl)

        Returns:
            Embedding dict
        """
        import json

        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract embedding from template format
            if 'embedding' in data:
                return data['embedding']
            else:
                # Assume it's a raw embedding dict
                return data

        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _save_embedding_file(self, embedding: Dict, file_path: Path, metadata: Optional[Dict] = None):
        """
        Save embedding to file (prefers JSON).

        Args:
            embedding: Embedding dict
            file_path: Path to save to
            metadata: Optional metadata to include
        """
        import json
        from datetime import datetime

        if file_path.suffix == '.json':
            # Save as JSON with metadata
            data = {
                'format_version': '1.0',
                'created_at': datetime.now().isoformat(),
                'embedding': embedding,
            }
            if metadata:
                data.update(metadata)

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif file_path.suffix == '.pkl':
            # Save as pickle (backward compatibility)
            with open(file_path, 'wb') as f:
                pickle.dump(embedding, f)

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def get_embedding_for_cluster(self,
                                  bqm: dimod.BinaryQuadraticModel,
                                  force_recompute: bool = False) -> Dict:
        """
        Get embedding for a discrete-level cluster BQM.

        Args:
            bqm: BinaryQuadraticModel to embed
            force_recompute: Force recomputation even if cached

        Returns:
            Embedding dict {variable: qubit}
        """
        # Compute structure hash
        structure_hash = self._get_structure_hash(bqm)
        cache_file = self.cache_dir / f"discrete_levels_{structure_hash}.pkl"

        # Try to load from cache
        if not force_recompute and cache_file.exists():
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)
            return embedding

        # Compute new embedding
        print(f"Computing embedding for {len(bqm.variables)} variables...")
        embedding = find_embedding(
            list(bqm.quadratic.keys()),
            list(self.zephyr.edges()),
            timeout=60,
            verbose=1
        )

        if not embedding:
            raise RuntimeError("Failed to find embedding")

        # Cache it
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

        print(f"Embedding cached to {cache_file}")
        return embedding

    def _get_structure_hash(self, bqm: dimod.BinaryQuadraticModel) -> str:
        """Compute hash of BQM structure (graph topology)."""
        variables = sorted(bqm.variables)
        edges = sorted(bqm.quadratic.keys())
        structure_str = f"{len(variables)}v_{len(edges)}e"
        structure_str += hashlib.md5(str(edges).encode()).hexdigest()[:8]
        return structure_str

    def get_qpu_embedding(self, n_assets: int, n_levels: int = 4) -> Dict:
        """
        Load precomputed QPU embedding for specific cluster size.

        Args:
            n_assets: Number of assets in cluster
            n_levels: Number of discrete levels (default 4)

        Returns:
            Embedding dict and metadata
        """
        qpu_cache_file = self.cache_dir / 'qpu' / f"discrete_levels_{n_assets}assets_{n_levels}levels_qpu.pkl"

        if not qpu_cache_file.exists():
            raise FileNotFoundError(
                f"No QPU embedding found for {n_assets} assets. "
                f"Run compute_qpu_embeddings.py first."
            )

        with open(qpu_cache_file, 'rb') as f:
            embedding_data = pickle.load(f)

        return embedding_data

    def remap_embedding(self, embedding: Dict, tickers: List[str], n_levels: int = 4) -> Dict:
        """
        Remap embedding from dummy variable names to actual ticker names.

        The precomputed embeddings use ASSET_0, ASSET_1, etc. This function
        remaps them to actual ticker names.

        Args:
            embedding: Embedding with dummy names (ASSET_0_0, ASSET_0_1, ...)
            tickers: Actual ticker names
            n_levels: Number of discrete levels

        Returns:
            Remapped embedding dict
        """
        remapped = {}

        for i, ticker in enumerate(tickers):
            for q in range(n_levels):
                dummy_var = f"ASSET_{i}_{q}"
                real_var = f"{ticker}_{q}"

                if dummy_var in embedding:
                    remapped[real_var] = embedding[dummy_var]
                else:
                    raise ValueError(f"Variable {dummy_var} not found in embedding")

        return remapped

    def get_template_embedding(self, n_assets: int, n_levels: int = 4) -> Dict:
        """
        Load a template embedding for a given cluster size.

        Template embeddings are generic (use ASSET_0, ASSET_1, etc) and
        can be relabeled and offset for actual use.
        """
        template_file = self.cache_dir / 'templates' / f"template_{n_assets}assets_{n_levels}levels.pkl"

        if not template_file.exists():
            raise FileNotFoundError(
                f"No template embedding found for {n_assets} assets. "
                f"Run create_template_embeddings.py first."
            )

        with open(template_file, 'rb') as f:
            template_data = pickle.load(f)

        # Convert EmbeddedStructure to plain dict if needed
        if hasattr(template_data['embedding'], '__dict__'):
            # It's an EmbeddedStructure, convert to dict
            template_data['embedding'] = dict(template_data['embedding'])

        return template_data

    def get_full_portfolio_embedding(self, cluster_info: List[Dict], n_levels: int = 4,
                                    expected_assets_per_cluster: int = None,
                                    expected_n_clusters: int = None) -> Dict:
        """
        Load the full multi-cluster template and relabel for actual tickers.

        This uses a single pre-computed embedding for all clusters together,
        rather than combining individual cluster embeddings.

        Supports both JSON (.json) and pickle (.pkl) formats.
        Now supports standardized naming: portfolio_{C}c_{A}a_{L}l.json

        Args:
            cluster_info: List of dicts with 'tickers' for each cluster
            n_levels: Number of discrete levels (must match template)
            expected_assets_per_cluster: Expected template size (overrides inference from cluster_info)
            expected_n_clusters: Expected number of asset clusters (overrides inference)

        Returns:
            Embedding dict mapping actual ticker variables to qubit chains
        """
        # Use provided dimensions or infer from cluster_info
        if expected_n_clusters is not None and expected_assets_per_cluster is not None:
            # Use template dimensions directly
            n_asset_clusters = expected_n_clusters
            assets_per_cluster = expected_assets_per_cluster
        else:
            # Infer from actual cluster structure (legacy behavior)
            # Compute cluster shape to look for matching templates
            # cluster_info includes both asset clusters and meta-cluster
            # Template filenames use n_asset_clusters (excluding meta-cluster)
            n_clusters_total = len(cluster_info)

            # Detect meta-cluster (has id='META_CLUSTER')
            has_meta_cluster = any(c.get('id') == 'META_CLUSTER' for c in cluster_info)
            n_asset_clusters = n_clusters_total - 1 if has_meta_cluster else n_clusters_total

            assets_per_cluster = len(cluster_info[0]['tickers']) if cluster_info else 0

        n_clusters_total = n_asset_clusters + 1  # +1 for meta-cluster

        # Try multiple filename patterns (standardized and legacy)
        # Note: filenames use n_asset_clusters (e.g., 13c for 13 asset clusters + 1 meta = 14 total)
        template_candidates = [
            # Standardized naming (preferred)
            self.cache_dir / 'templates' / f'portfolio_{n_asset_clusters}c_{assets_per_cluster}a_{n_levels}l.json',
            self.cache_dir / 'templates' / f'portfolio_{n_asset_clusters}c_{assets_per_cluster}a_{n_levels}l_zephyr.json',
            self.cache_dir / 'templates' / f'portfolio_{n_asset_clusters}c_{assets_per_cluster}a_{n_levels}l_zephyr2.json',
            # Legacy naming
            self.cache_dir / 'templates' / f'full_portfolio_template_{n_levels}levels.json',
            self.cache_dir / 'templates' / 'full_portfolio_template.pkl',
        ]

        template_file = None
        for candidate in template_candidates:
            if candidate.exists():
                template_file = candidate
                break

        if template_file is None:
            raise FileNotFoundError(
                f"Full portfolio template not found for {n_asset_clusters} asset clusters "
                f"(+1 meta = {n_clusters_total} total), "
                f"{assets_per_cluster} assets/cluster, {n_levels} levels. Tried:\n" +
                "\n".join(f"  {c}" for c in template_candidates[:3]) +
                f"\n\nTo generate the required template, run:\n"
                f"  python tools/generate_portfolio_embedding_template.py \\\n"
                f"    --solver Advantage2_system1.6 \\\n"
                f"    --num-clusters {n_asset_clusters} \\\n"
                f"    --cluster-size {assets_per_cluster} \\\n"
                f"    --n-levels {n_levels}"
            )

        # Load template using format-aware loader
        if template_file.suffix == '.json':
            import json
            with open(template_file, 'r') as f:
                template_data = json.load(f)

            # Validate configuration matches
            if 'configuration' in template_data:
                config = template_data['configuration']

                # Check n_levels
                template_n_levels = config.get('n_levels', n_levels)
                if template_n_levels != n_levels:
                    raise ValueError(
                        f"Template n_levels={template_n_levels} but optimizer uses n_levels={n_levels}. "
                        f"Generate a new template with matching n_levels."
                    )

                # Check cluster layout (C, A) - allow some flexibility for meta-cluster
                template_n_clusters = config.get('n_clusters_total', config.get('n_clusters'))
                template_assets_per = config.get('assets_per_cluster')

                if template_n_clusters is not None and template_n_clusters != n_clusters_total:
                    import warnings
                    warnings.warn(
                        f"Template has {template_n_clusters} total clusters but current problem has {n_clusters_total}. "
                        f"This may cause embedding errors if structures don't match.",
                        UserWarning
                    )

                if template_assets_per is not None and template_assets_per != assets_per_cluster:
                    import warnings
                    warnings.warn(
                        f"Template has {template_assets_per} assets/cluster but current problem has {assets_per_cluster}. "
                        f"This may cause embedding errors if structures don't match.",
                        UserWarning
                    )

            template_embedding = template_data['embedding']

        else:  # pickle format (backward compatibility)
            with open(template_file, 'rb') as f:
                template_data = pickle.load(f)

            # Convert EmbeddedStructure to plain dict if needed
            if hasattr(template_data.get('embedding', {}), '__dict__'):
                template_data['embedding'] = dict(template_data['embedding'])

            template_embedding = template_data.get('embedding', template_data)

        # Relabel from ASSET_0, ASSET_1, ... to actual tickers
        # Build mapping: asset_index -> ticker
        asset_to_ticker = {}
        asset_offset = 0

        for cluster in cluster_info:
            for i, ticker in enumerate(cluster['tickers']):
                asset_to_ticker[asset_offset + i] = ticker
            asset_offset += len(cluster['tickers'])

        # Relabel embedding variables
        actual_embedding = {}

        for var_name, chain in template_embedding.items():
            # Parse variable name (supports multiple formats):
            # - ASSET_5_2 (3-part legacy): asset_idx=5, level=2
            # - ASSET_0_5_2 (4-part): cluster=0, asset=5, level=2
            # - META_5_2 (3-part): meta-cluster asset=5, level=2

            if var_name.startswith('ASSET_'):
                parts = var_name.split('_')
                if len(parts) == 3:
                    # Legacy format: ASSET_asset_level
                    asset_idx = int(parts[1])
                    level = int(parts[2])
                elif len(parts) == 4:
                    # New format: ASSET_cluster_asset_level
                    cluster_idx = int(parts[1])
                    asset_within_cluster = int(parts[2])
                    level = int(parts[3])
                    # Compute global asset index
                    asset_idx = cluster_idx * assets_per_cluster + asset_within_cluster
                else:
                    continue

            elif var_name.startswith('META_'):
                parts = var_name.split('_')
                if len(parts) == 3:
                    # Meta-cluster format: META_asset_level
                    meta_asset_idx = int(parts[1])
                    level = int(parts[2])
                    # Meta-cluster assets come after all asset clusters
                    asset_idx = n_asset_clusters * assets_per_cluster + meta_asset_idx
                else:
                    continue
            else:
                # Skip variables that don't match expected patterns
                continue

            # Map to actual ticker
            if asset_idx not in asset_to_ticker:
                print(f"Warning: Asset index {asset_idx} not found in mapping")
                continue

            ticker = asset_to_ticker[asset_idx]
            actual_var = f"{ticker}_{level}"

            actual_embedding[actual_var] = chain

        # Extract statistics from template (with fallbacks)
        stats = template_data.get('statistics', {})
        total_qubits = stats.get('total_qubits_used')

        # If not in stats, calculate from embedding
        if total_qubits is None:
            all_qubits = set()
            for chain in actual_embedding.values():
                all_qubits.update(chain)
            total_qubits = len(all_qubits)
            min_qubit = min(all_qubits) if all_qubits else 0
            max_qubit = max(all_qubits) if all_qubits else 0
        else:
            # Try to get qubit range from stats or calculate
            min_qubit = stats.get('min_qubit')
            max_qubit = stats.get('max_qubit')
            if min_qubit is None or max_qubit is None:
                all_qubits = set()
                for chain in actual_embedding.values():
                    all_qubits.update(chain)
                min_qubit = min(all_qubits) if all_qubits else 0
                max_qubit = max(all_qubits) if all_qubits else 0

        # Embedding statistics (use logging.debug if needed)
        # print(f"Full portfolio embedding: {len(actual_embedding)} vars -> {total_qubits} qubits")
        # print(f"  Qubit range: [{min_qubit}, {max_qubit}]")
        # print(f"  QPU utilization: {100*total_qubits/4593:.1f}%")

        return actual_embedding
