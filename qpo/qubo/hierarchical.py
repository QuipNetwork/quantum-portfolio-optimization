"""Hierarchical two-pass QUBO optimization for clusters."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import dimod


class ClusterStatisticsCalculator:
    """
    Compute cluster-level statistics for hierarchical optimization.

    Treats each optimized cluster as a "meta-asset" with its own
    expected return, variance, and covariances with other clusters.
    """

    def __init__(self):
        """Initialize cluster statistics calculator."""
        pass

    def compute_cluster_statistics(
        self,
        cluster_weights: Dict[str, pd.Series],
        clusters: Dict[str, List[str]],
        mu: pd.Series,
        Sigma: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute cluster-level expected returns and covariance matrix.

        For each cluster C_i with weights w_i:
            - Expected return: μ_i = w_i^T * μ_assets_i / ||w_i||_1
            - Variance: σ_i² = w_i^T * Σ_assets_i * w_i / ||w_i||_1²
            - Covariance: σ_ij = w_i^T * Σ_assets_ij * w_j / (||w_i||_1 * ||w_j||_1)

        Args:
            cluster_weights: {cluster_id: Series of asset weights within cluster}
            clusters: {cluster_id: [tickers]}
            mu: Global expected returns (all assets)
            Sigma: Global covariance matrix (all assets)

        Returns:
            (cluster_mu, cluster_Sigma) where:
                - cluster_mu: Series of cluster expected returns
                - cluster_Sigma: DataFrame of cluster covariance matrix
        """
        cluster_ids = list(clusters.keys())
        K = len(cluster_ids)

        # Initialize cluster-level statistics
        cluster_mu = pd.Series(index=cluster_ids, dtype=float)
        cluster_Sigma = pd.DataFrame(
            index=cluster_ids,
            columns=cluster_ids,
            dtype=float
        )

        # Normalize cluster weights for statistics computation
        normalized_weights = {}
        for cid, weights in cluster_weights.items():
            weight_sum = weights.sum()
            if weight_sum > 1e-10:
                normalized_weights[cid] = weights / weight_sum
            else:
                # Handle edge case: all zeros
                n_assets = len(weights)
                normalized_weights[cid] = pd.Series(
                    1.0/n_assets,
                    index=weights.index
                )

        # Compute cluster expected returns
        for cid, tickers in clusters.items():
            weights_norm = normalized_weights[cid]

            # μ_cluster = w^T * μ
            cluster_return = (weights_norm * mu[tickers]).sum()
            cluster_mu[cid] = cluster_return

        # Compute cluster covariance matrix
        for i, cid_i in enumerate(cluster_ids):
            tickers_i = clusters[cid_i]
            weights_i = normalized_weights[cid_i]

            for j, cid_j in enumerate(cluster_ids):
                tickers_j = clusters[cid_j]
                weights_j = normalized_weights[cid_j]

                # Extract cross-covariance block between clusters
                Sigma_ij = Sigma.loc[tickers_i, tickers_j]

                # σ_ij = w_i^T * Σ_ij * w_j
                cluster_cov = weights_i @ Sigma_ij @ weights_j
                cluster_Sigma.loc[cid_i, cid_j] = cluster_cov

        return cluster_mu, cluster_Sigma

    def validate_cluster_statistics(
        self,
        cluster_mu: pd.Series,
        cluster_Sigma: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Validate cluster-level statistics.

        Args:
            cluster_mu: Cluster expected returns
            cluster_Sigma: Cluster covariance matrix

        Returns:
            (is_valid, error_message)
        """
        # Check for NaN or Inf
        if cluster_mu.isna().any() or np.isinf(cluster_mu).any():
            return False, "Cluster returns contain NaN or Inf"

        if cluster_Sigma.isna().any().any() or np.isinf(cluster_Sigma).any().any():
            return False, "Cluster covariance contains NaN or Inf"

        # Check symmetry
        if not np.allclose(cluster_Sigma, cluster_Sigma.T):
            return False, "Cluster covariance matrix is not symmetric"

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(cluster_Sigma.values)
        if np.any(eigenvalues < -1e-6):
            return False, f"Cluster covariance has negative eigenvalues: {eigenvalues.min()}"

        return True, ""


class ClusterLevelQUBOFormulator:
    """
    Formulate cluster allocation optimization as QUBO.

    This is the "Pass 2" optimization that determines how to allocate
    budget across optimized clusters.
    """

    def __init__(
        self,
        n_bits: int = 10,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_budget: float = 10.0
    ):
        """
        Initialize cluster-level QUBO formulator.

        Args:
            n_bits: Binary discretization bits for cluster weights
            alpha: Weight on cluster expected return (higher = more aggressive)
            beta: Weight on inter-cluster risk (higher = more conservative)
            lambda_budget: Penalty for budget violations

        Note:
            Uses same encoding as asset-level QUBO for consistency.
            Each cluster gets a weight w_c ∈ [0, 1] encoded in n_bits.
        """
        self.n_bits = n_bits
        self.K = 2**n_bits - 1  # Normalization constant
        self.alpha = alpha
        self.beta = beta
        self.lambda_budget = lambda_budget

    def formulate_cluster_allocation(
        self,
        cluster_ids: List[str],
        cluster_mu: pd.Series,
        cluster_Sigma: pd.DataFrame,
        budget: float = 1.0
    ) -> dimod.BinaryQuadraticModel:
        """
        Create QUBO for optimal cluster allocation.

        Binary encoding: w_c = (1/K) * Σ 2^q * x_{c,q}

        Objective: H = H_risk - H_return + H_budget
            H_risk = β * w_c^T Σ_cluster w_c  (minimize inter-cluster variance)
            H_return = α * μ_cluster^T w_c  (maximize cluster returns)
            H_budget = λ * (Σw_c - B)²  (enforce budget)

        Args:
            cluster_ids: List of cluster IDs
            cluster_mu: Expected returns for each cluster
            cluster_Sigma: Covariance matrix between clusters
            budget: Budget constraint (sum of cluster weights, default 1.0)

        Returns:
            dimod.BinaryQuadraticModel for cluster allocation
        """
        K_clusters = len(cluster_ids)

        # Initialize QUBO
        Q = {}  # Quadratic terms
        h = {}  # Linear terms

        # Create variable names: "CLUSTER_ID_BIT" (e.g., "cluster_0_0", "cluster_0_1", ...)
        var_names = [
            [f"{cluster_id}_{q}" for q in range(self.n_bits)]
            for cluster_id in cluster_ids
        ]

        # 1. RETURN TERM (linear, negative to maximize)
        # H_return = -α * Σ μ_c * w_c
        for c, cluster_id in enumerate(cluster_ids):
            mu_c = cluster_mu[cluster_id]

            for q in range(self.n_bits):
                var = var_names[c][q]
                coeff = -self.alpha * mu_c * (2**q) / self.K
                h[var] = h.get(var, 0) + coeff

        # 2. RISK TERM (quadratic)
        # H_risk = β * Σ_{i,j} w_i * Σ_{i,j} * w_j
        for i in range(K_clusters):
            for j in range(K_clusters):
                sigma_ij = cluster_Sigma.iloc[i, j]

                for q in range(self.n_bits):
                    for p in range(self.n_bits):
                        var_i = var_names[i][q]
                        var_j = var_names[j][p]

                        coeff = self.beta * sigma_ij * (2**(q+p)) / (self.K**2)

                        if var_i == var_j:
                            # Diagonal term
                            h[var_i] = h.get(var_i, 0) + coeff
                        else:
                            # Off-diagonal term
                            key = tuple(sorted([var_i, var_j]))
                            Q[key] = Q.get(key, 0) + coeff

        # 3. BUDGET CONSTRAINT (penalty)
        # H_budget = λ * (Σw - B)²

        # Linear term: -2B * Σw
        for c in range(K_clusters):
            for q in range(self.n_bits):
                var = var_names[c][q]
                coeff = -2 * self.lambda_budget * budget * (2**q) / self.K
                h[var] = h.get(var, 0) + coeff

        # Quadratic terms: w_c² (self-terms)
        for c in range(K_clusters):
            for q in range(self.n_bits):
                for p in range(self.n_bits):
                    var_q = var_names[c][q]
                    var_p = var_names[c][p]

                    coeff = self.lambda_budget * (2**(q+p)) / (self.K**2)

                    if var_q == var_p:
                        h[var_q] = h.get(var_q, 0) + coeff
                    else:
                        key = tuple(sorted([var_q, var_p]))
                        Q[key] = Q.get(key, 0) + coeff

        # Cross terms: w_i * w_j for i ≠ j
        for i in range(K_clusters):
            for j in range(i+1, K_clusters):
                for q in range(self.n_bits):
                    for p in range(self.n_bits):
                        var_i = var_names[i][q]
                        var_j = var_names[j][p]

                        coeff = 2 * self.lambda_budget * (2**(q+p)) / (self.K**2)
                        key = tuple(sorted([var_i, var_j]))
                        Q[key] = Q.get(key, 0) + coeff

        # Constant offset: B²
        offset = self.lambda_budget * budget**2

        # Build Binary Quadratic Model
        bqm = dimod.BinaryQuadraticModel(h, Q, offset, dimod.BINARY)

        return bqm

    def decode_cluster_allocation(
        self,
        solution: Dict[str, int],
        cluster_ids: List[str]
    ) -> pd.Series:
        """
        Decode binary solution to cluster weights.

        Args:
            solution: Binary variable assignments {var_name: 0|1}
            cluster_ids: List of cluster IDs

        Returns:
            pd.Series with weight for each cluster
        """
        cluster_weights = {}

        for cluster_id in cluster_ids:
            # Sum binary values weighted by powers of 2
            binary_value = 0
            for q in range(self.n_bits):
                var_name = f"{cluster_id}_{q}"
                if var_name in solution:
                    binary_value += int(solution[var_name]) * (2**q)

            # Convert to continuous weight
            weight = binary_value / self.K
            cluster_weights[cluster_id] = weight

        return pd.Series(cluster_weights)


class TwoPassHierarchicalOptimizer:
    """
    Orchestrator for two-pass hierarchical QUBO optimization.

    Pass 1: Optimize assets within each cluster (intra-cluster)
    Pass 2: Optimize allocation across clusters (inter-cluster)
    Combine: Scale asset weights by cluster allocation weights
    """

    def __init__(
        self,
        intra_cluster_params: Dict,
        inter_cluster_params: Dict
    ):
        """
        Initialize two-pass optimizer.

        Args:
            intra_cluster_params: Parameters for Pass 1 (asset-level QUBO)
                - n_bits, alpha, beta, lambda_budget
            inter_cluster_params: Parameters for Pass 2 (cluster-level QUBO)
                - n_bits, alpha, beta, lambda_budget
        """
        self.intra_params = intra_cluster_params
        self.inter_params = inter_cluster_params

        self.stats_calculator = ClusterStatisticsCalculator()
        self.cluster_formulator = ClusterLevelQUBOFormulator(**inter_cluster_params)

    def compute_final_weights(
        self,
        cluster_weights_pass1: Dict[str, pd.Series],
        cluster_allocation_pass2: pd.Series,
        clusters: Dict[str, List[str]]
    ) -> pd.Series:
        """
        Combine Pass 1 and Pass 2 results into final portfolio weights.

        Final weight for asset i in cluster c:
            w_i_final = w_i_cluster * w_c_allocation

        Where:
            - w_i_cluster: Normalized weight of asset i within cluster c (Pass 1)
            - w_c_allocation: Weight allocated to cluster c (Pass 2)

        Args:
            cluster_weights_pass1: {cluster_id: Series of asset weights}
            cluster_allocation_pass2: Series of cluster weights
            clusters: {cluster_id: [tickers]}

        Returns:
            pd.Series with final portfolio weights (all assets)
        """
        final_weights = {}

        for cluster_id, tickers in clusters.items():
            # Get cluster allocation from Pass 2
            cluster_budget = cluster_allocation_pass2[cluster_id]

            # Get asset weights from Pass 1
            asset_weights = cluster_weights_pass1[cluster_id]

            # Normalize asset weights within cluster
            asset_sum = asset_weights.sum()
            if asset_sum > 1e-10:
                normalized_asset_weights = asset_weights / asset_sum
            else:
                # Equal weight if cluster optimization failed
                n = len(tickers)
                normalized_asset_weights = pd.Series(1.0/n, index=tickers)

            # Scale by cluster allocation
            final_asset_weights = normalized_asset_weights * cluster_budget

            # Add to global portfolio
            final_weights.update(final_asset_weights.to_dict())

        return pd.Series(final_weights)

    def optimize_hierarchically(
        self,
        cluster_weights_pass1: Dict[str, pd.Series],
        clusters: Dict[str, List[str]],
        mu: pd.Series,
        Sigma: pd.DataFrame,
        solver
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Execute Pass 2 optimization and combine with Pass 1 results.

        Args:
            cluster_weights_pass1: Results from Pass 1 {cluster_id: asset_weights}
            clusters: {cluster_id: [tickers]}
            mu: Global expected returns (all assets)
            Sigma: Global covariance matrix (all assets)
            solver: Quantum solver instance for Pass 2

        Returns:
            (final_weights, metadata) where:
                - final_weights: pd.Series of final portfolio weights
                - metadata: Dict with Pass 2 statistics and diagnostics
        """
        # Step 1: Compute cluster-level statistics
        cluster_mu, cluster_Sigma = self.stats_calculator.compute_cluster_statistics(
            cluster_weights_pass1,
            clusters,
            mu,
            Sigma
        )

        # Validate cluster statistics
        is_valid, error_msg = self.stats_calculator.validate_cluster_statistics(
            cluster_mu,
            cluster_Sigma
        )

        if not is_valid:
            raise ValueError(f"Invalid cluster statistics: {error_msg}")

        # Step 2: Formulate cluster-level QUBO
        cluster_ids = list(clusters.keys())
        cluster_bqm = self.cluster_formulator.formulate_cluster_allocation(
            cluster_ids,
            cluster_mu,
            cluster_Sigma,
            budget=1.0
        )

        # Step 3: Solve cluster allocation QUBO
        cluster_solution = solver.solve_single_bqm(cluster_bqm)

        if 'error' in cluster_solution:
            raise ValueError(f"Cluster allocation solve failed: {cluster_solution['error']}")

        # Step 4: Decode cluster allocation
        cluster_allocation = self.cluster_formulator.decode_cluster_allocation(
            cluster_solution['solution'],
            cluster_ids
        )

        # Normalize cluster allocation to sum to 1.0
        cluster_sum = cluster_allocation.sum()
        if cluster_sum > 1e-10:
            cluster_allocation = cluster_allocation / cluster_sum
        else:
            # Fallback: equal allocation
            cluster_allocation = pd.Series(
                1.0 / len(cluster_ids),
                index=cluster_ids
            )

        # Step 5: Combine Pass 1 and Pass 2
        final_weights = self.compute_final_weights(
            cluster_weights_pass1,
            cluster_allocation,
            clusters
        )

        # Collect metadata
        metadata = {
            'cluster_mu': cluster_mu.to_dict(),
            'cluster_volatility': np.sqrt(np.diag(cluster_Sigma)),
            'cluster_allocation': cluster_allocation.to_dict(),
            'cluster_sharpe': (cluster_mu / np.sqrt(np.diag(cluster_Sigma))).to_dict(),
            'inter_cluster_correlation': cluster_Sigma.to_dict(),
            'pass2_energy': cluster_solution['energy'],
            'pass2_runtime': cluster_solution['runtime']
        }

        return final_weights, metadata
