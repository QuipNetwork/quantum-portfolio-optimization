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

"""Quantum portfolio optimizer using Independent Clusters Approach."""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass

from clustering import CorrelationClusterer, VolatilityClusterer
from qpo.qubo.formulation import QUBOFormulator
from qpo.qubo.solver import ParallelQuantumSolver
from qpo.qubo.decoder import QUBODecoder
from qpo.qubo.aggregation import ClusterAggregator
from qpo.qubo.hierarchical import TwoPassHierarchicalOptimizer


@dataclass
class QuantumOptimizationResult:
    """Result from quantum portfolio optimization."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    n_clusters: int
    cluster_stats: Dict[str, Any]
    runtime: float
    solver_info: Dict[str, Any]
    success: bool
    message: str


class IndependentClustersOptimizer:
    """
    Portfolio optimizer using Independent Clusters Approach with quantum annealing.

    Pipeline:
        1. Cluster assets into groups (≤18 assets per cluster)
        2. Formulate QUBO for each cluster
        3. Solve each cluster on QPU/simulator in parallel
        4. Decode binary solutions to weights
        5. Aggregate cluster weights into global portfolio
    """

    def __init__(self,
                 max_cluster_size: int = 24,
                 n_bits: int = 10,
                 alpha: float = 1.5,
                 beta: float = 0.8,
                 lambda_budget: float = 5.0,
                 solver_type: str = 'simulated',
                 num_reads: int = 2000,
                 intra_cluster_num_reads: Optional[int] = None,
                 annealing_time: int = 20,
                 aggregation_strategy: str = 'proportional',
                 cardinality: Optional[int] = None,
                 clusterer: Optional[Any] = None,
                 use_two_pass: bool = True,
                 inter_cluster_alpha: Optional[float] = None,
                 inter_cluster_beta: Optional[float] = None,
                 inter_cluster_num_reads: Optional[int] = None):
        """
        Initialize quantum optimizer.

        Args:
            max_cluster_size: Maximum assets per cluster (default 24, tuned for MV baseline matching)
            n_bits: Binary discretization bits (default 10)
            alpha: Return coefficient for Pass 1 (default 1.5, tuned for MV baseline matching)
            beta: Risk coefficient for Pass 1 (default 0.8, tuned for MV baseline matching)
            lambda_budget: Budget constraint penalty (default 5.0, reduced to lessen dominance)
            solver_type: 'simulated', 'qpu', or 'hybrid'
            num_reads: Number of QPU samples (default 2000, DEPRECATED - use intra_cluster_num_reads)
            intra_cluster_num_reads: Number of QPU samples for Pass 1 intra-cluster optimization (default: None, uses num_reads or 128)
            annealing_time: Annealing time in microseconds (default 20)
            aggregation_strategy: 'proportional', 'concatenate', or 'uniform' (ignored if use_two_pass=True)
            cardinality: Maximum non-zero assets (optional)
            clusterer: Custom clusterer instance (default: VolatilityClusterer, best performer in benchmarks)
            use_two_pass: Enable two-pass hierarchical optimization (default True, addresses inter-cluster covariances)
            inter_cluster_alpha: Return coefficient for Pass 2 (default: alpha * 1.15, more aggressive)
            inter_cluster_beta: Risk coefficient for Pass 2 (default: beta * 0.85, less risk-averse)
            inter_cluster_num_reads: Number of QPU samples for Pass 2 inter-cluster optimization (default: max(256, intra*2))
        """
        self.max_cluster_size = max_cluster_size
        self.n_bits = n_bits
        self.alpha = alpha
        self.beta = beta
        self.lambda_budget = lambda_budget
        self.solver_type = solver_type
        self.num_reads = num_reads  # Backward compatibility
        self.annealing_time = annealing_time
        self.aggregation_strategy = aggregation_strategy
        self.cardinality = cardinality
        self.use_two_pass = use_two_pass

        # Intra-cluster num_reads (Pass 1): Default 128 based on diminishing returns analysis
        # If not specified, check if legacy num_reads was provided, otherwise default to 128
        if intra_cluster_num_reads is not None:
            self.intra_cluster_num_reads = intra_cluster_num_reads
        elif num_reads != 2000:  # User explicitly set num_reads (not default)
            self.intra_cluster_num_reads = num_reads
        else:  # Use optimized default
            self.intra_cluster_num_reads = 128

        # Two-pass parameters (more aggressive for inter-cluster to recover diversification)
        # Default: inter_alpha slightly higher than alpha, inter_beta slightly lower than beta
        self.inter_cluster_alpha = inter_cluster_alpha if inter_cluster_alpha is not None else min(2.0, alpha * 1.15)
        self.inter_cluster_beta = inter_cluster_beta if inter_cluster_beta is not None else max(0.5, beta * 0.85)

        # Inter-cluster num_reads (Pass 2): Default 2x intra, minimum 256
        # Pass 2 is single critical optimization → needs higher quality than parallelized Pass 1
        self.inter_cluster_num_reads = inter_cluster_num_reads if inter_cluster_num_reads is not None else max(256, self.intra_cluster_num_reads * 2)

        # Initialize components
        # Use VolatilityClusterer by default (best Sharpe 1.364 in benchmarks)
        # Mixes high/low-vol assets within clusters for better intra-cluster diversification
        self.clusterer = clusterer or VolatilityClusterer(max_cluster_size=max_cluster_size)
        self.formulator = QUBOFormulator(n_bits, alpha, beta, lambda_budget)
        self.solver = ParallelQuantumSolver(solver_type, self.intra_cluster_num_reads, annealing_time)
        self.decoder = QUBODecoder(n_bits)
        self.aggregator = ClusterAggregator(aggregation_strategy)

        # Initialize two-pass optimizer if enabled
        if use_two_pass:
            intra_params = {
                'n_bits': n_bits,
                'alpha': alpha,
                'beta': beta,
                'lambda_budget': lambda_budget,
                'num_reads': self.intra_cluster_num_reads
            }
            inter_params = {
                'n_bits': n_bits,
                'alpha': self.inter_cluster_alpha,
                'beta': self.inter_cluster_beta,
                'lambda_budget': lambda_budget,
                'num_reads': self.inter_cluster_num_reads
            }
            self.hierarchical_optimizer = TwoPassHierarchicalOptimizer(intra_params, inter_params)
        else:
            self.hierarchical_optimizer = None

    def optimize(self, returns: pd.DataFrame, return_dict: bool = False) -> Union[QuantumOptimizationResult, Dict[str, Any]]:
        """
        Optimize portfolio using quantum annealing.

        Args:
            returns: Daily returns DataFrame (days × assets)
            return_dict: If True, return dict format (Backtester compatible).
                        If False, return QuantumOptimizationResult dataclass.

        Returns:
            QuantumOptimizationResult or dict with keys: weights, metrics, runtime
        """
        start_time = time.time()
        clustering_time = 0.0
        solver_time = 0.0

        try:
            # Stage 1: Clustering (track time separately)
            clustering_start = time.time()
            clusters = self._cluster_assets(returns)
            clustering_time = time.time() - clustering_start

            # Stage 2: QUBO Formulation
            mu, Sigma = self._compute_statistics(returns)

            # Start solver timing (excludes clustering)
            solver_start = time.time()
            bqms = self._formulate_qubos(clusters, mu, Sigma)

            # Stage 3: Quantum Solve
            solutions = self._solve_qubos(bqms)

            # Check for solver errors
            failed = {cid: res for cid, res in solutions.items() if 'error' in res}
            if failed:
                error_msg = f"Solver failed for {len(failed)}/{len(clusters)} clusters"
                if return_dict:
                    return {
                        'weights': pd.Series(dtype=float),
                        'metrics': {
                            'sharpe_ratio': 0.0,
                            'expected_return': 0.0,
                            'volatility': 0.0,
                            'solver_only_runtime': time.time() - solver_start,
                            'error': error_msg
                        },
                        'runtime': time.time() - start_time
                    }
                return self._create_error_result(error_msg, time.time() - start_time)

            # Stage 4: Decode Solutions
            cluster_weights = self._decode_solutions(solutions, clusters)

            # Stage 5: Aggregate to Global Portfolio
            if self.use_two_pass:
                # Two-pass hierarchical optimization
                global_weights, pass2_metadata = self._aggregate_clusters_two_pass(
                    cluster_weights, clusters, mu, Sigma
                )
            else:
                # Original single-pass aggregation
                global_weights = self._aggregate_clusters(cluster_weights, clusters)
                pass2_metadata = None

            solver_time = time.time() - solver_start

            # Calculate metrics
            metrics = self._calculate_metrics(global_weights, mu, Sigma)

            # Total runtime
            runtime = time.time() - start_time

            # Return dict format for Backtester
            if return_dict:
                return {
                    'weights': global_weights,
                    'metrics': {
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'expected_return': metrics['expected_return'],
                        'volatility': metrics['volatility'],
                        'solver_only_runtime': solver_time,
                        'clustering_time': clustering_time,
                        'n_clusters': len(clusters),
                        'solver_type': self.solver_type
                    },
                    'runtime': runtime
                }

            # Return dataclass format
            cluster_stats = self._get_cluster_statistics(clusters, cluster_weights)
            solver_info = self._extract_solver_info(solutions)
            solver_info['solver_only_runtime'] = solver_time
            solver_info['clustering_time'] = clustering_time

            # Add two-pass metadata if applicable
            if pass2_metadata is not None:
                solver_info['pass2_metadata'] = pass2_metadata
                solver_info['optimization_mode'] = 'two_pass_hierarchical'
            else:
                solver_info['optimization_mode'] = 'single_pass'

            return QuantumOptimizationResult(
                weights=global_weights,
                expected_return=metrics['expected_return'],
                volatility=metrics['volatility'],
                sharpe_ratio=metrics['sharpe_ratio'],
                n_clusters=len(clusters),
                cluster_stats=cluster_stats,
                runtime=runtime,
                solver_info=solver_info,
                success=True,
                message="Optimization completed successfully"
            )

        except Exception as e:
            runtime = time.time() - start_time
            if return_dict:
                return {
                    'weights': pd.Series(dtype=float),
                    'metrics': {'sharpe_ratio': 0.0, 'expected_return': 0.0, 'volatility': 0.0},
                    'runtime': runtime,
                    'error': str(e)
                }
            return self._create_error_result(str(e), runtime)

    def _cluster_assets(self, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """Cluster assets using configured clusterer."""
        return self.clusterer.cluster(returns)

    def _compute_statistics(self, returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Compute expected returns and covariance matrix (annualized)."""
        mu = returns.mean() * 252  # Annualized return
        Sigma = returns.cov() * 252  # Annualized covariance
        return mu, Sigma

    def _formulate_qubos(self,
                        clusters: Dict[str, List[str]],
                        mu: pd.Series,
                        Sigma: pd.DataFrame) -> Dict[str, Any]:
        """Formulate QUBO for each cluster."""
        bqms = {}

        for cluster_id, tickers in clusters.items():
            # Extract cluster statistics
            cluster_mu = mu[tickers]
            cluster_Sigma = Sigma.loc[tickers, tickers]

            # Formulate QUBO
            bqm = self.formulator.formulate_cluster(tickers, cluster_mu, cluster_Sigma)
            bqms[cluster_id] = bqm

        return bqms

    def _solve_qubos(self, bqms: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Solve all QUBOs on quantum hardware/simulator."""
        return self.solver.solve_all_clusters(bqms)

    def _decode_solutions(self,
                         solutions: Dict[str, Dict[str, Any]],
                         clusters: Dict[str, List[str]]) -> Dict[str, pd.Series]:
        """Decode binary solutions to weights."""
        cluster_weights = {}

        for cluster_id, tickers in clusters.items():
            solution = solutions[cluster_id]['solution']
            weights = self.decoder.decode_cluster_solution(solution, tickers)
            cluster_weights[cluster_id] = weights

        return cluster_weights

    def _aggregate_clusters(self,
                           cluster_weights: Dict[str, pd.Series],
                           clusters: Dict[str, List[str]]) -> pd.Series:
        """Aggregate cluster solutions to global portfolio (single-pass)."""
        return self.aggregator.aggregate(
            cluster_weights,
            clusters,
            target_budget=1.0,
            cardinality=self.cardinality
        )

    def _aggregate_clusters_two_pass(self,
                                     cluster_weights: Dict[str, pd.Series],
                                     clusters: Dict[str, List[str]],
                                     mu: pd.Series,
                                     Sigma: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Aggregate cluster solutions using two-pass hierarchical optimization.

        Pass 2: Formulates and solves a cluster-level QUBO to determine
        optimal allocation across clusters, considering inter-cluster correlations.

        Args:
            cluster_weights: Pass 1 results {cluster_id: asset_weights}
            clusters: {cluster_id: [tickers]}
            mu: Global expected returns
            Sigma: Global covariance matrix

        Returns:
            (global_weights, pass2_metadata)
        """
        return self.hierarchical_optimizer.optimize_hierarchically(
            cluster_weights,
            clusters,
            mu,
            Sigma,
            self.solver
        )

    def _calculate_metrics(self,
                          weights: pd.Series,
                          mu: pd.Series,
                          Sigma: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio metrics."""
        expected_return = (weights * mu).sum()
        variance = weights @ Sigma @ weights
        volatility = np.sqrt(variance)

        sharpe_ratio = expected_return / volatility if volatility > 0 else 0.0

        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def _get_cluster_statistics(self,
                                clusters: Dict[str, List[str]],
                                cluster_weights: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Get statistics about clustering."""
        cluster_sizes = [len(tickers) for tickers in clusters.values()]
        cluster_budgets = [weights.sum() for weights in cluster_weights.values()]

        return {
            'n_clusters': len(clusters),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'total_assets': sum(cluster_sizes),
            'cluster_budgets': cluster_budgets,
            'cluster_sizes': cluster_sizes
        }

    def _extract_solver_info(self, solutions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract solver metadata."""
        energies = [sol['energy'] for sol in solutions.values()]
        runtimes = [sol['runtime'] for sol in solutions.values()]

        return {
            'solver_type': self.solver_type,
            'num_reads': self.num_reads,
            'n_clusters_solved': len(solutions),
            'total_solve_time': sum(runtimes),
            'avg_solve_time': np.mean(runtimes),
            'energies': energies
        }

    def _create_error_result(self, error_msg: str, runtime: float) -> QuantumOptimizationResult:
        """Create error result."""
        return QuantumOptimizationResult(
            weights=pd.Series(),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            n_clusters=0,
            cluster_stats={},
            runtime=runtime,
            solver_info={},
            success=False,
            message=f"Error: {error_msg}"
        )

    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {
            'max_cluster_size': self.max_cluster_size,
            'n_bits': self.n_bits,
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda_budget': self.lambda_budget,
            'solver_type': self.solver_type,
            'num_reads': self.num_reads,
            'annealing_time': self.annealing_time,
            'aggregation_strategy': self.aggregation_strategy,
            'cardinality': self.cardinality,
            'clusterer_type': type(self.clusterer).__name__,
            'use_two_pass': self.use_two_pass,
            'inter_cluster_alpha': self.inter_cluster_alpha if self.use_two_pass else None,
            'inter_cluster_beta': self.inter_cluster_beta if self.use_two_pass else None
        }


class QuantumOptimizerWrapper:
    """
    Wrapper for IndependentClustersOptimizer to work with Backtester.

    Automatically sets return_dict=True and tracks solver-only runtime.
    """

    def __init__(self, optimizer: IndependentClustersOptimizer):
        """
        Initialize wrapper.

        Args:
            optimizer: IndependentClustersOptimizer instance
        """
        self.optimizer = optimizer

    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize portfolio and return dict format for Backtester.

        Args:
            returns: Returns DataFrame

        Returns:
            Dict with keys: weights, metrics, runtime
        """
        return self.optimizer.optimize(returns, return_dict=True)
