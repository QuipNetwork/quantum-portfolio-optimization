"""Quantum portfolio optimizer using Independent Clusters Approach."""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from clustering import CorrelationClusterer
from qpo.qubo.formulation import QUBOFormulator
from qpo.qubo.solver import ParallelQuantumSolver
from qpo.qubo.decoder import QUBODecoder
from qpo.qubo.aggregation import ClusterAggregator


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
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 lambda_budget: float = 10.0,
                 solver_type: str = 'simulated',
                 num_reads: int = 1000,
                 annealing_time: int = 20,
                 aggregation_strategy: str = 'concatenate',
                 cardinality: Optional[int] = None,
                 clusterer: Optional[Any] = None):
        """
        Initialize quantum optimizer.

        Args:
            max_cluster_size: Maximum assets per cluster (default 18)
            n_bits: Binary discretization bits (default 10)
            alpha: Return coefficient (higher = more aggressive)
            beta: Risk coefficient (higher = more conservative)
            lambda_budget: Budget constraint penalty (default 10.0)
            solver_type: 'simulated', 'qpu', or 'hybrid'
            num_reads: Number of QPU samples (default 1000)
            annealing_time: Annealing time in microseconds (default 20)
            aggregation_strategy: 'concatenate', 'proportional', or 'uniform'
            cardinality: Maximum non-zero assets (optional)
            clusterer: Custom clusterer instance (default: CorrelationClusterer)
        """
        self.max_cluster_size = max_cluster_size
        self.n_bits = n_bits
        self.alpha = alpha
        self.beta = beta
        self.lambda_budget = lambda_budget
        self.solver_type = solver_type
        self.num_reads = num_reads
        self.annealing_time = annealing_time
        self.aggregation_strategy = aggregation_strategy
        self.cardinality = cardinality

        # Initialize components
        self.clusterer = clusterer or CorrelationClusterer(max_cluster_size=max_cluster_size)
        self.formulator = QUBOFormulator(n_bits, alpha, beta, lambda_budget)
        self.solver = ParallelQuantumSolver(solver_type, num_reads, annealing_time)
        self.decoder = QUBODecoder(n_bits)
        self.aggregator = ClusterAggregator(aggregation_strategy)

    def optimize(self, returns: pd.DataFrame) -> QuantumOptimizationResult:
        """
        Optimize portfolio using quantum annealing.

        Args:
            returns: Daily returns DataFrame (days × assets)

        Returns:
            QuantumOptimizationResult with portfolio weights and metrics
        """
        start_time = time.time()

        try:
            # Stage 1: Clustering
            clusters = self._cluster_assets(returns)

            # Stage 2: QUBO Formulation
            mu, Sigma = self._compute_statistics(returns)
            bqms = self._formulate_qubos(clusters, mu, Sigma)

            # Stage 3: Quantum Solve
            solutions = self._solve_qubos(bqms)

            # Check for solver errors
            failed = {cid: res for cid, res in solutions.items() if 'error' in res}
            if failed:
                error_msg = f"Solver failed for {len(failed)}/{len(clusters)} clusters"
                return self._create_error_result(error_msg, time.time() - start_time)

            # Stage 4: Decode Solutions
            cluster_weights = self._decode_solutions(solutions, clusters)

            # Stage 5: Aggregate to Global Portfolio
            global_weights = self._aggregate_clusters(cluster_weights, clusters)

            # Calculate metrics
            metrics = self._calculate_metrics(global_weights, mu, Sigma)

            # Prepare result
            cluster_stats = self._get_cluster_statistics(clusters, cluster_weights)
            solver_info = self._extract_solver_info(solutions)

            runtime = time.time() - start_time

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
        """Aggregate cluster solutions to global portfolio."""
        return self.aggregator.aggregate(
            cluster_weights,
            clusters,
            target_budget=1.0,
            cardinality=self.cardinality
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
            'clusterer_type': type(self.clusterer).__name__
        }
