"""Quantum solver for QUBO problems."""

import time
from typing import Dict, Any
import dimod


class QuantumSolver:
    """Execute QUBO on quantum hardware or simulators."""

    def __init__(self,
                 solver_type: str = 'simulated',
                 num_reads: int = 1000,
                 annealing_time: int = 20):
        """
        Initialize quantum solver.

        Args:
            solver_type: 'simulated', 'qpu', or 'hybrid'
            num_reads: Number of QPU samples (default 1000)
            annealing_time: Annealing time in microseconds (default 20)
        """
        self.solver_type = solver_type
        self.num_reads = num_reads
        self.annealing_time = annealing_time

    def solve(self, bqm: dimod.BinaryQuadraticModel, label: str = 'Portfolio') -> Dict[str, Any]:
        """
        Solve QUBO on quantum hardware or simulator.

        Args:
            bqm: Binary quadratic model
            label: Job label for tracking

        Returns:
            {
                'solution': dict,      # Binary variable assignments
                'energy': float,       # Objective value
                'info': dict,          # Solver metadata
                'runtime': float,      # Seconds
                'sampleset': SampleSet # Full results
            }
        """
        start_time = time.time()

        if self.solver_type == 'qpu':
            sampleset = self._solve_qpu(bqm, label)

        elif self.solver_type == 'hybrid':
            sampleset = self._solve_hybrid(bqm, label)

        elif self.solver_type == 'simulated':
            sampleset = self._solve_simulated(bqm)

        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}. "
                           f"Choose from: 'simulated', 'qpu', 'hybrid'")

        runtime = time.time() - start_time

        # Get best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy

        return {
            'solution': best_sample,
            'energy': best_energy,
            'info': sampleset.info,
            'runtime': runtime,
            'sampleset': sampleset
        }

    def _solve_qpu(self, bqm: dimod.BinaryQuadraticModel, label: str) -> Any:
        """
        Solve on D-Wave QPU.

        Requires:
            - D-Wave Ocean SDK installed
            - Valid API token configured
        """
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError:
            raise ImportError(
                "D-Wave Ocean SDK not installed. "
                "Install with: pip install dwave-ocean-sdk"
            )

        sampler = EmbeddingComposite(DWaveSampler())

        sampleset = sampler.sample(
            bqm,
            num_reads=self.num_reads,
            annealing_time=self.annealing_time,
            label=label,
            chain_strength='auto'  # Auto-tune chain strength
        )

        return sampleset

    def _solve_hybrid(self, bqm: dimod.BinaryQuadraticModel, label: str) -> Any:
        """
        Solve using D-Wave Leap Hybrid Solver.

        Good for larger problems (>200 variables).
        """
        try:
            from dwave.system import LeapHybridSampler
        except ImportError:
            raise ImportError(
                "D-Wave Ocean SDK not installed. "
                "Install with: pip install dwave-ocean-sdk"
            )

        sampler = LeapHybridSampler()

        sampleset = sampler.sample(
            bqm,
            label=label
        )

        return sampleset

    def _solve_simulated(self, bqm: dimod.BinaryQuadraticModel) -> Any:
        """
        Solve using simulated annealing (classical simulation).

        No QPU access required. Good for testing.
        """
        try:
            from neal import SimulatedAnnealingSampler
        except ImportError:
            raise ImportError(
                "neal not installed. "
                "Install with: pip install dwave-neal"
            )

        sampler = SimulatedAnnealingSampler()

        sampleset = sampler.sample(
            bqm,
            num_reads=self.num_reads
        )

        return sampleset


class ParallelQuantumSolver:
    """Solve multiple clusters in parallel."""

    def __init__(self,
                 solver_type: str = 'simulated',
                 num_reads: int = 1000,
                 annealing_time: int = 20,
                 max_workers: int = 5):
        """
        Initialize parallel solver.

        Args:
            solver_type: 'simulated', 'qpu', or 'hybrid'
            num_reads: Number of QPU samples
            annealing_time: Annealing time in microseconds
            max_workers: Max concurrent QPU jobs (default 5)
        """
        self.solver = QuantumSolver(solver_type, num_reads, annealing_time)
        self.max_workers = max_workers

    def solve_cluster(self,
                     bqm: dimod.BinaryQuadraticModel,
                     cluster_id: str) -> Dict[str, Any]:
        """
        Solve a single cluster QUBO.

        Args:
            bqm: Binary quadratic model for this cluster
            cluster_id: Cluster identifier

        Returns:
            Result dictionary with cluster_id added
        """
        result = self.solver.solve(bqm, label=f'Portfolio-{cluster_id}')
        result['cluster_id'] = cluster_id
        return result

    def solve_single_bqm(self, bqm: dimod.BinaryQuadraticModel, label: str = 'QUBO') -> Dict[str, Any]:
        """
        Solve a single BQM (used for cluster-level optimization in Pass 2).

        Args:
            bqm: Binary quadratic model
            label: Problem label

        Returns:
            Result dictionary
        """
        return self.solver.solve(bqm, label=label)

    def solve_all_clusters(self,
                          bqms: Dict[str, dimod.BinaryQuadraticModel]) -> Dict[str, Dict[str, Any]]:
        """
        Solve all clusters in parallel.

        Args:
            bqms: {cluster_id: BQM}

        Returns:
            {cluster_id: result_dict}
        """
        results = {}

        # For simulated annealing, we can use ThreadPoolExecutor
        # For QPU, Leap handles queueing automatically
        if self.solver.solver_type == 'simulated':
            results = self._solve_parallel_threaded(bqms)
        else:
            # For QPU/hybrid, solve sequentially (Leap manages queue)
            results = self._solve_sequential(bqms)

        return results

    def _solve_parallel_threaded(self,
                                bqms: Dict[str, dimod.BinaryQuadraticModel]) -> Dict[str, Dict[str, Any]]:
        """Solve clusters in parallel using threads (for simulated only)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(self.solve_cluster, bqm, cid): cid
                for cid, bqm in bqms.items()
            }

            # Collect results as they complete
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    result = future.result()
                    results[cid] = result
                except Exception as e:
                    print(f"Cluster {cid} failed: {e}")
                    results[cid] = {'error': str(e), 'cluster_id': cid}

        return results

    def _solve_sequential(self,
                         bqms: Dict[str, dimod.BinaryQuadraticModel]) -> Dict[str, Dict[str, Any]]:
        """Solve clusters sequentially (for QPU/hybrid)."""
        results = {}

        for cid, bqm in bqms.items():
            try:
                result = self.solve_cluster(bqm, cid)
                results[cid] = result
            except Exception as e:
                print(f"Cluster {cid} failed: {e}")
                results[cid] = {'error': str(e), 'cluster_id': cid}

        return results
