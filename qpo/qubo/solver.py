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

"""Quantum solver for QUBO problems using unified multi-cluster formulation."""

import time
import logging
from typing import Dict, Any, List
import dimod


class ParallelQuantumSolver:
    """Unified multi-cluster quantum solver."""

    def __init__(self,
                 solver_type: str = 'simulated',
                 num_reads: int = 10,
                 annealing_time: int = 20,
                 max_workers: int = 5,
                 use_fixed_templates: bool = True,
                 n_levels: int = 6):
        """
        Initialize unified multi-cluster solver.

        Args:
            solver_type: 'simulated', 'qpu', or 'hybrid'
            num_reads: Number of samples per solve
            annealing_time: Annealing time in microseconds
            max_workers: Unused (kept for compatibility)
            use_fixed_templates: Use pre-computed embeddings for QPU
            n_levels: Number of discrete weight levels
        """
        self.solver_type = solver_type
        self.num_reads = num_reads
        self.annealing_time = annealing_time
        self.max_workers = max_workers
        self.use_fixed_templates = use_fixed_templates
        self.n_levels = n_levels
        self.logger = logging.getLogger(__name__)

    def solve_unified_clusters(self,
                              unified_bqm: dimod.BinaryQuadraticModel,
                              clusters: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Solve unified multi-cluster BQM in a single quantum/SA call.

        Args:
            unified_bqm: BQM with all clusters and cluster representatives
            clusters: {cluster_id: [tickers]}

        Returns:
            {cluster_id: result_dict} decoded from unified solution
        """
        num_clusters = len(clusters)
        assets_per_cluster = {cid: len(tickers) for cid, tickers in clusters.items()}
        max_assets = max(assets_per_cluster.values())

        # Solve unified BQM once (both QPU and SA use same template structure)
        if self.solver_type == 'qpu':
            result = self._solve_unified_qpu(unified_bqm, num_clusters, max_assets, clusters)
        else:
            result = self._solve_unified_sa(unified_bqm, num_clusters, max_assets, clusters)

        # Decode unified solution into per-cluster results
        solution = result['solution']
        results = {}

        for cluster_id, tickers in clusters.items():
            # Extract this cluster's variables from unified solution
            cluster_solution = {
                var: val for var, val in solution.items()
                if any(var.startswith(f"{ticker}_") for ticker in tickers)
            }

            results[cluster_id] = {
                'solution': cluster_solution,
                'energy': result['energy'],
                'info': result['info'],
                'runtime': result['runtime'] / num_clusters,  # Amortize
                'sampleset': result['sampleset'],
                'cluster_id': cluster_id
            }

        return results

    def _relabel_bqm_for_template(self,
                                   unified_bqm: dimod.BinaryQuadraticModel,
                                   clusters: Dict[str, List[str]]) -> tuple:
        """
        Relabel BQM variables from ticker-based names to template format.

        BQM uses: TICKER_LEVEL (e.g., "AAPL_0") + CLUSTER_{cluster_id}_{level}
        Template uses: ASSET_{cluster_idx}_{asset_idx}_{level} + META_{cluster_idx}_{level}

        Args:
            unified_bqm: BQM with ticker-based variable names
            clusters: {cluster_id: [tickers]}

        Returns:
            (relabeled_bqm, reverse_relabeling)
        """
        # Build mapping: ticker → (cluster_idx, asset_idx_within_cluster)
        # and cluster_id → cluster_idx
        ticker_to_location = {}
        cluster_id_to_idx = {}
        for cluster_idx, (cluster_id, tickers) in enumerate(clusters.items()):
            cluster_id_to_idx[cluster_id] = cluster_idx
            for asset_idx, ticker in enumerate(tickers):
                ticker_to_location[ticker] = (cluster_idx, asset_idx)

        # Relabel BQM variables
        relabeling = {}
        for var in unified_bqm.variables:
            if var.startswith('CLUSTER_'):
                # Cluster representative: CLUSTER_{cluster_id}_{level} → META_{cluster_idx}_{level}
                # Parse: CLUSTER_cluster_0_2 → META_0_2
                parts = var.split('_')
                if len(parts) >= 3:
                    # Format: CLUSTER_cluster_X_Y or CLUSTER_{cluster_id}_Y
                    cluster_id = '_'.join(parts[1:-1])  # Everything between CLUSTER_ and _level
                    level = parts[-1]

                    if cluster_id in cluster_id_to_idx:
                        cluster_idx = cluster_id_to_idx[cluster_id]
                        template_var = f"META_{cluster_idx}_{level}"
                        relabeling[var] = template_var
                    else:
                        relabeling[var] = var  # Keep as is if not found
                else:
                    relabeling[var] = var
            else:
                # Asset variable: TICKER_LEVEL → ASSET_C_A_L
                parts = var.rsplit('_', 1)
                if len(parts) == 2:
                    ticker, level = parts[0], parts[1]
                    if ticker in ticker_to_location:
                        cluster_idx, asset_idx = ticker_to_location[ticker]
                        template_var = f"ASSET_{cluster_idx}_{asset_idx}_{level}"
                        relabeling[var] = template_var
                    else:
                        relabeling[var] = var  # Keep as is if not found
                else:
                    relabeling[var] = var

        # Create relabeled BQM
        relabeled_bqm = unified_bqm.relabel_variables(relabeling, inplace=False)

        # Create reverse mapping for decoding solution
        reverse_relabeling = {v: k for k, v in relabeling.items()}

        return relabeled_bqm, reverse_relabeling

    def _solve_unified_qpu(self, unified_bqm: dimod.BinaryQuadraticModel,
                          num_clusters: int, assets_per_cluster: int,
                          clusters: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Solve unified BQM on QPU using multi-cluster template."""
        from qpo.qubo.fixed_embeddings import load_template_embedding, create_fixed_embedding_sampler_from_template
        from dwave.system import DWaveSampler
        import dwave.embedding

        start_time = time.time()

        # Load multi-cluster template
        shape = {
            'clusters': num_clusters,
            'assets_per_cluster': assets_per_cluster,
            'n_levels': self.n_levels
        }

        template_embedding, template_meta = load_template_embedding('discrete_levels', shape, 'zephyr')

        if template_embedding is None:
            raise RuntimeError(
                f"No multi-cluster template found for {num_clusters} clusters × "
                f"{assets_per_cluster} assets × {self.n_levels} levels. "
                f"Generate using tools/generate_portfolio_embedding_template.py"
            )

        # Relabel BQM from ticker-based names to template format
        relabeled_bqm, reverse_relabeling = self._relabel_bqm_for_template(unified_bqm, clusters)

        self.logger.info(f"Relabeled {len(unified_bqm.variables)} variables for template compatibility")

        # Filter template to variables in relabeled BQM
        relabeled_vars = set(relabeled_bqm.variables)
        filtered_embedding = {var: qubits for var, qubits in template_embedding.items() if var in relabeled_vars}

        # Check for unmapped cluster representative variables
        unmapped_cluster_reps = {v for v in relabeled_vars if v.startswith('CLUSTER_')}
        if len(unmapped_cluster_reps) > 0:
            self.logger.error(f"Failed to map {len(unmapped_cluster_reps)} cluster representative variables: {list(unmapped_cluster_reps)[:5]}")

        # Verify META variables are present
        meta_vars = {v for v in relabeled_vars if v.startswith('META_')}
        if len(meta_vars) > 0:
            self.logger.info(f"Mapped {len(meta_vars)} cluster representative variables to META_ format")

        if len(filtered_embedding) < len(relabeled_vars):
            missing = len(relabeled_vars) - len(filtered_embedding)
            raise RuntimeError(f"Template missing {missing} variables after relabeling - cannot proceed")

        # Create sampler
        base_sampler = DWaveSampler()
        sampler = create_fixed_embedding_sampler_from_template(base_sampler, filtered_embedding)

        # Use relabeled BQM for solving
        bqm_to_solve = relabeled_bqm

        # Compute chain strength
        chain_strength = dwave.embedding.chain_strength.uniform_torque_compensation(relabeled_bqm)

        self.logger.info(f"Submitting unified multi-cluster BQM to QPU: {num_clusters} clusters, "
                        f"{len(unified_bqm.variables)} vars, {self.num_reads} reads")

        # Sample
        sampleset = sampler.sample(
            bqm_to_solve,
            num_reads=self.num_reads,
            annealing_time=self.annealing_time,
            label=f'Portfolio-MultiCluster-{num_clusters}c',
            chain_strength=chain_strength
        )

        runtime = time.time() - start_time

        # Check chain breaks
        if hasattr(sampleset.first, 'chain_break_fraction'):
            cbf = sampleset.first.chain_break_fraction
            if cbf > 0.1:
                self.logger.warning(f"High chain break fraction: {cbf:.2%}")
            elif cbf > 0:
                self.logger.info(f"Chain break fraction: {cbf:.2%}")

        # Reverse relabeling to get back to original ticker names
        original_solution = {
            reverse_relabeling.get(var, var): val
            for var, val in sampleset.first.sample.items()
        }

        return {
            'solution': original_solution,
            'energy': sampleset.first.energy,
            'info': sampleset.info,
            'runtime': runtime,
            'sampleset': sampleset
        }

    def _solve_unified_sa(self, unified_bqm: dimod.BinaryQuadraticModel,
                         num_clusters: int, assets_per_cluster: int,
                         clusters: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Solve unified BQM using simulated annealing.

        SA solves the same unified BQM as QPU (same formulation with cluster representatives),
        but doesn't need physical embedding since it runs in software.
        """
        from neal import SimulatedAnnealingSampler
        from qpo.qubo.fixed_embeddings import load_template_embedding

        start_time = time.time()

        # Validate template exists (for structural compatibility checking)
        # NOTE: This is intentional for SA mode - it ensures we catch bugs where:
        # 1. Clustering produces shapes incompatible with QPU templates
        # 2. BQM structure doesn't match expected template format
        # 3. Variable naming conventions are incorrect
        # This validation prevents SA from "working" on problems that would fail on QPU,
        # ensuring fair apples-to-apples comparison and catching embedding issues early.
        shape = {
            'clusters': num_clusters,
            'assets_per_cluster': assets_per_cluster,
            'n_levels': self.n_levels
        }

        template_embedding, template_meta = load_template_embedding('discrete_levels', shape, 'zephyr')

        if template_embedding is None:
            raise RuntimeError(
                f"No multi-cluster template found for {num_clusters} clusters × "
                f"{assets_per_cluster} assets × {self.n_levels} levels. "
                f"Generate using tools/generate_portfolio_embedding_template.py"
            )

        self.logger.info(f"Validated template compatibility: {num_clusters} clusters, "
                        f"{assets_per_cluster} assets/cluster, {self.n_levels} levels")

        # Relabel BQM from ticker-based names to template format
        relabeled_bqm, reverse_relabeling = self._relabel_bqm_for_template(unified_bqm, clusters)

        self.logger.info(f"Relabeled {len(unified_bqm.variables)} variables for template compatibility")

        # Filter template to variables in relabeled BQM
        relabeled_vars = set(relabeled_bqm.variables)
        filtered_embedding = {var: qubits for var, qubits in template_embedding.items() if var in relabeled_vars}

        # Check for unmapped cluster representative variables
        unmapped_cluster_reps = {v for v in relabeled_vars if v.startswith('CLUSTER_')}
        if len(unmapped_cluster_reps) > 0:
            self.logger.error(f"Failed to map {len(unmapped_cluster_reps)} cluster representative variables: {list(unmapped_cluster_reps)[:5]}")

        # Verify META variables are present
        meta_vars = {v for v in relabeled_vars if v.startswith('META_')}
        if len(meta_vars) > 0:
            self.logger.info(f"Mapped {len(meta_vars)} cluster representative variables to META_ format")

        if len(filtered_embedding) < len(relabeled_vars):
            missing = len(relabeled_vars) - len(filtered_embedding)
            raise RuntimeError(f"Template missing {missing} variables after relabeling - cannot proceed")

        # Use relabeled BQM for solving
        bqm_to_solve = relabeled_bqm

        # Log submission info (match QPU logging)
        self.logger.info(f"Submitting unified multi-cluster BQM to SA: {num_clusters} clusters, "
                        f"{len(unified_bqm.variables)} vars, {self.num_reads} reads")

        # SA solves the unified BQM directly (no embedding needed)
        sampler = SimulatedAnnealingSampler()

        # Make num_sweeps derived from annealing_time for parity with QPU
        num_sweeps = max(256, self.annealing_time * 10)

        sampleset = sampler.sample(
            bqm_to_solve,
            num_reads=self.num_reads,
            num_sweeps=num_sweeps
        )

        runtime = time.time() - start_time

        # Post-solve diagnostics
        best_energy = sampleset.first.energy
        self.logger.info(f"SA completed in {runtime:.2f}s, best energy: {best_energy:.6f}")

        # Reverse relabeling to get back to original ticker names
        original_solution = {
            reverse_relabeling.get(var, var): val
            for var, val in sampleset.first.sample.items()
        }

        return {
            'solution': original_solution,
            'energy': sampleset.first.energy,
            'info': sampleset.info,
            'runtime': runtime,
            'sampleset': sampleset
        }
