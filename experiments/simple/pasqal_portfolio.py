#!/usr/bin/env python3
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

"""
Pasqal neutral-atom portfolio optimizers.

Three solver paths, all running on local emulators:

1. solve_qubo  — qubosolver LocalEmulator on the QUBO penalty matrix.
                 COMPARABLE to cuopt/phi/qhd benchmark rows.

2. solve_mis   — Maximum Independent Set on a pairwise budget-conflict
                 graph, followed by score-aware greedy post-selection.
                 NOT COMPARABLE: only pairwise constraints are encoded
                 in the quantum step; scores and multi-asset budget
                 interactions are post-processed classically.

3. solve_pulser — Hand-built Rydberg adiabatic pulse sequence on a
                  1D atom register encoding the same conflict graph
                  as solve_mis. Bounded by Qutip emulation cost
                  (n <= 12). NOT COMPARABLE for the same reasons as
                  solve_mis plus encoding lossiness from 1/r^6
                  interactions.

See PASQAL_NOTES.md for the client-facing writeup of what each path
actually measures.
"""

from typing import Any, Dict, List

import numpy as np

from simple_portfolio_qubo import (
    SimplePortfolioQUBO,
    get_example_assets,
    get_example_constraints,
)

# Hard cap on raw-Pulser problem size in local Qutip emulation.
# Qutip state-vector cost scales as 2^n; n>12 takes minutes per shot.
_PULSER_MAX_N = 12


class PasqalPortfolioOptimizer:
    """Portfolio optimizer with three Pasqal-flavored solver paths."""

    def __init__(
        self,
        assets: List[Dict[str, Any]],
        budget: float,
        max_duration: float,
        max_cardinality: int,
        lambda_budget: float = 2.0,
        lambda_duration: float = 10.0,
        lambda_cardinality: float = 5.0,
    ):
        """
        Initialize the Pasqal portfolio optimizer.

        Args:
            assets: List of asset dicts with keys 'id', 'price',
                'duration', 'score'.
            budget: Maximum total price (<=).
            max_duration: Maximum total duration (<=).
            max_cardinality: Maximum number of assets (<=).
            lambda_budget: Penalty weight for the budget constraint
                in the QUBO penalty matrix (used by solve_qubo /
                solve_pulser via the internal SimplePortfolioQUBO).
            lambda_duration: Penalty weight for the duration constraint.
            lambda_cardinality: Penalty weight for the cardinality
                constraint.
        """
        self.assets = assets
        self.n = len(assets)
        self.budget = budget
        self.max_duration = max_duration
        self.max_cardinality = max_cardinality

        self.prices = np.array([a['price'] for a in assets])
        self.durations = np.array([a['duration'] for a in assets])
        self.scores = np.array([a['score'] for a in assets])
        self.asset_ids = [a['id'] for a in assets]

        self._qubo_optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=budget,
            max_duration=max_duration,
            max_cardinality=max_cardinality,
            lambda_budget=lambda_budget,
            lambda_duration=lambda_duration,
            lambda_cardinality=lambda_cardinality,
        )

    def check_constraints(self, selection: np.ndarray) -> Dict[str, Any]:
        """Check constraint satisfaction for a binary selection vector."""
        x = np.asarray(selection)
        selected = np.where(x == 1)[0]
        total_price = float(np.sum(self.prices[selected])) if len(selected) else 0.0
        total_duration = (
            float(np.sum(self.durations[selected])) if len(selected) else 0.0
        )
        num_selected = len(selected)
        return {
            'budget_satisfied': total_price <= self.budget,
            'duration_satisfied': total_duration <= self.max_duration,
            'cardinality_satisfied': num_selected <= self.max_cardinality,
            'total_price': total_price,
            'total_duration': total_duration,
            'num_selected': num_selected,
            'budget_limit': self.budget,
            'duration_limit': self.max_duration,
            'cardinality_limit': self.max_cardinality,
        }

    def _is_feasible(self, selection: np.ndarray) -> bool:
        c = self.check_constraints(selection)
        return (
            c['budget_satisfied']
            and c['duration_satisfied']
            and c['cardinality_satisfied']
        )

    def _compute_energy(self, selection: np.ndarray) -> float:
        q = self._qubo_optimizer.build_qubo_matrix()
        x = np.asarray(selection, dtype=float)
        return float(x @ q @ x)

    def _round_and_repair(self, continuous: np.ndarray) -> np.ndarray:
        """Threshold at 0.5, then greedily drop lowest-score selected assets
        until feasible."""
        x = (np.asarray(continuous) > 0.5).astype(int)
        if self._is_feasible(x):
            return x
        selected = np.where(x == 1)[0]
        order = sorted(selected, key=lambda i: self.scores[i])
        for idx in order:
            x[idx] = 0
            if self._is_feasible(x):
                break
        return x

    def _build_result(
        self,
        selection: np.ndarray,
        energy: float,
        is_feasible: bool,
    ) -> Dict[str, Any]:
        selected = np.where(selection == 1)[0]
        if len(selected):
            total_price = float(np.sum(self.prices[selected]))
            total_duration = float(np.sum(self.durations[selected]))
            total_score = float(np.sum(self.scores[selected]))
        else:
            total_price = 0.0
            total_duration = 0.0
            total_score = 0.0
        return {
            'selection': selection,
            'selected_assets': [self.asset_ids[i] for i in selected],
            'energy': energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected),
            'is_feasible': is_feasible,
        }

    def solve_qubo(
        self,
        n_shots: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Solve via qubosolver's LocalEmulator on the QUBO penalty matrix.

        This is the comparable Pasqal path: same QUBO objective as
        SimplePortfolioQUBO / phi.solve_qubo / dwave-neal SA. The
        difference is the backend — Pasqal's neutral-atom emulator
        instead of classical SA or quantum-inspired QIHD.

        Args:
            n_shots: Number of bitstring samples (unused by some
                qubosolver backends; kept for API symmetry).
            seed: RNG seed. Currently unused — qubosolver 0.5's
                LocalEmulator does not accept a seed; kept for API
                symmetry with the other solve_* methods.

        Returns:
            Standard result dict (see _build_result).
        """
        from qubosolver import (
            LocalEmulator,
            QUBOInstance,
            SolverConfig,
        )
        from qubosolver.solver import QuboSolver

        q_qubo = self._qubo_optimizer.build_qubo_matrix()
        instance = QUBOInstance(q_qubo)
        config = SolverConfig(use_quantum=True, backend=LocalEmulator())
        solver = QuboSolver(instance, config)
        solution = solver.solve()

        if not len(solution.bitstrings):
            raise RuntimeError(
                "qubosolver LocalEmulator returned no bitstrings. "
                "Check QUBOInstance/SolverConfig validity or upgrade "
                "qubosolver."
            )
        raw = np.asarray(solution.bitstrings[0], dtype=int)
        selection = self._round_and_repair(raw.astype(float))
        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)

    def _build_conflict_graph(self):
        """
        Build a pairwise asset-conflict graph for the MIS encoding.

        Nodes are asset indices 0..n-1. An edge (i, j) means assets i
        and j cannot coexist under a pairwise relaxation of the
        constraints — combined price > budget OR combined duration >
        max_duration. The MIS of this graph is the largest set of
        assets that fits PAIRWISE; it does NOT enforce the full
        multi-asset budget, total duration, or cardinality, and does
        NOT consider scores. Those are handled in post-selection.

        Returns:
            networkx.Graph
        """
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                over_budget = (
                    self.prices[i] + self.prices[j] > self.budget
                )
                over_duration = (
                    self.durations[i] + self.durations[j]
                    > self.max_duration
                )
                if over_budget or over_duration:
                    g.add_edge(i, j)
        return g

    def _score_aware_post_selection(self, candidates: List[int]) -> np.ndarray:
        """
        From an MIS candidate set, greedily pick highest-score assets
        until adding another would violate any full constraint
        (budget, duration, cardinality).
        """
        ordered = sorted(candidates, key=lambda i: -self.scores[i])
        chosen: List[int] = []
        for idx in ordered:
            trial = chosen + [idx]
            if (
                np.sum(self.prices[trial]) <= self.budget
                and np.sum(self.durations[trial]) <= self.max_duration
                and len(trial) <= self.max_cardinality
            ):
                chosen.append(idx)
        x = np.zeros(self.n, dtype=int)
        x[chosen] = 1
        return x

    def solve_mis(
        self,
        runs: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Solve via pairwise-conflict MIS + score-aware post-selection.

        NOT directly comparable to solve_qubo: the quantum step finds
        the largest set of assets that fit PAIRWISE within budget and
        duration; scores and multi-asset interactions are handled
        classically afterwards.

        Args:
            runs: Number of MIS solver shots.
            seed: RNG seed. Currently unused — kept for API symmetry
                with the other solve_* methods. (mis library does not
                accept a seed in BackendConfig as of 0.3.x.)

        Returns:
            Standard result dict (see _build_result).
        """
        from mis import (
            BackendConfig,
            MISInstance,
            MISSolver,
            SolverConfig as MisSolverConfig,
        )

        graph = self._build_conflict_graph()
        instance = MISInstance(graph)
        config = MisSolverConfig(
            backend=BackendConfig(backend="qutip"),
            runs=runs,
            max_number_of_solutions=5,
        )
        solver = MISSolver(instance, config)
        solutions = solver.solve()

        if not solutions:
            selection = np.zeros(self.n, dtype=int)
        else:
            best = solutions[0]
            selection = self._score_aware_post_selection(list(best.nodes))

        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)
