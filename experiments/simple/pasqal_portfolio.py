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
                in the QUBO penalty matrix. Used by solve_qubo
                directly; used by solve_mis and solve_pulser only
                for post-hoc energy reporting via _compute_energy
                (their quantum step optimizes the pairwise conflict
                graph, not the QUBO matrix).
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

    def _portfolio_score(self, selection: np.ndarray) -> float:
        """Sum of selected asset scores. Used to rank candidates by
        portfolio objective (not by QUBO/Ising energy, which the
        Pasqal-QUBO bug-#1 investigation showed can diverge from the
        true portfolio objective for badly-conditioned QUBO matrices)."""
        x = np.asarray(selection)
        return float(np.sum(self.scores[x == 1]))

    def _pick_best_feasible(
        self,
        candidates: List[np.ndarray],
    ) -> np.ndarray:
        """
        Choose the best feasible binary selection from a list of
        candidates, ranked by portfolio score.

        Strategy:
        1. Round each candidate to {0,1} (idempotent for already-binary).
        2. Keep candidates that satisfy ALL constraints as-is.
        3. If any feasible candidate exists, return the highest-score one.
        4. Otherwise, repair each candidate via _round_and_repair and
           return the best-scoring of the repaired set.

        This handles the common case where a Pasqal/D-Wave solver
        returns multiple low-cost bitstrings but the lowest-cost one
        isn't the best portfolio (matrix convention or formulation
        biases). O(k·n) for k candidates.
        """
        if not candidates:
            return np.zeros(self.n, dtype=int)

        rounded = [
            (np.asarray(c) > 0.5).astype(int) for c in candidates
        ]
        feasible = [x for x in rounded if self._is_feasible(x)]
        if feasible:
            return max(feasible, key=self._portfolio_score)

        repaired = [self._round_and_repair(x.astype(float)) for x in rounded]
        return max(repaired, key=self._portfolio_score)

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

        # qubosolver evaluates x^T Q x with the matrix as-is. Our
        # build_qubo_matrix returns a SYMMETRIC Q where each off-diagonal
        # coupling lives in both Q[i,j] and Q[j,i] — so x^T Q x would
        # double-count pair interactions vs the dwave-neal convention
        # (upper-triangle only). Pass the upper triangle so the energy
        # landscape matches what dwave-neal sees and what the QUBO
        # formulation in simple_portfolio_qubo.py was designed for.
        q_qubo = np.triu(self._qubo_optimizer.build_qubo_matrix())
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
        # Evaluate ALL returned bitstrings by portfolio score. The
        # lowest-QUBO-cost bitstring isn't guaranteed to be the best
        # portfolio (the QUBO penalty matrix may bias toward states
        # that minimize penalty terms over score); picking the best
        # feasible across the full sample list is O(k·n), trivially
        # cheap compared to the solver run.
        candidates = [
            np.asarray(b, dtype=int) for b in solution.bitstrings
        ]
        selection = self._pick_best_feasible(candidates)
        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)

    def solve_qubo_slack(
        self,
        n_shots: int = 100,
        seed: int = 42,
        budget_precision: float = 1.0,
        duration_precision: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Solve via qubosolver on the slack-variable QUBO matrix.

        Uses SlackPortfolioQUBO instead of SimplePortfolioQUBO. The
        slack formulation converts ≤ constraints into = constraints
        via auxiliary slack variables, eliminating the
        "penalty pushes toward equality" bias of the simple penalty
        matrix. The resulting QUBO has more variables (n + slack
        bits) but a global minimum that aligns with the true portfolio
        optimum.

        Args:
            n_shots: Number of bitstring samples (unused by some
                qubosolver backends; kept for API symmetry).
            seed: RNG seed. Currently unused by LocalEmulator.
            budget_precision: Slack discretization step for budget.
                Smaller value = more slack bits = larger QUBO matrix
                = much slower local emulation. Defaults are coarse
                (1.0) since Pasqal's LocalEmulator scales poorly with
                problem size; SlackPortfolioQUBO's own SA-based
                default is 0.1 but that takes ~4 min/call here.
            duration_precision: Same tradeoff (default 5.0; SA path
                uses 1.0).

        Returns:
            Standard result dict (see _build_result).
        """
        from qubosolver import (
            LocalEmulator,
            QUBOInstance,
            SolverConfig,
        )
        from qubosolver.solver import QuboSolver
        from slack_portfolio_qubo import SlackPortfolioQUBO

        slack_opt = SlackPortfolioQUBO(
            assets=self.assets,
            budget=self.budget,
            max_duration=self.max_duration,
            max_cardinality=self.max_cardinality,
            lambda_budget=self._qubo_optimizer.lambda_b,
            lambda_duration=self._qubo_optimizer.lambda_d,
            lambda_cardinality=self._qubo_optimizer.lambda_c,
            budget_precision=budget_precision,
            duration_precision=duration_precision,
        )

        q_slack = np.triu(slack_opt.build_qubo_matrix())
        instance = QUBOInstance(q_slack)
        config = SolverConfig(use_quantum=True, backend=LocalEmulator())
        solver = QuboSolver(instance, config)
        solution = solver.solve()

        if not len(solution.bitstrings):
            raise RuntimeError(
                "qubosolver LocalEmulator returned no bitstrings on "
                "slack QUBO. Check QUBOInstance/SolverConfig or "
                "upgrade qubosolver."
            )

        # Slack bitstring layout: first n bits are asset selections,
        # remaining bits are budget/duration/cardinality slack
        # variables. Strip the slack bits before scoring.
        candidates = [
            np.asarray(b, dtype=int)[: self.n] for b in solution.bitstrings
        ]
        selection = self._pick_best_feasible(candidates)
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

        Note: This is a greedy first-fit heuristic, not an optimal
        sub-solver. It may miss higher-total-score combinations where
        skipping a high-scoring asset would unlock multiple lower-
        scoring ones.

        Args:
            candidates: Asset indices returned by the MIS solver.

        Returns:
            Binary numpy array of shape (n,).
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
            if not hasattr(best, 'nodes'):
                raise RuntimeError(
                    f"MISSolver returned a solution with no 'nodes' "
                    f"attribute (got {type(best).__name__}). Check "
                    f"mis library version (expected >= 0.3.x)."
                )
            selection = self._score_aware_post_selection(list(best.nodes))

        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)

    def _build_pulser_register(self):
        """
        Build a 1D atomic register encoding the pairwise conflict
        graph as Rydberg blockade.

        Conflicting pairs (edges in _build_conflict_graph) are placed
        within the blockade radius; non-conflicting pairs are placed
        outside it. This is a heuristic 1D layout — sufficient for
        small problems (n <= 12) but cannot in general realize an
        arbitrary conflict graph in 1D.

        Returns:
            (pulser.Register, pulser.devices.Device, float blockade_radius_um)
        """
        import pulser
        from pulser import AnalogDevice

        device = AnalogDevice
        # Use Rabi=3.0 rad/us throughout (placement + pulse). Higher
        # Rabi shrinks blockade radius (~Ω^-1/6) which lets more atoms
        # fit within the device radial constraint, AND tightens the
        # adiabatic criterion dδ/dt << Ω². At Ω=1 the default sweep
        # was 2500× too fast for adiabaticity; at Ω=3 it's manageable.
        rabi_rad_per_us = 3.0
        blockade_um = device.rydberg_blockade_radius(rabi_rad_per_us)
        self._pulser_rabi = rabi_rad_per_us

        # AnalogDevice has a max_radial_distance (atoms must lie within
        # that radius of the array center). A 1D chain of n atoms with
        # uniform step S, centered at origin, has extremes at ±(n-1)·S/2,
        # so S ≤ 2·max_radial / (n-1). Cap the non-conflict step (which
        # we'd like > blockade) at 0.95×max_step so it still fits when
        # max_step < 2·blockade. This makes non-conflict pairs only
        # weakly outside blockade for tight problems — a known limitation
        # documented above.
        max_step = 2.0 * device.max_radial_distance / max(self.n - 1, 1)
        conflict_step = min(0.85 * blockade_um, 0.95 * max_step)
        non_conflict_step = min(2.0 * blockade_um, 0.95 * max_step)
        conflict_step = max(conflict_step, device.min_atom_distance)
        non_conflict_step = max(non_conflict_step, device.min_atom_distance)

        graph = self._build_conflict_graph()
        coords = [(0.0, 0.0)]
        for i in range(1, self.n):
            prev = coords[-1]
            step = conflict_step if graph.has_edge(i - 1, i) else non_conflict_step
            coords.append((prev[0] + step, 0.0))

        # Center the chain at the origin so coords are symmetric.
        span_mid = coords[-1][0] / 2.0
        coords = [(x - span_mid, y) for (x, y) in coords]

        register = pulser.Register.from_coordinates(coords, prefix="q")
        return register, device, blockade_um

    def _build_adiabatic_sequence(self, register, device):
        """
        Build a linear-detuning adiabatic Rydberg sequence.

        Sweeps detuning from negative (favoring |g>) to positive
        (favoring |r>) at constant Rabi amplitude. Total duration
        4 us is a heuristic — adjust for problem hardness.
        """
        import pulser
        from pulser.waveforms import RampWaveform, ConstantWaveform

        sequence = pulser.Sequence(register, device)
        sequence.declare_channel("rydberg_global", "rydberg_global")

        # AnalogDevice caps sequence duration at 6000 ns. Use 6000 ns
        # at Rabi=3 rad/us: dδ/dt = 16/6 ≈ 2.67 rad/us² vs Ω² = 9 —
        # adiabatic ratio ~0.3, marginal but acceptable. Sweep -8 → +8
        # crosses resonance with symmetric margin.
        duration_ns = 6000
        rabi_value = getattr(self, '_pulser_rabi', 3.0)
        rabi = ConstantWaveform(duration_ns, rabi_value)
        detuning = RampWaveform(duration_ns, -8.0, 8.0)
        pulse = pulser.Pulse(rabi, detuning, phase=0.0)
        sequence.add(pulse, "rydberg_global")
        return sequence

    def solve_pulser(
        self,
        n_shots: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Solve via a hand-built Rydberg adiabatic pulse sequence.

        Encodes the same pairwise conflict graph as solve_mis as a 1D
        atom register, then runs an adiabatic detuning sweep at
        constant Rabi amplitude. The most-common bitstring is taken
        and repaired to feasibility.

        NOT comparable to solve_qubo. Hard cap n <= 12 due to Qutip
        emulation cost (2^n state vector).

        Args:
            n_shots: Number of samples. Currently unused —
                QutipBackendV2.run() does not expose a shot count
                parameter; kept for API symmetry with the other
                solve_* methods.
            seed: RNG seed. Currently unused — Pulser's QutipBackendV2
                does not accept a seed; kept for API symmetry with
                the other solve_* methods.

        Raises:
            ValueError: if n > 12 (emulation infeasible).

        Returns:
            Standard result dict (see _build_result).
        """
        if self.n > _PULSER_MAX_N:
            raise ValueError(
                f"solve_pulser only supports n <= {_PULSER_MAX_N} in "
                f"local Qutip emulation; got n={self.n}."
            )

        from pulser.backends import QutipBackendV2

        register, device, _blockade = self._build_pulser_register()
        sequence = self._build_adiabatic_sequence(register, device)

        backend = QutipBackendV2(sequence)
        result = backend.run()

        if not hasattr(result, 'final_bitstrings'):
            raise RuntimeError(
                f"QutipBackendV2.run() returned an object with no "
                f"'final_bitstrings' attribute (got "
                f"{type(result).__name__}). Check pulser version."
            )
        counts = result.final_bitstrings
        if not counts:
            raise RuntimeError(
                "QutipBackendV2 returned no bitstrings. Check "
                "register and pulse sequence."
            )

        # Most common bitstring, lexicographic tie-break for determinism.
        best_str = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        raw = np.array([int(c) for c in best_str], dtype=int)
        if raw.shape[0] != self.n:
            raise RuntimeError(
                f"Pulser returned a bitstring of length "
                f"{raw.shape[0]} but register has {self.n} atoms. "
                f"Check pulser version."
            )

        selection = self._round_and_repair(raw.astype(float))
        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)
