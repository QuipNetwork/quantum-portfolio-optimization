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

Six solver paths exposed to the experiments/simple benchmark
(Pasqal-QUBO, Pasqal-Slack, Pasqal-MIS, Pasqal-Pulser,
Pasqal-Pulser-DMM, Pasqal-Pulser-QUBO). All run on local emulators —
no Pasqal cloud account required. Two of the six are directly
comparable to the other QUBO rows in the benchmark (cuopt, QHD, Phi);
the other four are problem reformulations or hardware-control
demonstrations included for completeness and should be read with the
caveats below.


Why an arbitrary QUBO underperforms on neutral atoms
----------------------------------------------------

The same portfolio QUBO matrix solves to the optimum on classical
simulated annealing AND on a real D-Wave Advantage2 QPU, but only
reaches ~80% of optimum on Pasqal-QUBO. The difference is NOT
"embedding is hard" (D-Wave embeds too) — it is whether the
machine's couplings are independently PROGRAMMABLE:

  - D-Wave is a programmable Ising machine. Its problem Hamiltonian
    is H = Σ h_i s_i + Σ J_ij s_i s_j, and every h_i and J_ij is an
    independent dial. Minor-embedding maps each logical variable to
    a chain of physical qubits; the logical couplings are realized
    exactly on inter-chain couplers. As long as chains don't break,
    the embedded ground state IS the logical ground state. So
    QUBO-on-QPU == QUBO-on-SA == optimum.

  - Pasqal is an analog simulator with GEOMETRIC interactions. The
    Rydberg coupling V_ij = C6 / r_ij^6 is fixed by the physical
    distance between atoms — you place atoms, physics sets the
    couplings. It is always positive (repulsive), decays as 1/r^6,
    and is geometrically coupled (moving one atom changes all its
    pairwise interactions). An arbitrary QUBO coupling matrix (mixed
    signs, arbitrary magnitudes, all-to-all) cannot be realized
    exactly. qubosolver's greedy triangular-lattice embedding finds
    a best-fit atom layout that APPROXIMATES the couplings, and that
    approximation shifts the energy-landscape minimum off the true
    optimum. (Per-atom detuning via the DMM handles the linear/
    diagonal terms; it's the quadratic couplings that can't be
    realized.)

This is exactly why solve_mis works far better than solve_qubo:
Maximum Independent Set is the NATIVE Rydberg problem (blockade =
independence), so it maps onto the hardware with no approximation.
Match the encoding to the hardware's native interaction and Pasqal
shines; force an arbitrary QUBO onto it and the geometric
approximation costs you. (D-Wave's analogue of this wall is chain
breaks in minor-embedding, which is why Pasqal-Slack AND the larger
QUBO-Slack-QPU row both degrade as variable count grows.)


Pasqal-QUBO (solve_qubo) — comparable
-------------------------------------

Backend: qubo-solver 0.5.x (PyPI name; imported as `qubosolver`) with
LocalEmulator (neutral-atom emulator running locally).

Input: the same QUBO penalty matrix used by D-Wave SA, cuOpt-QP, and
Phi-QUBO — soft penalties for budget, duration, and cardinality from
SimplePortfolioQUBO.

Two implementation details worth knowing about, both fixed here:

1. Matrix convention. build_qubo_matrix() returns a SYMMETRIC matrix
   where each off-diagonal coupling lives in both Q[i,j] and Q[j,i].
   dwave-neal's BQM consumes the upper triangle once; qubosolver
   evaluates x^T Q x directly, which double-counts symmetric pairs.
   We pass np.triu(Q) to QUBOInstance so the energy landscape Pasqal
   sees matches the one D-Wave SA sees.

2. Sample selection. qubosolver returns multiple bitstrings sorted by
   QUBO cost. The lowest-cost bitstring isn't guaranteed to be the
   best portfolio — the soft penalty matrix can have its minimum at a
   suboptimal selection. _pick_best_feasible() iterates ALL returned
   bitstrings and returns the highest-portfolio-score feasible one.
   O(k·n) per call, negligible next to the solver runtime.


Pasqal-Slack (solve_qubo_slack) — comparable, but tradeoff
----------------------------------------------------------

Backend: qubo-solver LocalEmulator on the SlackPortfolioQUBO matrix
instead of SimplePortfolioQUBO.

The slack formulation converts each ≤ constraint into an = constraint
via auxiliary binary slack variables — eliminating the "penalty
pushes toward equality with target" bias of the simple penalty
matrix. The slack QUBO's global minimum aligns with the true
portfolio optimum (D-Wave SA confirms this on the same matrix).

Tradeoff: the slack matrix is much larger (asset bits + log2(B/prec)
slack bits per constraint × 3 constraints). qubosolver's emulator
scales poorly past ~20 total variables — empirically n=5 takes 3 s,
n=12 takes 2.4 HOURS at default coarse precision (1.0 / 5.0).
solve_qubo_slack hard-caps at n <= 8 (see _SLACK_MAX_N) to keep
benchmark runs practical.

Lesson for the client: "formulation quality" and "solver quality"
multiply, not add. A correct formulation paired with a solver that
under-samples the larger space can be empirically worse than a
biased formulation paired with a thorough solver.


Pasqal-MIS (solve_mis) — NOT directly comparable
-------------------------------------------------

Backend: maximum-independent-set 0.3.x with the local Qutip emulator.

The portfolio problem is re-encoded as a pairwise asset-conflict
graph (see _build_conflict_graph): nodes are assets, edges connect
pairs whose combined price exceeds the budget OR whose combined
duration exceeds the max-duration constraint. The quantum step finds
a Maximum Independent Set of that graph — the largest set of
mutually non-conflicting assets.

The MIS encoding is a relaxation:
  - It only captures PAIRWISE constraints. Multi-asset budget
    interactions (e.g., three cheap assets that together exceed the
    budget) are not represented.
  - The cardinality constraint is not encoded in the quantum step.
  - Asset scores are not encoded at all — MIS is unweighted.

After the quantum step, _score_aware_post_selection takes the
returned independent set and greedily picks assets in decreasing
score order until the next would violate any full constraint
(budget, duration, cardinality). This recovers some score signal,
but the optimization itself was unweighted. Empirically the MIS
encoding hits the global optimum when the optimal portfolio happens
to be pairwise-feasible (often the case for the problem sizes here).


Pasqal-Pulser (solve_pulser) — NOT directly comparable; n <= 12
---------------------------------------------------------------

Backend: raw pulser 1.6.x with the local QutipBackendV2 simulator.

A 1D atom register is built that mirrors the same pairwise conflict
graph as Pasqal-MIS — conflicting pairs placed within Rydberg
blockade radius, non-conflicting pairs outside it (see
_build_pulser_register). A linear-detuning adiabatic sweep at
constant Rabi amplitude is then run on the register, and the
most-common bitstring is sampled and repaired to feasibility.

Caveats on top of the Pasqal-MIS caveats:
  - The 1D layout cannot in general realize an arbitrary conflict
    graph (only interval-graph-like structures are exact).
  - Rydberg interactions are 1/r^6 repulsive only — the encoded
    "constraint" strength varies smoothly across pairs, not as a
    hard pairwise penalty.
  - Pulse parameters (Rabi=3 rad/µs, detuning sweep -8→+8 rad/µs,
    duration 6 µs) are tuned so dδ/dt < Ω² (adiabatic criterion)
    given AnalogDevice's 6000 ns sequence cap; they are NOT tuned
    per problem instance.
  - Local Qutip emulation cost scales as 2^n; solve_pulser caps at
    n <= 12 (see _PULSER_MAX_N).

This row is included as a demonstration that the problem CAN be
expressed at the hardware-control level on neutral atoms, not as a
benchmark of solver quality.


Pasqal-Pulser-DMM (solve_pulser_dmm) — weighted MIS (recommended DMM path)
--------------------------------------------------------------------------

Backend: raw pulser with the local QutipBackendV2 simulator on
DigitalAnalogDevice (which, unlike AnalogDevice, exposes a DMM
channel — see DEVNOTES/DMM_QUBO_ENCODING.md §2, and §10 for how this
repo uses the two DMM paths).

This path does NOT embed the full QUBO matrix; it reduces the problem
to a conflict graph instead. Embedding the full QUBO IS possible (that
is the Pasqal-Pulser-QUBO path below), but it is delicate: the
soft-penalty QUBO's largest off-diagonals are pairwise duration
products (2*lambda_d*d_i*d_j), so a NAIVE fixed-Rabi embedding places
feasible, high-value pairs CLOSE, and the Rydberg interaction —
positive-only and, at those magnitudes, a HARD blockade rather than the
finite penalty the QUBO intends — then forbids exciting both. The
Pasqal-Pulser-QUBO path fixes that by adapting the pulse energy to the
register (omega = 0.5*U_max). The weighted-MIS path here sidesteps the
whole issue: it never asks the off-diagonal to encode anything but a
binary "these two conflict," which the blockade represents exactly.

The fix splits the problem the way neutral atoms handle it best:

  - Register geometry encodes HARD pairwise feasibility (the conflict
    graph): conflict edges within the blockade, non-conflicting pairs
    outside it. Feasible pairs like A,E stay free to be co-excited.
    See _embed_conflict_register_2d.

  - The DMM diagonal encodes the OBJECTIVE: score-only epsilon
    (DEVNOTES §3a), high score -> low epsilon -> encouraged. Because
    constraints already live in the geometry, the naive score-only
    recipe is exactly right — there is no constraint term left to
    dominate the diagonal. This is a weighted MIS, and it prefers the
    higher-scoring A,E over the larger-but-lower-scoring A,C,D that the
    unweighted Pasqal-MIS path settles on.

Caveats: the conflict graph is a pairwise relaxation (it does not
encode multi-asset budget/duration or cardinality, so the dynamics can
sample infeasible states; _pick_best_feasible then reports the best
feasible sample), arbitrary conflict graphs are not exactly realizable
as a 2D unit-disk graph, and Qutip emulation is 2^n. Capped at
n <= _PULSER_DMM_MAX_N.


Pasqal-Pulser-QUBO (solve_pulser_qubo) — manual full-QUBO embedding
-------------------------------------------------------------------

A by-hand mapping of the COMPLETE penalty QUBO onto the hardware.
Where solve_pulser_dmm reduces the problem to a conflict graph, this
path keeps both parts of Q:

  - Off-diagonal Q[i,j] -> atom positions, by matching the Rydberg
    interaction C6/r_ij^6 to the (symmetric) off-diagonal directly
    (Nelder-Mead with a compactness penalty and a ring start; see
    _embed_register_2d), then scaling the register up to honour the
    minimum atom spacing.
  - Diagonal Q[i,i] -> detuning. The diagonal carries a strong negative
    linear bias (-2*lambda*b*g, built into SimplePortfolioQUBO) so
    there is no trivial all-zeros minimum; the global sweep provides the
    bulk detuning and a DMM adds a per-atom score tilt
    (epsilon = normalize(-score)).

The earlier naive version of this failed because it embedded the
off-diagonal at a FIXED Rabi, which blockaded feasible high-value pairs
apart. The fix is to tie the pulse energy scale to the register's OWN
strongest interaction: omega = 0.5 * U_max with
U_max = C6 / min_pairwise_distance^6, so the blockade radius tracks the
embedded geometry instead of fighting it. Selection is the best
feasible SAMPLE (no score-inflating repair). Capped at
n <= _PULSER_DMM_MAX_N (2^n emulation + 2n-coordinate embedding).
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

# Estimated neutral-atom register-cycle time per shot: atom loading +
# tweezer rearrangement (tens of ms, defect-free assembly) + state prep
# + fluorescence readout (the dominant, ms-scale component). The pulse
# evolution itself is ~microseconds and negligible against this. Modern
# neutral-atom demos run at ~30-50 cycles/second (~20-33 ms/cycle), so
# 30 ms is a representative per-shot figure. This is a fixed HARDWARE
# cost the local Qutip emulator does not (and cannot) model, so we add
# it as a documented estimate when reporting hardware-equivalent time.
# Source: neutral-atom cycle-time literature (readout-dominated,
# ~30-50 cycles/s); see e.g. arXiv:2510.25982.
_NEUTRAL_ATOM_CYCLE_MS = 30.0

# Hard cap on Pasqal-Slack problem size. The slack QUBO matrix adds
# log2(B/precision) bits per constraint × 3 constraints ≈ 15-20 extra
# variables at default coarse precision. qubosolver's LocalEmulator
# scales poorly past ~20 total variables; empirically n=5 takes 3 s,
# n=12 takes 2.4 hours. Cap at n=8 to keep benchmark runs reasonable.
_SLACK_MAX_N = 8

# Hard cap on the DMM Pulser paths (solve_pulser_dmm weighted-MIS and
# solve_pulser_qubo manual QUBO embedding). Two independent costs bite:
# Qutip emulation is 2^n, and the Nelder-Mead register embedding
# optimizes 2n coordinates and degrades in quality as n grows.
_PULSER_DMM_MAX_N = 10


class PasqalPortfolioOptimizer:
    """Portfolio optimizer with six Pasqal-flavored solver paths.

    QUBO and Slack (qubosolver), MIS and Pulser (conflict graph),
    Pulser-DMM (weighted MIS: conflict-graph register + score DMM), and
    Pulser-QUBO (manual full-QUBO embedding: off-diagonal -> atom
    positions, diagonal -> DMM detuning). See the module docstring.
    """

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

        Repair is a last-resort fallback only (step 4): when the solver
        returns at least one feasible bitstring we report the best of
        those as sampled, so the result reflects what the dynamics
        actually produced rather than what classical repair could
        reconstruct from infeasible samples.

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

        Raises:
            ValueError: if n > 8. The slack matrix at larger n
                requires emulation runs that take >10 minutes; cap
                here keeps benchmark runs practical.

        Returns:
            Standard result dict (see _build_result).
        """
        if self.n > _SLACK_MAX_N:
            raise ValueError(
                f"solve_qubo_slack capped at n <= {_SLACK_MAX_N} due "
                f"to qubosolver LocalEmulator runtime explosion on "
                f"the expanded slack matrix; got n={self.n}."
            )

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

        # Hardware-equivalent time. Two figures:
        #   hw_pure_ms  = the pulse-sequence (coherent evolution) duration
        #                 alone — the analogue of D-Wave's per-sample
        #                 anneal time. Microseconds.
        #   hw_total_ms = estimated FULL hardware time for all n_shots:
        #                 each shot needs a fresh register cycle (load +
        #                 rearrange + prep + readout, ~_NEUTRAL_ATOM_CYCLE_MS)
        #                 because fluorescence readout destroys the register.
        #                 This is an ESTIMATE (the emulator cannot measure
        #                 real apparatus timing); the pulse part is exact.
        hw_pure_ms = sequence.get_duration() / 1e6
        hw_total_ms = n_shots * (_NEUTRAL_ATOM_CYCLE_MS + hw_pure_ms)

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
        result = self._build_result(selection, energy, is_feasible)
        result['hw_total_ms'] = hw_total_ms
        result['hw_pure_ms'] = hw_pure_ms
        return result

    # ------------------------------------------------------------------
    # DMM Pulser paths (solve_pulser_dmm / solve_pulser_qubo) on
    # DigitalAnalogDevice. solve_pulser_dmm is a weighted MIS over a
    # conflict-graph register; solve_pulser_qubo embeds the full QUBO
    # (off-diagonal -> atom positions, diagonal -> per-atom DMM
    # detuning). See DEVNOTES/DMM_QUBO_ENCODING.md and the module
    # docstring.
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_epsilon(diag: np.ndarray) -> np.ndarray:
        """Normalize a diagonal vector to DMM weights epsilon in [0,1].

        Higher diagonal -> higher epsilon -> more suppressed |1> (the
        sign convention in DEVNOTES §3: ε↑ ⟺ diag↑ ⟺ less desirable).
        Includes the §3d range guarantee.
        """
        diag = np.asarray(diag, dtype=float)
        rng = diag.max() - diag.min()
        if rng <= 0:
            eps = np.zeros(len(diag))
        else:
            eps = (diag - diag.min()) / rng
        eps = np.clip(eps, 0.0, 1.0)
        assert 0.0 <= eps.min() <= eps.max() <= 1.0, (
            f"epsilon out of range: [{eps.min()}, {eps.max()}]"
        )
        return eps

    @staticmethod
    def _enforce_device_geometry(coords: np.ndarray, device) -> np.ndarray:
        """Recenter and rescale coords to satisfy device geometry.

        Scales up if the closest pair is below min_atom_distance; raises
        if the register cannot fit within max_radial_distance (the QUBO
        off-diagonal spread is not realizable in 2D at this size).
        """
        from scipy.spatial.distance import pdist

        coords = coords - coords.mean(axis=0)
        d = pdist(coords)
        if len(d):
            dmin = d.min()
            if 0 < dmin < device.min_atom_distance:
                coords = coords * (device.min_atom_distance * 1.05 / dmin)
        radial = (
            float(np.linalg.norm(coords, axis=1).max()) if len(coords) else 0.0
        )
        if radial > device.max_radial_distance:
            raise RuntimeError(
                f"Embedded register radius {radial:.1f} um exceeds device "
                f"max_radial_distance {device.max_radial_distance} um; the "
                f"QUBO off-diagonal spread is not realizable in 2D at this "
                f"size. Try fewer assets."
            )
        return coords

    def _embed_register_2d(self, q_off: np.ndarray):
        """Find a 2D register whose Rydberg interactions match the QUBO
        off-diagonal directly (the manual register mapping).

        Minimizes ||C6/r_ij^6 - |q_off|||  plus a small compactness
        penalty (keeps atoms near the array centre), starting from a
        ring layout. The fit is intentionally NOT rescaled into a device
        "blockade window" — the pulse energy scale is instead adapted to
        the register's own strongest interaction U_max in
        solve_pulser_qubo, so the blockade tracks the embedded geometry
        rather than fighting it. After the fit, the whole register is
        scaled up if any pair is below the device minimum spacing.

        Args:
            q_off: symmetric off-diagonal coupling matrix (diagonal
                ignored). Larger |q_off[i,j]| -> stronger desired
                repulsion -> atoms placed closer.

        Returns:
            (pulser.Register, DigitalAnalogDevice, coords ndarray).
        """
        from scipy.optimize import minimize
        from scipy.spatial.distance import pdist, squareform
        import pulser
        from pulser.devices import DigitalAnalogDevice

        device = DigitalAnalogDevice
        if not device.dmm_channels:
            raise RuntimeError(f"{device.name} has no DMM channel")

        m = q_off.shape[0]
        target = np.abs(np.array(q_off, dtype=float))
        np.fill_diagonal(target, 0.0)
        c6 = device.interaction_coeff

        def cost(flat):
            coords = flat.reshape(m, 2)
            dists = np.maximum(pdist(coords), 1e-6)
            new_q = squareform(c6 / dists ** 6)
            compact = 1e-4 * np.sum(np.linalg.norm(coords, axis=1) ** 2)
            return float(np.linalg.norm(new_q - target) + compact)

        # Ring initial guess (r0 = 15 um atoms on a circle).
        angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
        r0 = 15.0
        x0 = np.column_stack([r0 * np.cos(angles), r0 * np.sin(angles)]).ravel()
        res = minimize(
            cost, x0, method="Nelder-Mead",
            options={"maxiter": 200000, "xatol": 0.01, "fatol": 1e-6},
        )

        coords = self._enforce_device_geometry(res.x.reshape(m, 2), device)
        register = pulser.Register(
            {f"q{i}": tuple(coords[i]) for i in range(m)}
        )
        return register, device, coords

    def _build_dmm_sequence(
        self,
        register,
        device,
        epsilon: np.ndarray,
        omega: float,
        delta_0: float,
        delta_f: float,
        duration_ns: int = 4000,
        alpha_dmm: float = 0.6,
    ):
        """Build a global adiabatic sequence plus a DMM detuning channel.

        Follows DEVNOTES §4 (two-step DMM attach), §5 (amplitude), §6
        (hardware clamps), and §7 (7-point InterpolatedWaveform pulse).

        The energy scales (omega, delta_0, delta_f) are supplied by the
        caller: the weighted-MIS path fixes them to a fraction of the
        device limits, while the manual-QUBO path ties omega to the
        register's own strongest interaction U_max so the blockade
        tracks the embedded geometry. All three are assumed already
        clamped to the global channel's max_amp / max_abs_detuning.
        """
        import pulser
        from pulser.waveforms import InterpolatedWaveform, ConstantWaveform

        n = len(epsilon)
        ch = device.channels["rydberg_global"]

        # Duration: clamp to device max and round to the clock period.
        max_dur = getattr(device, "max_sequence_duration", None)
        T = min(duration_ns, max_dur) if max_dur else duration_ns
        cp = ch.clock_period
        T = max(int(T // cp) * cp, ch.min_duration)

        # §4 step 1 — spatial mask (static per-atom epsilon).
        det_map = register.define_detuning_map(
            {f"q{i}": float(epsilon[i]) for i in range(n)}
        )

        seq = pulser.Sequence(register, device)
        seq.declare_channel("global", "rydberg_global")
        # §4 step 2 — bind the spatial map to a hardware DMM channel.
        seq.config_detuning_map(det_map, "dmm_0")

        # §7 — global adiabatic pulse (Omega up/hold/down; delta swept
        # from negative through resonance to positive).
        omega_wf = InterpolatedWaveform(
            T, [1e-9, omega * 0.4, omega * 0.9, omega,
                omega * 0.7, omega * 0.3, 1e-9]
        )
        delta_wf = InterpolatedWaveform(
            T, [delta_0, delta_0 * 0.7, delta_0 * 0.2, delta_f * 0.2,
                delta_f * 0.4, delta_f * 0.7, delta_f]
        )
        seq.add(pulser.Pulse(omega_wf, delta_wf, phase=0.0), "global")

        # §5/§6 — DMM amplitude (negative, fraction of final global
        # detuning) with per-atom and total-array hardware clamps.
        dmm_ch = device.dmm_channels["dmm_0"]
        dmm_amplitude = -delta_f * alpha_dmm
        if dmm_ch.bottom_detuning is not None:
            dmm_amplitude = max(dmm_amplitude, dmm_ch.bottom_detuning)
        if dmm_ch.total_bottom_detuning is not None:
            total = float(np.sum(epsilon)) * dmm_amplitude
            if total < dmm_ch.total_bottom_detuning and total < 0:
                dmm_amplitude *= dmm_ch.total_bottom_detuning / total
        # §4 step 3 — temporal DMM waveform (scaled per-atom by epsilon).
        seq.add_dmm_detuning(ConstantWaveform(T, dmm_amplitude), "dmm_0")
        return seq, delta_f, dmm_amplitude

    def _run_dmm_backend(self, seq):
        """Run the sequence on the local Qutip emulator, return the
        bitstring->count dict."""
        from pulser.backends import QutipBackendV2

        result = QutipBackendV2(seq).run()
        if not hasattr(result, 'final_bitstrings'):
            raise RuntimeError(
                f"QutipBackendV2.run() returned an object with no "
                f"'final_bitstrings' attribute (got "
                f"{type(result).__name__}). Check pulser version."
            )
        counts = result.final_bitstrings
        if not counts:
            raise RuntimeError(
                "QutipBackendV2 returned no bitstrings on the DMM "
                "sequence. Check register/pulse/DMM configuration."
            )
        return counts

    def _conflict_pairs(self) -> List[tuple]:
        """Pairwise asset conflicts: (i, j) where i and j cannot coexist
        under a pairwise relaxation (combined price > budget OR combined
        duration > max_duration). Same relation as _build_conflict_graph,
        but returns plain tuples so the DMM-MWIS path needs no networkx.
        """
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (
                    self.prices[i] + self.prices[j] > self.budget
                    or self.durations[i] + self.durations[j] > self.max_duration
                ):
                    pairs.append((i, j))
        return pairs

    def _embed_conflict_register_2d(self, seed: int = 42):
        """Place atoms so conflicting pairs are inside the Rydberg
        blockade and non-conflicting pairs outside it.

        This is the register the weighted-MIS DMM path uses. Unlike
        _embed_register_2d (which matches the full QUBO off-diagonal and
        ends up blockading feasible high-value pairs), the target here is
        BINARY — strong interaction for conflict edges, weak for
        non-edges — so the geometry encodes only hard pairwise
        feasibility. Arbitrary conflict graphs are not exactly realizable
        as a 2D unit-disk graph, but the binary target embeds far more
        reliably than the full QUBO.

        Returns:
            (pulser.Register, DigitalAnalogDevice)
        """
        from scipy.optimize import minimize
        from scipy.spatial.distance import pdist, squareform
        import pulser
        from pulser.devices import DigitalAnalogDevice

        device = DigitalAnalogDevice
        if not device.dmm_channels:
            raise RuntimeError(f"{device.name} has no DMM channel")

        m = self.n
        c6 = device.interaction_coeff
        # Blockade radius at the Rabi the pulse will actually use
        # (_build_dmm_sequence drives at 0.5 * max_amp).
        omega = 0.5 * device.channels["rydberg_global"].max_amp
        rb = device.rydberg_blockade_radius(omega)

        # Target interaction: conflict edges placed well inside blockade,
        # non-edges well outside it.
        r_edge = max(0.6 * rb, device.min_atom_distance * 1.1)
        r_non = min(1.8 * rb, 2.0 * device.max_radial_distance)
        u_edge = c6 / r_edge ** 6
        u_non = c6 / r_non ** 6

        target = np.full((m, m), u_non)
        for (i, j) in self._conflict_pairs():
            target[i, j] = target[j, i] = u_edge
        np.fill_diagonal(target, 0.0)

        def cost(flat):
            coords = flat.reshape(m, 2)
            dists = np.maximum(pdist(coords), 1e-6)
            return float(np.linalg.norm(squareform(c6 / dists ** 6) - target))

        rng = np.random.default_rng(seed)
        spread = device.max_radial_distance / 2.0
        best, best_cost = None, np.inf
        for _ in range(5):
            x0 = rng.uniform(-spread, spread, size=2 * m)
            res = minimize(
                cost, x0, method="Nelder-Mead",
                options={"maxiter": 2000 * m, "xatol": 1e-3, "fatol": 1e-3},
            )
            if res.fun < best_cost:
                best_cost, best = res.fun, res.x

        coords = self._enforce_device_geometry(best.reshape(m, 2), device)
        register = pulser.Register(
            {f"q{i}": tuple(coords[i]) for i in range(m)}
        )
        return register, device

    def solve_pulser_dmm(
        self,
        n_shots: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Solve as a weighted Maximum Independent Set on a DMM register.

        This is the recommended DMM path. Rather than embedding the full
        QUBO off-diagonal (which needs the careful energy-scale handling
        of solve_pulser_qubo to avoid blockading feasible high-value
        pairs apart), it splits the problem the way neutral atoms handle
        it best:

          - Register geometry encodes HARD pairwise feasibility: conflict
            edges (combined price/duration over limit) within the Rydberg
            blockade, non-conflicting pairs outside it. The blockade then
            forbids selecting both members of a conflicting pair, while
            leaving feasible pairs free to be co-excited.

          - The DMM diagonal encodes the OBJECTIVE: per-atom detuning with
            score-only epsilon (DEVNOTES §3a) — high score -> low epsilon
            -> encouraged. Constraints are already in the geometry, so the
            naive score-only recipe is exactly right here (no constraint
            term to dominate the diagonal). This turns the unweighted MIS
            (which prefers more atoms, e.g. A,C,D) into a weighted MIS
            that prefers the higher-scoring A,E.

        The conflict graph is a pairwise relaxation (it does not encode
        multi-asset budget/duration or cardinality), so the dynamics can
        sample infeasible states; _pick_best_feasible reports the best
        already-feasible sample (repairing only if none is feasible).

        Args:
            n_shots: Sampling shots. Currently informational — Pulser's
                QutipBackendV2 uses EmulationConfig.default_num_shots
                (1000); kept for API symmetry.
            seed: RNG seed for the Nelder-Mead embedding restarts.

        Raises:
            ValueError: if n > _PULSER_DMM_MAX_N.

        Returns:
            Standard result dict (see _build_result).
        """
        if self.n > _PULSER_DMM_MAX_N:
            raise ValueError(
                f"solve_pulser_dmm capped at n <= {_PULSER_DMM_MAX_N} "
                f"(2^n Qutip emulation + Nelder-Mead embedding of 2n "
                f"coordinates); got n={self.n}."
            )

        register, device = self._embed_conflict_register_2d(seed=seed)
        # Score-only epsilon (§3a): -score so high score -> low epsilon.
        epsilon = self._normalize_epsilon(-self.scores)
        # Fixed energy scales (fraction of device limits): the conflict
        # register's blockade radius is sized off the same Rabi.
        ch = device.channels["rydberg_global"]
        omega = 0.5 * ch.max_amp
        delta_f = 0.4 * ch.max_abs_detuning
        seq, _df, _amp = self._build_dmm_sequence(
            register, device, epsilon, omega, -delta_f, delta_f
        )
        counts = self._run_dmm_backend(seq)

        candidates = [
            np.array([int(c) for c in b], dtype=int) for b in counts
        ]
        selection = self._pick_best_feasible(candidates)
        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)

    def solve_pulser_qubo(
        self,
        n_shots: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Manual full-QUBO embedding on a DMM register.

        The complete penalty QUBO is mapped onto the hardware by hand,
        rather than reduced to a conflict graph as solve_pulser_dmm does.

          - Off-diagonal Q[i,j] -> atom positions. The symmetric
            off-diagonal is matched to the Rydberg interaction C6/r_ij^6
            by _embed_register_2d (Nelder-Mead + compactness penalty).
          - Diagonal Q[i,i] -> detuning. The negative linear bias in the
            diagonal (-2*lambda*b*g, built into SimplePortfolioQUBO) makes
            selection favorable (no trivial all-zeros minimum); the global
            sweep supplies the bulk detuning and the DMM adds a per-atom
            score tilt (epsilon = normalize(-score)).

        The key to making the embedding physical (rather than blockading
        feasible pairs apart, as a naive fixed-Rabi embedding does) is
        that the pulse energy scale is tied to the register's OWN
        strongest interaction: omega = 0.5 * U_max with
        U_max = C6 / min_pairwise_distance^6. The blockade radius then
        tracks the embedded geometry instead of fighting it.

        Selection is by best feasible SAMPLE (via _pick_best_feasible,
        which only repairs as a last resort if nothing feasible was
        sampled) — the reported score reflects what the dynamics
        produced, not classical reconstruction.

        Args:
            n_shots: Sampling shots. Informational — Pulser's
                QutipBackendV2 uses EmulationConfig.default_num_shots
                (1000); kept for API symmetry.
            seed: RNG seed. Currently unused — the embedding init is a
                deterministic ring and QutipBackendV2 is unseeded; kept
                for API symmetry.

        Raises:
            ValueError: if n > _PULSER_DMM_MAX_N.

        Returns:
            Standard result dict (see _build_result).
        """
        if self.n > _PULSER_DMM_MAX_N:
            raise ValueError(
                f"solve_pulser_qubo capped at n <= {_PULSER_DMM_MAX_N} "
                f"(2^n Qutip emulation + Nelder-Mead embedding of 2n "
                f"coordinates); got n={self.n}."
            )

        from scipy.spatial.distance import pdist

        # Full penalty QUBO; symmetric off-diagonal drives the geometry.
        Q = self._qubo_optimizer.build_qubo_matrix()
        q_off = (Q + Q.T) / 2.0
        np.fill_diagonal(q_off, 0.0)

        register, device, coords = self._embed_register_2d(q_off)

        # Score-only DMM tilt (constraints live in the off-diagonal).
        epsilon = self._normalize_epsilon(-self.scores)

        # Energy scale adapts to the register's strongest interaction so
        # the blockade matches the embedded distances; clamp to the
        # global channel limits.
        ch = device.channels["rydberg_global"]
        u_max = device.interaction_coeff / float(pdist(coords).min()) ** 6
        omega = min(0.5 * u_max, 0.95 * ch.max_amp)
        delta_0 = max(-1.5 * omega, -0.95 * ch.max_abs_detuning)
        delta_f = min(0.8 * omega, 0.95 * ch.max_abs_detuning)

        seq, _df, _amp = self._build_dmm_sequence(
            register, device, epsilon, omega, delta_0, delta_f
        )
        counts = self._run_dmm_backend(seq)

        candidates = [
            np.array([int(c) for c in b], dtype=int) for b in counts
        ]
        selection = self._pick_best_feasible(candidates)
        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)


# Sampling each emulator path performs, for the demo summary. These are
# fixed by the backend, not by our n_shots argument (which the lazy
# imports document as unused / ignored):
#   - Pasqal-QUBO / Slack: shots are set inside qubosolver's
#     LocalEmulator() default; we pass no shot count, and solve_qubo's
#     n_shots is documented as unused.
#   - Pasqal-MIS:    runs=100 on the Qutip emulator (see solve_mis).
#   - Pasqal-Pulser / Pulser-DMM / Pulser-QUBO: 1000 shots —
#     Pulser's EmulationConfig default_num_shots for the BitStrings
#     observable; our n_shots is not forwarded to QutipBackendV2.
_PATH_SAMPLING = {
    "Pasqal-QUBO": "qubosolver LocalEmulator default shots (n_shots ignored)",
    "Pasqal-Slack": "qubosolver LocalEmulator default shots (n_shots ignored)",
    "Pasqal-MIS": "100 runs (Qutip emulator)",
    "Pasqal-Pulser": "1000 shots (Pulser default_num_shots; n_shots ignored)",
    "Pasqal-Pulser-DMM": "1000 shots (Pulser default_num_shots; n_shots ignored)",
    "Pasqal-Pulser-QUBO": "1000 shots (Pulser default_num_shots; n_shots ignored)",
}


def _demo_main():
    """Run ONLY the Pasqal solver paths on one problem, with QUBO info.

    Standalone entrypoint so you can iterate on the Pasqal paths without
    running the full multi-solver benchmark. The simple and slack QUBO
    matrices are always printed first (the --no-qubo flag suppresses
    them). Optional solver libs that aren't installed are reported as
    SKIPPED rather than failing the run.
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Run only the Pasqal portfolio solver paths."
    )
    parser.add_argument(
        "--num-assets",
        type=int,
        default=None,
        help="Random problem of this size; default uses the 5-asset "
             "example problem.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the random problem (default: 42).",
    )
    parser.add_argument(
        "--no-qubo",
        action="store_true",
        help="Skip the QUBO / slack-QUBO matrix display.",
    )
    args = parser.parse_args()

    if args.num_assets is None:
        assets = get_example_assets()
        constraints = get_example_constraints()
        title = "Simple Example Problem (n=5)"
    else:
        from benchmark_simple_qubo import generate_random_problem
        assets, constraints = generate_random_problem(
            args.num_assets, seed=args.seed
        )
        title = f"Random Problem (n={args.num_assets}, seed={args.seed})"

    print("=" * 100)
    print(" Pasqal-Only Portfolio Solver Run")
    print("=" * 100)
    print(f"Problem: {title}")
    print(f"  Assets: {len(assets)}")
    print(f"  Budget target: ${constraints['budget']:.2f}")
    print(f"  Max duration: {constraints['max_duration']}")
    print(f"  Max cardinality: {constraints['max_cardinality']}")

    # show_qubo is enabled automatically (per request); reuse the
    # benchmark's display so output matches `--show-qubo` exactly.
    if not args.no_qubo:
        from benchmark_simple_qubo import print_qubo_info, print_slack_qubo_info
        print_qubo_info(assets, constraints)
        print_slack_qubo_info(assets, constraints)

    optimizer = PasqalPortfolioOptimizer(
        assets=assets,
        budget=constraints['budget'],
        max_duration=constraints['max_duration'],
        max_cardinality=constraints['max_cardinality'],
        lambda_budget=constraints.get('lambda_budget', 2.0),
        lambda_duration=constraints.get('lambda_duration', 10.0),
        lambda_cardinality=constraints.get('lambda_cardinality', 5.0),
    )

    paths = [
        ("Pasqal-QUBO", optimizer.solve_qubo),
        ("Pasqal-Slack", optimizer.solve_qubo_slack),
        ("Pasqal-MIS", optimizer.solve_mis),
        ("Pasqal-Pulser", optimizer.solve_pulser),
        ("Pasqal-Pulser-DMM", optimizer.solve_pulser_dmm),
        ("Pasqal-Pulser-QUBO", optimizer.solve_pulser_qubo),
    ]

    width = 110
    print("=" * width)
    print(" Pasqal Solver Results")
    print("=" * width)
    print(f"{'Path':<26}{'Score':>8}{'Feasible':>10}{'Time(ms)':>10}  "
          f"{'Picks':<18}Sampling")
    print("-" * width)
    for name, fn in paths:
        start = time.perf_counter()
        try:
            r = fn()
            elapsed = (time.perf_counter() - start) * 1000
            picks = ",".join(r['selected_assets']) or "(none)"
            print(f"{name:<26}{r['total_score']:>8.1f}"
                  f"{str(r['is_feasible']):>10}{elapsed:>10.1f}  "
                  f"{picks:<18}{_PATH_SAMPLING[name]}")
        except ImportError as e:
            print(f"{name:<26}{'SKIPPED':>8}{'—':>10}{'—':>10}  "
                  f"{'(lib missing)':<18}{e}")
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            print(f"{name:<26}{'FAIL':>8}{'—':>10}{elapsed:>10.1f}  "
                  f"{'':<18}{type(e).__name__}: {e}")
    print("-" * width)


if __name__ == "__main__":
    _demo_main()
