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
OpenPhiSolve (Artephi Computing) Portfolio Optimizer.

Uses PhiSolve's quantum-inspired QIHD backend with PDQP refinement
to solve the binary asset selection problem. Two formulation paths:

1. QUBO path: Feeds the QUBO penalty matrix directly to PhiSolve's
   native QUBO solver. Same energy landscape as simulated annealing,
   solved with quantum-inspired Hamiltonian descent instead.

2. MIQP path: Formulates the problem as a mixed-integer QP with
   explicit linear constraints (budget, duration, cardinality).
   All variables are binary. The QIHD backend generates samples
   and PDQP refines them respecting the constraints.

Both paths use the two-stage pipeline:
   QIHD (sample generation) → PDQP (solution refinement)

Usage:
    from phi_portfolio import PhiPortfolioOptimizer

    assets = [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        ...
    ]

    optimizer = PhiPortfolioOptimizer(
        assets=assets, budget=3.0, max_duration=15, max_cardinality=3
    )

    # QUBO path (penalty matrix)
    result = optimizer.solve_qubo()

    # MIQP path (native linear constraints)
    result = optimizer.solve_miqp()
"""

import numpy as np
from typing import List, Dict, Any, Optional

from simple_portfolio_qubo import (
    SimplePortfolioQUBO,
    get_example_assets,
    get_example_constraints,
)


class PhiPortfolioOptimizer:
    """
    Portfolio optimizer using OpenPhiSolve (Artephi Computing).

    Solves the binary asset selection problem via quantum-inspired
    Hamiltonian descent (QIHD) with PDQP refinement.
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
        Initialize the PhiSolve portfolio optimizer.

        Args:
            assets: List of asset dicts with 'id', 'price',
                'duration', 'score'.
            budget: Maximum total price (<=).
            max_duration: Maximum total duration (<=).
            max_cardinality: Maximum number of assets (<=).
            lambda_budget: Penalty weight for budget (QUBO path).
            lambda_duration: Penalty weight for duration
                (QUBO path).
            lambda_cardinality: Penalty weight for cardinality
                (QUBO path).
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

        # Internal QUBO optimizer for matrix construction
        self._qubo_optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=budget,
            max_duration=max_duration,
            max_cardinality=max_cardinality,
            lambda_budget=lambda_budget,
            lambda_duration=lambda_duration,
            lambda_cardinality=lambda_cardinality,
        )

    def build_qubo_problem(self):
        """
        Build a PhiSolve QUBO problem from the penalty matrix.

        PhiSolve QUBO convention: minimize 1/2 x^T Q x.
        Our QUBO: minimize x^T Q_qubo x.
        Therefore: Q_phi = 2 * Q_qubo.

        Returns:
            phisolve.problems.QUBO instance.
        """
        from phisolve.problems.qubo import QUBO

        q_qubo = self._qubo_optimizer.build_qubo_matrix()
        q_phi = 2.0 * q_qubo
        return QUBO(Q=q_phi)

    def build_miqp_problem(self):
        """
        Build a PhiSolve MIQP with native linear constraints.

        Formulates: minimize -score^T x
        subject to:
            price^T x <= budget
            duration^T x <= max_duration
            1^T x <= max_cardinality
            x_i in {0, 1}

        The objective is encoded as 1/2 x^T Q x + w^T x with
        Q = 0 (no quadratic term) and w = -scores.

        Returns:
            phisolve.problems.MIQP instance.
        """
        from phisolve import MIQP

        n = self.n

        # Pure linear objective: w = -scores (minimize = maximize score)
        Q = np.zeros((n, n))
        w = -self.scores.astype(float)

        # Inequality constraints: A x <= b
        A = np.vstack([
            self.prices,
            self.durations,
            np.ones(n),
        ])
        b = np.array([
            self.budget,
            self.max_duration,
            self.max_cardinality,
        ], dtype=float)

        # No equality constraints
        C = np.zeros((0, n))
        d = np.zeros(0)

        # Bounds: all binary [0, 1]
        lbs = np.zeros(n)
        ubs = np.ones(n)

        return MIQP(
            Q=Q, w=w,
            A=A, b=b,
            C=C, d=d,
            n_binary_vars=n,
            bounds=(lbs, ubs),
        )

    def solve_qubo(
        self,
        n_shots: int = 100,
        # OpenPhiSolve 0.2.0 has a bug: qihd.py:152 divides by
        # (n_steps - 1000), so n_steps <= 1000 causes div-by-zero.
        # Use the upstream default of 10000 until the bug is fixed.
        n_steps: int = 10000,
        seed: int = 42,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Solve using the QUBO path (penalty matrix).

        Args:
            n_shots: Number of samples to generate.
            n_steps: Number of QIHD evolution steps (must be > 1000).
            seed: Random seed for reproducibility.
            device: JAX device ("cpu" or "gpu").

        Returns:
            Standard result dict with selection, metrics,
            feasibility.
        """
        from phisolve import PhiMIQP, QIHD, PDQP

        prob = self.build_qubo_problem()
        backend = QIHD(
            n_shots=n_shots, n_steps=n_steps,
            seed=seed, device=device,
        )
        refiner = PDQP(iterations=n_steps, device=device)

        model = PhiMIQP(prob, backend, refiner)
        response = model.solve()

        raw = np.array(response.minimizer)
        selection = np.round(raw).astype(int)
        selection = np.clip(selection, 0, 1)

        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)

        if not is_feasible:
            selection = self._round_and_repair(
                raw.astype(float)
            )
            energy = self._compute_energy(selection)
            is_feasible = self._is_feasible(selection)

        return self._build_result(selection, energy, is_feasible)

    def solve_miqp(
        self,
        n_shots: int = 100,
        # OpenPhiSolve 0.2.0 has a bug: qihd.py:152 divides by
        # (n_steps - 1000), so n_steps <= 1000 causes div-by-zero.
        # Use the upstream default of 10000 until the bug is fixed.
        n_steps: int = 10000,
        seed: int = 42,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Solve using the MIQP path (native linear constraints).

        Args:
            n_shots: Number of samples to generate.
            n_steps: Number of QIHD evolution steps (must be > 1000).
            seed: Random seed for reproducibility.
            device: JAX device ("cpu" or "gpu").

        Returns:
            Standard result dict with selection, metrics,
            feasibility.
        """
        from phisolve import PhiMIQP, QIHD, PDQP

        prob = self.build_miqp_problem()
        backend = QIHD(
            n_shots=n_shots, n_steps=n_steps,
            seed=seed, device=device,
        )
        refiner = PDQP(iterations=n_steps, device=device)

        model = PhiMIQP(prob, backend, refiner)
        response = model.solve()

        raw = np.array(response.minimizer)
        selection = self._round_and_repair(raw.astype(float))
        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)

    def _round_and_repair(
        self, continuous: np.ndarray
    ) -> np.ndarray:
        """
        Round continuous solution to binary and repair if infeasible.

        Strategy:
        1. Threshold at 0.5
        2. If infeasible, greedily drop lowest-score assets

        Args:
            continuous: Continuous solution in [0,1]^n.

        Returns:
            Binary selection vector.
        """
        x = (np.asarray(continuous) > 0.5).astype(int)

        if self._is_feasible(x):
            return x

        # Greedy repair: drop lowest-score selected assets first
        selected = np.where(x == 1)[0]
        order = sorted(selected, key=lambda i: self.scores[i])
        for idx in order:
            x[idx] = 0
            if self._is_feasible(x):
                break

        return x

    def _is_feasible(self, selection: np.ndarray) -> bool:
        """Check if selection satisfies all constraints."""
        check = self.check_constraints(selection)
        return (
            check['budget_satisfied']
            and check['duration_satisfied']
            and check['cardinality_satisfied']
        )

    def _compute_energy(self, selection: np.ndarray) -> float:
        """Compute QUBO energy for a binary selection."""
        q = self._qubo_optimizer.build_qubo_matrix()
        x = np.asarray(selection, dtype=float)
        return float(x @ q @ x)

    def _build_result(
        self,
        selection: np.ndarray,
        energy: float,
        is_feasible: bool,
    ) -> Dict[str, Any]:
        """Build standard result dictionary."""
        selected_indices = np.where(selection == 1)[0]

        if len(selected_indices) > 0:
            total_price = float(
                np.sum(self.prices[selected_indices])
            )
            total_duration = float(
                np.sum(self.durations[selected_indices])
            )
            total_score = float(
                np.sum(self.scores[selected_indices])
            )
        else:
            total_price = 0.0
            total_duration = 0.0
            total_score = 0.0

        return {
            'selection': selection,
            'selected_assets': [
                self.asset_ids[i] for i in selected_indices
            ],
            'energy': energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_indices),
            'is_feasible': is_feasible,
        }

    def check_constraints(
        self, selection: np.ndarray
    ) -> Dict[str, Any]:
        """
        Check if a selection satisfies all constraints.

        Args:
            selection: Binary selection vector.

        Returns:
            Dictionary with constraint satisfaction status.
        """
        x = np.asarray(selection)
        selected_indices = np.where(x == 1)[0]

        if len(selected_indices) > 0:
            total_price = float(
                np.sum(self.prices[selected_indices])
            )
            total_duration = float(
                np.sum(self.durations[selected_indices])
            )
        else:
            total_price = 0.0
            total_duration = 0.0

        num_selected = len(selected_indices)

        return {
            'budget_satisfied': total_price <= self.budget,
            'duration_satisfied': (
                total_duration <= self.max_duration
            ),
            'cardinality_satisfied': (
                num_selected <= self.max_cardinality
            ),
            'total_price': total_price,
            'total_duration': total_duration,
            'num_selected': num_selected,
            'budget_limit': self.budget,
            'duration_limit': self.max_duration,
            'cardinality_limit': self.max_cardinality,
        }


def main():
    """Demo using the example problem."""
    print("=" * 80)
    print("OpenPhiSolve Portfolio Optimizer Demo")
    print("Using PhiSolve (Quantum-Inspired Hamiltonian Descent)")
    print("=" * 80)

    assets = get_example_assets()
    constraints = get_example_constraints()

    print("\nAsset Data:")
    print("-" * 50)
    for a in assets:
        print(
            f"  {a['id']}: price={a['price']:.2f}, "
            f"duration={a['duration']}, score={a['score']}"
        )

    print("\nConstraints (all are <= inequalities):")
    print(f"  Budget: <= ${constraints['budget']:.2f}")
    print(f"  Max Duration: <= {constraints['max_duration']}")
    print(f"  Max Cardinality: <= {constraints['max_cardinality']}")

    optimizer = PhiPortfolioOptimizer(
        assets=assets,
        budget=constraints['budget'],
        max_duration=constraints['max_duration'],
        max_cardinality=constraints['max_cardinality'],
        lambda_budget=constraints.get('lambda_budget', 2.0),
        lambda_duration=constraints.get(
            'lambda_duration', 10.0
        ),
        lambda_cardinality=constraints.get(
            'lambda_cardinality', 5.0
        ),
    )

    # --- QUBO Path ---
    print("\n" + "=" * 80)
    print("Solving with Phi-QUBO (penalty matrix via QIHD)...")
    print("=" * 80)

    result_qubo = optimizer.solve_qubo()

    print(f"\n  Selected Assets: {result_qubo['selected_assets']}")
    print(f"  Energy: {result_qubo['energy']:.2f}")
    print(f"  Total Score: {result_qubo['total_score']}")
    print(f"  Total Price: ${result_qubo['total_price']:.2f}")
    print(f"  Total Duration: {result_qubo['total_duration']}")
    print(f"  Is Feasible: {result_qubo['is_feasible']}")

    # --- MIQP Path ---
    print("\n" + "=" * 80)
    print(
        "Solving with Phi-MIQP "
        "(native linear constraints via QIHD)..."
    )
    print("=" * 80)

    result_miqp = optimizer.solve_miqp()

    print(
        f"\n  Selected Assets: {result_miqp['selected_assets']}"
    )
    print(f"  Energy: {result_miqp['energy']:.2f}")
    print(f"  Total Score: {result_miqp['total_score']}")
    print(f"  Total Price: ${result_miqp['total_price']:.2f}")
    print(f"  Total Duration: {result_miqp['total_duration']}")
    print(f"  Is Feasible: {result_miqp['is_feasible']}")


if __name__ == "__main__":
    main()
