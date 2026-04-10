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
NVIDIA cuOpt Portfolio Optimizer.

Uses cuOpt's GPU-accelerated optimization engine for the binary asset
selection problem. Two formulation paths:

1. MILP path: Binary integer programming with linear constraints.
   Each asset is a binary variable (0/1). Budget, duration, and
   cardinality are linear inequality constraints. Exact solution
   with no rounding needed — direct GPU analog of scipy.optimize.milp.

2. QP path: Reuses the QUBO penalty matrix via quadratic programming.
   Same energy landscape as simulated annealing and QHD-QP, solved
   with cuOpt's GPU QP solver (PDLP / Barrier / Dual Simplex).
   Continuous relaxation on [0,1]^n, then rounded to binary.

Usage:
    from cuopt_portfolio import CuOptPortfolioOptimizer

    assets = [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        ...
    ]

    optimizer = CuOptPortfolioOptimizer(
        assets=assets, budget=3.0, max_duration=15, max_cardinality=3
    )

    # MILP path (binary variables, exact)
    result = optimizer.solve_milp()

    # QP path (QUBO matrix, continuous relaxation)
    result = optimizer.solve_qp()
"""

import numpy as np
from typing import List, Dict, Any, Optional

from simple_portfolio_qubo import (
    SimplePortfolioQUBO,
    get_example_assets,
    get_example_constraints,
)


class CuOptPortfolioOptimizer:
    """
    Portfolio optimizer using NVIDIA cuOpt.

    Solves the binary asset selection problem via GPU-accelerated
    MILP (exact) or QP (continuous relaxation + rounding).
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
        Initialize the cuOpt portfolio optimizer.

        Args:
            assets: List of asset dicts with 'id', 'price',
                'duration', 'score'.
            budget: Maximum total price (<=).
            max_duration: Maximum total duration (<=).
            max_cardinality: Maximum number of assets (<=).
            lambda_budget: Penalty weight for budget (QP path).
            lambda_duration: Penalty weight for duration (QP path).
            lambda_cardinality: Penalty weight for cardinality
                (QP path).
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

        # Internal QUBO optimizer for QP path matrix construction
        self._qubo_optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=budget,
            max_duration=max_duration,
            max_cardinality=max_cardinality,
            lambda_budget=lambda_budget,
            lambda_duration=lambda_duration,
            lambda_cardinality=lambda_cardinality,
        )

        # Store last continuous solution for analysis (QP path)
        self._last_continuous: Optional[np.ndarray] = None

    def build_milp_model(self):
        """
        Build a cuOpt MILP model for binary asset selection.

        Maximize total score subject to budget, duration, and
        cardinality constraints. Each asset is a binary variable.

        Returns:
            Tuple of (Problem, list of variables).
        """
        from cuopt.linear_programming.problem import (
            Problem, VType, MINIMIZE,
        )

        prob = Problem("portfolio_milp")

        # Binary decision variables: x_i in {0, 1}
        x = [
            prob.addVariable(
                lb=0, ub=1, vtype=VType.INTEGER,
                name=f"x_{i}",
            )
            for i in range(self.n)
        ]

        # Objective: minimize negative score (maximize score)
        obj = sum(
            float(-self.scores[i]) * x[i]
            for i in range(self.n)
        )
        prob.setObjective(obj, sense=MINIMIZE)

        # Budget constraint: sum(price_i * x_i) <= budget
        budget_expr = sum(
            float(self.prices[i]) * x[i]
            for i in range(self.n)
        )
        prob.addConstraint(
            budget_expr <= self.budget, name="budget"
        )

        # Duration constraint: sum(dur_i * x_i) <= max_duration
        duration_expr = sum(
            float(self.durations[i]) * x[i]
            for i in range(self.n)
        )
        prob.addConstraint(
            duration_expr <= self.max_duration,
            name="duration",
        )

        # Cardinality constraint: sum(x_i) <= max_cardinality
        card_expr = sum(x[i] for i in range(self.n))
        prob.addConstraint(
            card_expr <= self.max_cardinality,
            name="cardinality",
        )

        return prob, x

    def build_qp_model(self):
        """
        Build a cuOpt QP model from the QUBO penalty matrix.

        cuOpt QP convention: minimize (1/2) x^T Q x + b^T x.
        Our QUBO: minimize x^T Q_qubo x.
        Therefore: Q_cuopt = 2 * Q_qubo, b = 0.

        Returns:
            Tuple of (Problem, list of variables).
        """
        from cuopt.linear_programming.problem import (
            Problem, MINIMIZE,
        )

        prob = Problem("portfolio_qp")

        # Continuous decision variables on [0, 1]
        x = [
            prob.addVariable(lb=0.0, ub=1.0, name=f"x_{i}")
            for i in range(self.n)
        ]

        # Build quadratic objective from QUBO matrix
        q_qubo = self._qubo_optimizer.build_qubo_matrix()
        q_cuopt = 2.0 * q_qubo

        quad_expr = None
        for i in range(self.n):
            for j in range(self.n):
                if abs(q_cuopt[i, j]) > 1e-12:
                    term = float(q_cuopt[i, j]) * x[i] * x[j]
                    if quad_expr is None:
                        quad_expr = term
                    else:
                        quad_expr += term

        prob.setObjective(quad_expr, sense=MINIMIZE)

        return prob, x

    def solve_milp(self) -> Dict[str, Any]:
        """
        Solve using the MILP path (binary variables, exact).

        Returns:
            Standard result dict with selection, metrics,
            feasibility.
        """
        prob, x = self.build_milp_model()
        prob.solve()

        selection = np.array([
            int(round(x[i].Value)) for i in range(self.n)
        ])

        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)

    def solve_qp(self) -> Dict[str, Any]:
        """
        Solve using the QP path (QUBO matrix, continuous relaxation).

        Solves on [0,1]^n then rounds to binary and repairs if
        infeasible.

        Returns:
            Standard result dict with selection, metrics,
            feasibility.
        """
        prob, x = self.build_qp_model()
        prob.solve()

        continuous = np.array([
            float(x[i].Value) for i in range(self.n)
        ])
        self._last_continuous = continuous

        selection = self._round_and_repair(continuous)
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

    def get_continuous_solution(self) -> Optional[np.ndarray]:
        """Return the last continuous (pre-rounding) solution."""
        return self._last_continuous


def main():
    """Demo using the example problem."""
    print("=" * 80)
    print("cuOpt Portfolio Optimizer Demo")
    print("Using NVIDIA cuOpt (GPU-accelerated optimization)")
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

    optimizer = CuOptPortfolioOptimizer(
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

    # --- MILP Path ---
    print("\n" + "=" * 80)
    print("Solving with cuOpt MILP (binary variables, exact)...")
    print("=" * 80)

    result_milp = optimizer.solve_milp()

    print(f"\n  Selected Assets: {result_milp['selected_assets']}")
    print(f"  Energy: {result_milp['energy']:.2f}")
    print(f"  Total Score: {result_milp['total_score']}")
    print(f"  Total Price: ${result_milp['total_price']:.2f}")
    print(f"  Total Duration: {result_milp['total_duration']}")
    print(f"  Is Feasible: {result_milp['is_feasible']}")

    # --- QP Path ---
    print("\n" + "=" * 80)
    print(
        "Solving with cuOpt QP "
        "(QUBO matrix, continuous relaxation)..."
    )
    print("=" * 80)

    result_qp = optimizer.solve_qp()
    continuous_qp = optimizer.get_continuous_solution()

    print(f"\n  Continuous solution: {continuous_qp}")
    print(f"  Selected Assets: {result_qp['selected_assets']}")
    print(f"  Energy: {result_qp['energy']:.2f}")
    print(f"  Total Score: {result_qp['total_score']}")
    print(f"  Total Price: ${result_qp['total_price']:.2f}")
    print(f"  Total Duration: {result_qp['total_duration']}")
    print(f"  Is Feasible: {result_qp['is_feasible']}")


if __name__ == "__main__":
    main()
