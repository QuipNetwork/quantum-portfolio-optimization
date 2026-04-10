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
QHD (Quantum Hamiltonian Descent) Portfolio Optimizer.

Uses QHDOPT to solve the portfolio selection problem via continuous
relaxation of the binary QUBO formulation. Two formulation paths:

1. QP path: Reuses the QUBO penalty matrix via QHD.QP(). The same
   energy landscape as simulated annealing, solved with QHD instead.

2. SymPy path: Native symbolic formulation with an explicit binary
   enforcement penalty x_i(1-x_i). More natural for continuous
   optimization — competes with the NL (Stride) solver.

Both paths solve on [0,1]^n, then round to binary and repair if
infeasible. Available backends: classical (IPOPT/TNC), qutip
(quantum simulation), and D-Wave (quantum hardware).

Usage:
    from qhd_portfolio import QHDPortfolioOptimizer

    assets = [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        ...
    ]

    optimizer = QHDPortfolioOptimizer(
        assets=assets, budget=3.0, max_duration=15, max_cardinality=3
    )

    # QP path (reuses QUBO matrix)
    result = optimizer.solve_qp()

    # SymPy path (native formulation with binary enforcement)
    result = optimizer.solve_sympy()
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from simple_portfolio_qubo import (
    SimplePortfolioQUBO,
    get_example_assets,
    get_example_constraints,
)


@dataclass
class Asset:
    """Represents an asset with its properties."""
    id: str
    price: float
    duration: float
    score: float


class QHDPortfolioOptimizer:
    """
    Portfolio optimizer using QHDOPT (Quantum Hamiltonian Descent).

    Solves the binary asset selection problem by relaxing it to
    continuous optimization on [0,1]^n, then rounding to binary.
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
        lambda_binary: float = 10.0,
    ):
        """
        Initialize the QHD portfolio optimizer.

        Args:
            assets: List of asset dicts with 'id', 'price', 'duration', 'score'.
            budget: Maximum total price (<=).
            max_duration: Maximum total duration (<=).
            max_cardinality: Maximum number of assets (<=).
            lambda_budget: Penalty weight for budget constraint.
            lambda_duration: Penalty weight for duration constraint.
            lambda_cardinality: Penalty weight for cardinality constraint.
            lambda_binary: Penalty weight for binary enforcement (SymPy path).
        """
        self.assets = [Asset(**a) for a in assets]
        self.n = len(self.assets)
        self.budget = budget
        self.max_duration = max_duration
        self.max_cardinality = max_cardinality
        self.lambda_binary = lambda_binary

        self.prices = np.array([a.price for a in self.assets])
        self.durations = np.array([a.duration for a in self.assets])
        self.scores = np.array([a.score for a in self.assets])

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

        # Store last continuous solution for analysis
        self._last_continuous: Optional[np.ndarray] = None

    def build_qp_model(self):
        """
        Build a QHD model from the QUBO matrix via QP interface.

        QHDOPT QP convention: minimize (1/2) x^T Q x + b^T x.
        Our QUBO: minimize x^T Q_qubo x.
        Therefore: Q_qhd = 2 * Q_qubo, b = 0.

        Returns:
            QHD model instance ready for backend setup or solve.
        """
        from qhdopt import QHD

        q_qubo = self._qubo_optimizer.build_qubo_matrix()
        q_qhd = (2.0 * q_qubo).tolist()
        b = [0.0] * self.n
        return QHD.QP(q_qhd, b, bounds=(0, 1))

    def build_sympy_model(self):
        """
        Build a QHD model with explicit SymPy formulation.

        Constructs the objective symbolically with constraint penalties
        and a binary enforcement term lambda_bin * sum(x_i * (1 - x_i)).
        This term equals 0 at x=0 and x=1, pushing continuous solutions
        toward binary values.

        Returns:
            QHD model instance ready for backend setup or solve.
        """
        from qhdopt import QHD
        from sympy import symbols, Symbol

        x = symbols(f'x:{self.n}')

        # Objective: minimize negative score (maximize score)
        obj = sum(
            -self.scores[i] * x[i] for i in range(self.n)
        )

        # Budget penalty: lambda_b * (sum(price_i * x_i) - B)^2
        budget_sum = sum(
            self.prices[i] * x[i] for i in range(self.n)
        )
        budget_pen = self._qubo_optimizer.lambda_b * (
            budget_sum - self.budget
        ) ** 2

        # Duration penalty: lambda_d * (sum(dur_i * x_i) - D)^2
        duration_sum = sum(
            self.durations[i] * x[i] for i in range(self.n)
        )
        duration_pen = self._qubo_optimizer.lambda_d * (
            duration_sum - self.max_duration
        ) ** 2

        # Cardinality penalty: lambda_c * (sum(x_i) - K)^2
        count_sum = sum(x[i] for i in range(self.n))
        cardinality_pen = self._qubo_optimizer.lambda_c * (
            count_sum - self.max_cardinality
        ) ** 2

        # Binary enforcement: lambda_bin * sum(x_i * (1 - x_i))
        binary_pen = self.lambda_binary * sum(
            x[i] * (1 - x[i]) for i in range(self.n)
        )

        f = obj + budget_pen + duration_pen + cardinality_pen + binary_pen
        return QHD.SymPy(f, list(x), bounds=(0, 1))

    def _run_backend(self, model, backend, num_shots, resolution, **kwargs):
        """
        Configure and run a QHD backend.

        Args:
            model: QHD model instance.
            backend: One of "classical", "qutip", "dwave".
            num_shots: Number of initial samples (classical) or shots.
            resolution: Discretization resolution (qutip/dwave).
            **kwargs: Extra backend-specific parameters.

        Returns:
            numpy array of continuous solution values.
        """
        import jax
        jax.config.update('jax_enable_x64', True)

        if backend == "classical":
            solver = kwargs.get("solver", "IPOPT")
            model.classically_optimize(
                num_shots=num_shots, solver=solver
            )
        elif backend == "qutip":
            model.qutip_setup(
                resolution=resolution,
                shots=num_shots,
                embedding_scheme=kwargs.get(
                    "embedding_scheme", "onehot"
                ),
                time_discretization=kwargs.get(
                    "time_discretization", 10
                ),
                penalty_coefficient=kwargs.get(
                    "penalty_coefficient", 0
                ),
                post_processing_method=kwargs.get(
                    "post_processing_method", "TNC"
                ),
            )
            model.optimize(
                refine=kwargs.get("refine", True),
                verbose=kwargs.get("verbose", 0),
            )
        elif backend == "dwave":
            model.dwave_setup(
                resolution=resolution,
                shots=num_shots,
                api_key=kwargs.get("api_key"),
                penalty_coefficient=kwargs.get(
                    "penalty_coefficient", 0
                ),
                post_processing_method=kwargs.get(
                    "post_processing_method", "TNC"
                ),
            )
            model.optimize(
                refine=kwargs.get("refine", True),
                verbose=kwargs.get("verbose", 0),
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                f"Use 'classical', 'qutip', or 'dwave'."
            )

        return np.array(model.get_solution())

    def solve_qp(
        self,
        backend: str = "classical",
        num_shots: int = 100,
        resolution: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve using the QP path (QUBO matrix → QHD.QP).

        Args:
            backend: "classical", "qutip", or "dwave".
            num_shots: Number of optimization starts / shots.
            resolution: Discretization for quantum backends.
            **kwargs: Extra backend parameters.

        Returns:
            Standard result dict with selection, metrics, feasibility.
        """
        model = self.build_qp_model()
        continuous = self._run_backend(
            model, backend, num_shots, resolution, **kwargs
        )
        self._last_continuous = continuous

        selection = self._round_and_repair(continuous)
        energy = self._compute_energy(selection)
        is_feasible = self._is_feasible(selection)
        return self._build_result(selection, energy, is_feasible)

    def solve_sympy(
        self,
        backend: str = "classical",
        num_shots: int = 100,
        resolution: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve using the SymPy path (native formulation + binary penalty).

        Args:
            backend: "classical", "qutip", or "dwave".
            num_shots: Number of optimization starts / shots.
            resolution: Discretization for quantum backends.
            **kwargs: Extra backend parameters.

        Returns:
            Standard result dict with selection, metrics, feasibility.
        """
        model = self.build_sympy_model()
        continuous = self._run_backend(
            model, backend, num_shots, resolution, **kwargs
        )
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
        2. If infeasible, greedily drop lowest-score assets until feasible

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
        selected_assets = [
            self.assets[i].id for i in selected_indices
        ]

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
            'selected_assets': selected_assets,
            'energy': energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_assets),
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
    print("QHD Portfolio Optimizer Demo")
    print("Using QHDOPT (Quantum Hamiltonian Descent)")
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

    optimizer = QHDPortfolioOptimizer(
        assets=assets,
        budget=constraints['budget'],
        max_duration=constraints['max_duration'],
        max_cardinality=constraints['max_cardinality'],
        lambda_budget=constraints.get('lambda_budget', 2.0),
        lambda_duration=constraints.get('lambda_duration', 10.0),
        lambda_cardinality=constraints.get(
            'lambda_cardinality', 5.0
        ),
    )

    # --- QP Path ---
    print("\n" + "=" * 80)
    print("Solving with QHD-QP (QUBO matrix via QP interface)...")
    print("=" * 80)

    result_qp = optimizer.solve_qp(num_shots=50)
    continuous_qp = optimizer.get_continuous_solution()

    print(f"\n  Continuous solution: {continuous_qp}")
    print(f"  Selected Assets: {result_qp['selected_assets']}")
    print(f"  Energy: {result_qp['energy']:.2f}")
    print(f"  Total Score: {result_qp['total_score']}")
    print(f"  Total Price: ${result_qp['total_price']:.2f}")
    print(f"  Total Duration: {result_qp['total_duration']}")
    print(f"  Is Feasible: {result_qp['is_feasible']}")

    # --- SymPy Path ---
    print("\n" + "=" * 80)
    print(
        "Solving with QHD-SymPy "
        "(native formulation + binary enforcement)..."
    )
    print("=" * 80)

    result_sp = optimizer.solve_sympy(num_shots=50)
    continuous_sp = optimizer.get_continuous_solution()

    print(f"\n  Continuous solution: {continuous_sp}")
    print(f"  Selected Assets: {result_sp['selected_assets']}")
    print(f"  Energy: {result_sp['energy']:.2f}")
    print(f"  Total Score: {result_sp['total_score']}")
    print(f"  Total Price: ${result_sp['total_price']:.2f}")
    print(f"  Total Duration: {result_sp['total_duration']}")
    print(f"  Is Feasible: {result_sp['is_feasible']}")


if __name__ == "__main__":
    main()
