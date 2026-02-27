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
NL (Nonlinear Model) Portfolio Optimizer — Stride Hybrid Solver.

Uses D-Wave's dwave-optimization package to formulate the portfolio
optimization problem as a nonlinear model with native constraint support.
This model can be solved locally via brute-force enumeration or submitted
to D-Wave's Stride hybrid solver via LeapHybridNLSampler.

Key advantages:
- Native binary/integer variables and constraints
- Supports up to 2M variables on the Stride cloud solver
- Tensor-based DAG formulation (flexible, nonlinear objectives)
- Same constraint semantics as CQM (true inequality <=)

Solvers available:
- solve_exact(): Local brute-force enumeration (small problems only, ~20 vars)
- solve_nl(): Stride hybrid solver via LeapHybridNLSampler (requires Leap credentials)

Usage:
    from nl_portfolio import NLPortfolioOptimizer

    assets = [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        ...
    ]

    optimizer = NLPortfolioOptimizer(
        assets=assets,
        budget=3.0,
        max_duration=15,
        max_cardinality=3
    )

    # Exact solution (small problems)
    result = optimizer.solve_exact()

    # Stride hybrid solver (requires Leap credentials)
    result = optimizer.solve_nl(time_limit=5)
"""

import numpy as np
from dataclasses import dataclass
from itertools import product
from typing import List, Dict, Any, Optional

from dwave.optimization import Model


@dataclass
class Asset:
    """Represents an asset with its properties."""
    id: str
    price: float
    duration: float
    score: float


class NLPortfolioOptimizer:
    """
    NL Model portfolio optimizer using D-Wave's dwave-optimization package.

    Formulates the portfolio selection problem as a nonlinear model with
    binary decision variables and native inequality constraints, suitable
    for D-Wave's Stride hybrid solver.
    """

    def __init__(
        self,
        assets: List[Dict[str, Any]],
        budget: float,
        max_duration: float,
        max_cardinality: int
    ):
        """
        Initialize the NL portfolio optimizer.

        Args:
            assets: List of asset dictionaries with keys: 'id', 'price', 'duration', 'score'
            budget: Maximum total price constraint (<=)
            max_duration: Maximum total duration constraint (<=)
            max_cardinality: Maximum number of assets to select (<=)
        """
        self.assets = [Asset(**a) for a in assets]
        self.n = len(self.assets)
        self.budget = budget
        self.max_duration = max_duration
        self.max_cardinality = max_cardinality

        # Extract arrays for metrics calculation
        self.prices = np.array([a.price for a in self.assets])
        self.durations = np.array([a.duration for a in self.assets])
        self.scores = np.array([a.score for a in self.assets])

        # Cache for model
        self._model: Optional[Model] = None
        self._x = None  # Binary decision variable symbol

    def build_model(self) -> Model:
        """
        Build a dwave-optimization nonlinear model.

        Returns:
            dwave.optimization.Model with objective and constraints
        """
        model = Model()

        # Constants (negate scores for minimization)
        neg_scores = model.constant(-self.scores)
        prices = model.constant(self.prices)
        durations = model.constant(self.durations)

        # Binary decision variables: x[i] = 1 if asset i is selected
        x = model.binary(self.n)

        # Objective: minimize negative score (= maximize score)
        model.minimize((neg_scores * x).sum())

        # Budget constraint: Σ price_i * x_i <= budget
        model.add_constraint((prices * x).sum() <= self.budget)

        # Duration constraint: Σ duration_i * x_i <= max_duration
        model.add_constraint((durations * x).sum() <= self.max_duration)

        # Cardinality constraint: Σ x_i <= max_cardinality
        model.add_constraint(x.sum() <= self.max_cardinality)

        self._model = model
        self._x = x
        return model

    def _build_result(self, selection: np.ndarray, energy: float, is_feasible: bool) -> Dict[str, Any]:
        """
        Build a result dictionary from a selection vector.

        Args:
            selection: Binary selection vector
            energy: Objective value (negative score)
            is_feasible: Whether the solution satisfies all constraints

        Returns:
            Standard result dictionary
        """
        selected_indices = np.where(selection == 1)[0]
        selected_assets = [self.assets[i].id for i in selected_indices]

        total_price = float(np.sum(self.prices[selected_indices])) if len(selected_indices) > 0 else 0.0
        total_duration = float(np.sum(self.durations[selected_indices])) if len(selected_indices) > 0 else 0.0
        total_score = float(np.sum(self.scores[selected_indices])) if len(selected_indices) > 0 else 0.0

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

    def solve_exact(self) -> Dict[str, Any]:
        """
        Solve by brute-force enumeration of all 2^n binary states.

        Sets each possible binary assignment on the model's state,
        evaluates feasibility and objective, and returns the best
        feasible solution.

        Only suitable for small problems (~20 variables or less).

        Returns:
            Dictionary with selection, metrics, and constraint satisfaction
        """
        if self._model is None:
            self.build_model()

        model = self._model
        model.lock()

        decisions = list(model.iter_decisions())
        x_var = decisions[0]

        model.states.resize(1)

        best_selection = None
        best_energy = float('inf')

        # Enumerate all 2^n binary assignments
        for bits in product([0.0, 1.0], repeat=self.n):
            state = np.array(bits)
            x_var.set_state(0, state)

            if not model.feasible(0):
                continue

            energy = float(model.objective.state(0))
            if energy < best_energy:
                best_energy = energy
                best_selection = state.copy()

        model.unlock()

        # If no feasible solution found (shouldn't happen — empty set is always feasible)
        if best_selection is None:
            best_selection = np.zeros(self.n)
            best_energy = 0.0

        return self._build_result(best_selection, best_energy, is_feasible=True)

    def solve_nl(self, time_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Solve using D-Wave's Stride hybrid solver via LeapHybridNLSampler.

        Requires a valid Leap API token configured in the environment
        (DWAVE_API_TOKEN or dwave config file).

        Args:
            time_limit: Maximum solver runtime in seconds. If None, uses
                the solver's minimum estimated time.

        Returns:
            Dictionary with selection, metrics, and constraint satisfaction

        Raises:
            ImportError: If dwave-system is not installed
            RuntimeError: If Leap credentials are not configured
        """
        try:
            from dwave.system import LeapHybridNLSampler
        except ImportError:
            raise ImportError(
                "dwave-system is required for the Stride solver. "
                "Install it with: pip install dwave-system"
            )

        if self._model is None:
            self.build_model()

        sampler = LeapHybridNLSampler()

        kwargs = {}
        if time_limit is not None:
            kwargs['time_limit'] = time_limit

        sampler.sample(self._model, **kwargs)

        # Extract the best solution from the model's states
        decisions = list(self._model.iter_decisions())
        x_var = decisions[0]

        num_samples = self._model.states.size()

        best_selection = None
        best_energy = float('inf')
        best_feasible_selection = None
        best_feasible_energy = float('inf')

        for i in range(num_samples):
            state = np.array(x_var.state(i))
            energy = float(self._model.objective.state(i))
            feasible = self._model.feasible(i)

            if energy < best_energy:
                best_energy = energy
                best_selection = state.copy()

            if feasible and energy < best_feasible_energy:
                best_feasible_energy = energy
                best_feasible_selection = state.copy()

        # Prefer feasible solution
        if best_feasible_selection is not None:
            return self._build_result(best_feasible_selection, best_feasible_energy, is_feasible=True)
        elif best_selection is not None:
            return self._build_result(best_selection, best_energy, is_feasible=False)
        else:
            # Fallback: empty selection
            return self._build_result(np.zeros(self.n), 0.0, is_feasible=True)

    def check_constraints(self, selection: np.ndarray) -> Dict[str, Any]:
        """
        Check if a selection satisfies all constraints.

        Args:
            selection: Binary selection vector

        Returns:
            Dictionary with constraint satisfaction status
        """
        x = np.asarray(selection)
        selected_indices = np.where(x == 1)[0]

        total_price = float(np.sum(self.prices[selected_indices])) if len(selected_indices) > 0 else 0.0
        total_duration = float(np.sum(self.durations[selected_indices])) if len(selected_indices) > 0 else 0.0
        num_selected = len(selected_indices)

        return {
            'budget_satisfied': total_price <= self.budget,
            'duration_satisfied': total_duration <= self.max_duration,
            'cardinality_satisfied': num_selected <= self.max_cardinality,
            'total_price': total_price,
            'total_duration': total_duration,
            'num_selected': num_selected,
            'budget_limit': self.budget,
            'duration_limit': self.max_duration,
            'cardinality_limit': self.max_cardinality
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the NL model formulation.

        Returns:
            Dictionary with model statistics
        """
        if self._model is None:
            self.build_model()

        model = self._model
        model.lock()
        num_decisions = sum(1 for _ in model.iter_decisions())
        num_constraints = sum(1 for _ in model.iter_constraints())
        model.unlock()

        return {
            'num_assets': self.n,
            'num_decisions': num_decisions,
            'num_constraints': num_constraints,
            'budget': self.budget,
            'max_duration': self.max_duration,
            'max_cardinality': self.max_cardinality,
        }


def get_example_assets() -> List[Dict[str, Any]]:
    """Returns the 5-asset example for testing."""
    return [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        {'id': 'C', 'price': 0.89, 'duration': 4, 'score': 5},
        {'id': 'D', 'price': 1.05, 'duration': 3, 'score': 1},
        {'id': 'E', 'price': 1.02, 'duration': 9, 'score': 9},
    ]


def get_example_constraints() -> Dict[str, Any]:
    """Returns the constraint values for the example problem."""
    return {
        'budget': 3.0,
        'max_duration': 15,
        'max_cardinality': 3,
    }


def main():
    """Demo using the example problem."""
    print("=" * 80)
    print("NL Model Portfolio Optimizer Demo")
    print("Using D-Wave's dwave-optimization with Stride hybrid solver support")
    print("=" * 80)

    # Create optimizer with example problem
    assets = get_example_assets()
    constraints = get_example_constraints()

    print("\nAsset Data:")
    print("-" * 50)
    for a in assets:
        print(f"  {a['id']}: price={a['price']:.2f}, duration={a['duration']}, score={a['score']}")

    print("\nConstraints (all are <= inequalities):")
    print(f"  Budget: <= ${constraints['budget']:.2f}")
    print(f"  Max Duration: <= {constraints['max_duration']}")
    print(f"  Max Cardinality: <= {constraints['max_cardinality']}")

    optimizer = NLPortfolioOptimizer(
        assets=assets,
        **constraints
    )

    # Build and display model info
    optimizer.build_model()
    info = optimizer.get_model_info()
    print(f"\nNL Model Info:")
    print(f"  Assets: {info['num_assets']}")
    print(f"  Decision variables: {info['num_decisions']}")
    print(f"  Constraints: {info['num_constraints']}")

    # Solve using brute-force enumeration
    print("\n" + "=" * 80)
    print("Solving with brute-force enumeration (exact solution)...")
    print("=" * 80)

    result_exact = optimizer.solve_exact()

    print(f"\nExact Solution:")
    print(f"  Selected Assets: {result_exact['selected_assets']}")
    print(f"  Energy: {result_exact['energy']:.2f}")
    print(f"  Total Price: ${result_exact['total_price']:.2f}")
    print(f"  Total Duration: {result_exact['total_duration']}")
    print(f"  Total Score: {result_exact['total_score']}")
    print(f"  Num Selected: {result_exact['num_selected']}")
    print(f"  Is Feasible: {result_exact['is_feasible']}")

    # Check constraints
    constraint_check = optimizer.check_constraints(result_exact['selection'])
    print(f"\nConstraint Satisfaction:")
    print(f"  Budget (max ${constraint_check['budget_limit']:.2f}): "
          f"${constraint_check['total_price']:.2f} - "
          f"{'SATISFIED' if constraint_check['budget_satisfied'] else 'VIOLATED'}")
    print(f"  Duration (max {constraint_check['duration_limit']}): "
          f"{constraint_check['total_duration']} - "
          f"{'SATISFIED' if constraint_check['duration_satisfied'] else 'VIOLATED'}")
    print(f"  Cardinality (max {constraint_check['cardinality_limit']}): "
          f"{constraint_check['num_selected']} - "
          f"{'SATISFIED' if constraint_check['cardinality_satisfied'] else 'VIOLATED'}")

    # Try Stride solver if available
    print("\n" + "=" * 80)
    print("Attempting Stride hybrid solver via LeapHybridNLSampler...")
    print("=" * 80)

    try:
        result_nl = optimizer.solve_nl()
        print(f"\nStride Solution:")
        print(f"  Selected Assets: {result_nl['selected_assets']}")
        print(f"  Energy: {result_nl['energy']:.2f}")
        print(f"  Total Price: ${result_nl['total_price']:.2f}")
        print(f"  Total Duration: {result_nl['total_duration']}")
        print(f"  Total Score: {result_nl['total_score']}")
        print(f"  Num Selected: {result_nl['num_selected']}")
        print(f"  Is Feasible: {result_nl['is_feasible']}")

        # Compare with exact
        gap = 100.0 * (result_exact['total_score'] - result_nl['total_score']) / result_exact['total_score'] if result_exact['total_score'] > 0 else 0.0
        print(f"\nComparison:")
        print(f"  Exact score: {result_exact['total_score']}")
        print(f"  Stride score: {result_nl['total_score']}")
        print(f"  Gap: {gap:.2f}%")
    except (ImportError, Exception) as e:
        print(f"\n  Stride solver not available: {e}")
        print("  Install dwave-system and configure Leap credentials to use the Stride solver.")


if __name__ == "__main__":
    main()
