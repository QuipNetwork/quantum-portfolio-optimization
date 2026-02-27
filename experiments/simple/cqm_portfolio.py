#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
CQM (Constrained Quadratic Model) Portfolio Optimizer.

Uses D-Wave's CQM formulation with native inequality constraint support.
This provides a cleaner API compared to manual QUBO construction and
automatic penalty tuning when converting to BQM.

Key advantages over manual QUBO:
- Native <= / >= / == constraint support
- No under-budget penalty (true inequality semantics)
- Automatic lagrange multiplier tuning
- Cleaner, declarative syntax

Solvers available:
- ExactCQMSolver: Brute force enumeration (small problems only, ~20 vars)
- cqm_to_bqm + SA: Convert to BQM with penalties, solve with simulated annealing

Usage:
    from cqm_portfolio import CQMPortfolioOptimizer

    assets = [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        ...
    ]

    optimizer = CQMPortfolioOptimizer(
        assets=assets,
        budget=3.0,
        max_duration=15,
        max_cardinality=3
    )

    # Exact solution (small problems)
    result = optimizer.solve_exact()

    # Simulated annealing via CQM->BQM conversion
    result = optimizer.solve_sa(num_reads=1000)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import dimod
from dimod import ConstrainedQuadraticModel, Binary, ExactCQMSolver, cqm_to_bqm
import neal


@dataclass
class Asset:
    """Represents an asset with its properties."""
    id: str
    price: float
    duration: float
    score: float


class CQMPortfolioOptimizer:
    """
    CQM-based portfolio optimizer with native inequality constraints.

    Uses D-Wave's ConstrainedQuadraticModel to formulate the problem with
    true inequality constraints (<=) rather than soft equality penalties.
    """

    def __init__(
        self,
        assets: List[Dict[str, Any]],
        budget: float,
        max_duration: float,
        max_cardinality: int
    ):
        """
        Initialize the CQM portfolio optimizer.

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

        # Cache for CQM
        self._cqm: Optional[ConstrainedQuadraticModel] = None
        self._variables: Optional[Dict[str, Any]] = None

    def build_cqm(self) -> ConstrainedQuadraticModel:
        """
        Build CQM with native inequality constraints.

        Returns:
            ConstrainedQuadraticModel with objective and constraints
        """
        cqm = ConstrainedQuadraticModel()

        # Binary variables for asset selection
        x = {a.id: Binary(a.id) for a in self.assets}
        self._variables = x

        # Objective: maximize score (negate for minimization)
        # CQM minimizes, so we negate to maximize score
        objective = -sum(a.score * x[a.id] for a in self.assets)
        cqm.set_objective(objective)

        # Budget constraint: Σ price_i * x_i <= budget
        budget_expr = sum(a.price * x[a.id] for a in self.assets)
        cqm.add_constraint(budget_expr <= self.budget, label='budget')

        # Duration constraint: Σ duration_i * x_i <= max_duration
        duration_expr = sum(a.duration * x[a.id] for a in self.assets)
        cqm.add_constraint(duration_expr <= self.max_duration, label='duration')

        # Cardinality constraint: Σ x_i <= max_cardinality
        cardinality_expr = sum(x[a.id] for a in self.assets)
        cqm.add_constraint(cardinality_expr <= self.max_cardinality, label='cardinality')

        self._cqm = cqm
        return cqm

    def _parse_result(self, sampleset: dimod.SampleSet) -> Dict[str, Any]:
        """
        Parse sampleset into result dictionary.

        Args:
            sampleset: dimod SampleSet from solver

        Returns:
            Dictionary with selection, metrics, and constraint satisfaction
        """
        # Get best feasible sample, or best overall if none feasible
        feasible_samples = sampleset.filter(lambda s: s.is_feasible)
        if len(feasible_samples) > 0:
            best_sample = feasible_samples.first.sample
            best_energy = feasible_samples.first.energy
            is_feasible = True
        else:
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy
            is_feasible = False

        # Convert to selection vector
        selection = np.array([best_sample.get(a.id, 0) for a in self.assets])
        selected_indices = np.where(selection == 1)[0]
        selected_assets = [self.assets[i].id for i in selected_indices]

        # Compute metrics for selected assets
        total_price = float(np.sum(self.prices[selected_indices])) if len(selected_indices) > 0 else 0.0
        total_duration = float(np.sum(self.durations[selected_indices])) if len(selected_indices) > 0 else 0.0
        total_score = float(np.sum(self.scores[selected_indices])) if len(selected_indices) > 0 else 0.0

        return {
            'selection': selection,
            'selected_assets': selected_assets,
            'energy': best_energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_assets),
            'is_feasible': is_feasible,
            'sampleset': sampleset
        }

    def solve_exact(self) -> Dict[str, Any]:
        """
        Solve using ExactCQMSolver (small problems only).

        This solver enumerates all possible combinations, so it's only
        suitable for small problems (~20 variables or less).

        Returns:
            Dictionary with selection, metrics, and constraint satisfaction
        """
        if self._cqm is None:
            self.build_cqm()

        sampleset = ExactCQMSolver().sample_cqm(self._cqm)
        return self._parse_result(sampleset)

    def solve_sa(
        self,
        num_reads: int = 1000,
        lagrange_multiplier: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Solve via CQM->BQM conversion + Simulated Annealing.

        Converts the CQM to a BQM by encoding constraints as penalties,
        then solves with simulated annealing.

        Args:
            num_reads: Number of annealing samples
            lagrange_multiplier: Penalty strength for constraints.
                Defaults to 10x the largest bias in the objective.
            seed: Random seed for reproducibility

        Returns:
            Dictionary with selection, metrics, and constraint satisfaction
        """
        if self._cqm is None:
            self.build_cqm()

        # Convert CQM to BQM
        bqm, inverter = cqm_to_bqm(self._cqm, lagrange_multiplier=lagrange_multiplier)

        # Solve with simulated annealing
        sampler = neal.SimulatedAnnealingSampler()
        bqm_sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)

        # Invert samples to get CQM variable values
        # The inverter works on individual samples, not SampleSet
        best_sample = None
        best_energy = float('inf')
        best_feasible = None
        best_feasible_energy = float('inf')

        for sample, energy in zip(bqm_sampleset.samples(), bqm_sampleset.data_vectors['energy']):
            # Convert BQM sample to CQM variables
            cqm_sample = inverter(sample)

            # Extract selection for original asset variables
            selection = np.array([cqm_sample.get(a.id, 0) for a in self.assets])

            # Check feasibility
            check = self.check_constraints(selection)
            is_feasible = (check['budget_satisfied'] and
                          check['duration_satisfied'] and
                          check['cardinality_satisfied'])

            # Track best overall and best feasible
            if energy < best_energy:
                best_energy = energy
                best_sample = cqm_sample

            if is_feasible and energy < best_feasible_energy:
                best_feasible_energy = energy
                best_feasible = cqm_sample

        # Prefer feasible solution if available
        if best_feasible is not None:
            final_sample = best_feasible
            final_energy = best_feasible_energy
            is_feasible = True
        else:
            final_sample = best_sample
            final_energy = best_energy
            is_feasible = False

        # Build result from final sample
        selection = np.array([final_sample.get(a.id, 0) for a in self.assets])
        selected_indices = np.where(selection == 1)[0]
        selected_assets = [self.assets[i].id for i in selected_indices]

        total_price = float(np.sum(self.prices[selected_indices])) if len(selected_indices) > 0 else 0.0
        total_duration = float(np.sum(self.durations[selected_indices])) if len(selected_indices) > 0 else 0.0
        total_score = float(np.sum(self.scores[selected_indices])) if len(selected_indices) > 0 else 0.0

        return {
            'selection': selection,
            'selected_assets': selected_assets,
            'energy': final_energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_assets),
            'is_feasible': is_feasible,
            'sampleset': bqm_sampleset  # Original BQM sampleset for reference
        }

    def to_bqm(
        self,
        lagrange_multiplier: Optional[float] = None
    ) -> tuple:
        """
        Convert to BQM for debugging/inspection.

        Args:
            lagrange_multiplier: Penalty strength for constraints.
                Defaults to 10x the largest bias in the objective.

        Returns:
            Tuple of (BinaryQuadraticModel, inverter function)
        """
        if self._cqm is None:
            self.build_cqm()

        return cqm_to_bqm(self._cqm, lagrange_multiplier=lagrange_multiplier)

    def to_cqm(self) -> ConstrainedQuadraticModel:
        """
        Get the CQM for inspection or custom solving.

        Returns:
            ConstrainedQuadraticModel
        """
        if self._cqm is None:
            self.build_cqm()
        return self._cqm

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

    def get_cqm_info(self) -> Dict[str, Any]:
        """
        Get information about the CQM formulation.

        Returns:
            Dictionary with CQM statistics
        """
        if self._cqm is None:
            self.build_cqm()

        cqm = self._cqm
        return {
            'num_variables': cqm.num_variables(),
            'num_constraints': cqm.num_constraints(),
            'constraint_labels': list(cqm.constraints.keys()),
            'variable_labels': list(cqm.variables),
        }


def get_example_assets() -> List[Dict[str, Any]]:
    """
    Returns the 5-asset example for testing.
    """
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
    print("CQM Portfolio Optimizer Demo")
    print("Using D-Wave's ConstrainedQuadraticModel with native inequality constraints")
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

    optimizer = CQMPortfolioOptimizer(
        assets=assets,
        **constraints
    )

    # Build and display CQM info
    cqm = optimizer.build_cqm()
    info = optimizer.get_cqm_info()
    print(f"\nCQM Info:")
    print(f"  Variables: {info['num_variables']}")
    print(f"  Constraints: {info['num_constraints']}")
    print(f"  Constraint labels: {info['constraint_labels']}")

    # Solve using ExactCQMSolver
    print("\n" + "=" * 80)
    print("Solving with ExactCQMSolver (brute force, exact solution)...")
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

    # Solve using SA via CQM->BQM
    print("\n" + "=" * 80)
    print("Solving with CQM->BQM + Simulated Annealing...")
    print("=" * 80)

    result_sa = optimizer.solve_sa(num_reads=1000, seed=42)

    print(f"\nSimulated Annealing Solution:")
    print(f"  Selected Assets: {result_sa['selected_assets']}")
    print(f"  Energy: {result_sa['energy']:.2f}")
    print(f"  Total Price: ${result_sa['total_price']:.2f}")
    print(f"  Total Duration: {result_sa['total_duration']}")
    print(f"  Total Score: {result_sa['total_score']}")
    print(f"  Num Selected: {result_sa['num_selected']}")
    print(f"  Is Feasible: {result_sa['is_feasible']}")

    # Compare with exact
    print(f"\nComparison:")
    gap = 100.0 * (result_exact['total_score'] - result_sa['total_score']) / result_exact['total_score'] if result_exact['total_score'] > 0 else 0.0
    print(f"  Exact score: {result_exact['total_score']}")
    print(f"  SA score: {result_sa['total_score']}")
    print(f"  Gap: {gap:.2f}%")

    # Inspect BQM from CQM conversion
    print("\n" + "=" * 80)
    print("BQM from CQM->BQM conversion (for debugging)")
    print("=" * 80)

    bqm, inverter = optimizer.to_bqm()
    print(f"\nBQM Info:")
    print(f"  Num variables: {bqm.num_variables}")
    print(f"  Num interactions: {bqm.num_interactions}")
    print(f"  Variable types: {set(bqm.vartype for _ in [1])}")

    # Show linear biases
    print(f"\nLinear biases (h):")
    for var, bias in sorted(bqm.linear.items()):
        print(f"  {var}: {bias:.4f}")


if __name__ == "__main__":
    main()
