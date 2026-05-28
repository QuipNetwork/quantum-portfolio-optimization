#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Simple Portfolio QUBO Optimizer.

Implements the QUBO matrix construction for portfolio optimization.

This formulation selects assets to maximize score while respecting:
- Budget constraint (total price = B)
- Duration constraint (total duration = D)
- Cardinality constraint (max K assets)

QUBO Matrix Construction:
    Diagonal: Q[i,i] = -score_i + lambda_b(p_i^2 - 2*B*p_i)
                                + lambda_d(d_i^2 - 2*D*d_i)
                                + lambda_c(1 - 2*K)

    Off-diagonal (i < j): Q[i,j] = 2*lambda_b*p_i*p_j
                                 + 2*lambda_d*d_i*d_j
                                 + 2*lambda_c

    Matrix is symmetric: Q[j,i] = Q[i,j]

Usage:
    from tools.simple_portfolio_qubo import SimplePortfolioQUBO

    # Define asset data
    assets = [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        ...
    ]

    # Create optimizer with constraints
    optimizer = SimplePortfolioQUBO(
        assets=assets,
        budget=3.0,
        max_duration=15,
        max_cardinality=3,
        lambda_budget=2.0,
        lambda_duration=10.0,
        lambda_cardinality=5.0
    )

    # Build QUBO matrix
    Q = optimizer.build_qubo_matrix()

    # Solve using simulated annealing (default)
    result = optimizer.solve(num_reads=1000)
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import dimod
import neal


@dataclass
class Asset:
    """Represents an asset with its properties."""
    id: str
    price: float
    duration: float
    score: float


class SimplePortfolioQUBO:
    """
    QUBO-based portfolio optimizer for asset selection.

    Constructs a QUBO matrix that encodes the portfolio optimization problem
    with budget, duration, and cardinality constraints.
    """

    def __init__(
        self,
        assets: List[Dict[str, Any]],
        budget: float,
        max_duration: float,
        max_cardinality: int,
        lambda_budget: float = 2.0,
        lambda_duration: float = 10.0,
        lambda_cardinality: float = 5.0
    ):
        """
        Initialize the portfolio optimizer.

        Args:
            assets: List of asset dictionaries with keys: 'id', 'price', 'duration', 'score'
            budget: Target total price constraint
            max_duration: Maximum total duration constraint
            max_cardinality: Maximum number of assets to select
            lambda_budget: Penalty weight for budget constraint
            lambda_duration: Penalty weight for duration constraint
            lambda_cardinality: Penalty weight for cardinality constraint
        """
        self.assets = [Asset(**a) for a in assets]
        self.n = len(self.assets)
        self.budget = budget
        self.max_duration = max_duration
        self.max_cardinality = max_cardinality
        self.lambda_b = lambda_budget
        self.lambda_d = lambda_duration
        self.lambda_c = lambda_cardinality

        # Extract arrays for vectorized operations
        self.prices = np.array([a.price for a in self.assets])
        self.durations = np.array([a.duration for a in self.assets])
        self.scores = np.array([a.score for a in self.assets])

        # Cache for QUBO matrix
        self._Q: Optional[np.ndarray] = None
        self._h: Optional[np.ndarray] = None
        self._J: Optional[np.ndarray] = None

    def build_qubo_matrix(self) -> np.ndarray:
        """
        Build the QUBO matrix Q.

        Returns:
            Q: n x n symmetric QUBO matrix

        The diagonal entries encode individual asset contributions:
            Q[i,i] = -s_i + lambda_b(p_i^2 - 2*B*p_i)
                         + lambda_d(d_i^2 - 2*D*d_i)
                         + lambda_c(1 - 2*K)

        The off-diagonal entries encode pairwise interactions:
            Q[i,j] = 2*lambda_b*p_i*p_j + 2*lambda_d*d_i*d_j + 2*lambda_c  (for i < j)
        """
        n = self.n
        Q = np.zeros((n, n))

        # Diagonal entries
        # Q[i,i] = -score + lambda_b*(price^2 - 2*budget*price)
        #                 + lambda_d*(duration^2 - 2*max_duration*duration)
        #                 + lambda_c*(1 - 2*max_cardinality)
        Q[np.diag_indices(n)] = (
            -self.scores
            + self.lambda_b * (self.prices**2 - 2 * self.budget * self.prices)
            + self.lambda_d * (self.durations**2 - 2 * self.max_duration * self.durations)
            + self.lambda_c * (1 - 2 * self.max_cardinality)
        )

        # Off-diagonal entries (upper triangle, then mirror)
        # Q[i,j] = 2*lambda_b*p_i*p_j + 2*lambda_d*d_i*d_j + 2*lambda_c
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = (
                    2 * self.lambda_b * self.prices[i] * self.prices[j]
                    + 2 * self.lambda_d * self.durations[i] * self.durations[j]
                    + 2 * self.lambda_c
                )
                Q[j, i] = Q[i, j]  # Symmetric

        self._Q = Q
        return Q

    def qubo_to_ising(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert QUBO matrix to Ising model parameters (h, J).

        The transformation from QUBO (x in {0,1}) to Ising (z in {-1,+1}) uses:
            x_i = (1 + z_i) / 2

        Returns:
            h: Local field coefficients (n,)
            J: Coupling coefficients (n x n), symmetric with zero diagonal

        Ising energy: E(z) = sum_i h_i*z_i + sum_{i<j} J_ij*z_i*z_j
        """
        if self._Q is None:
            self.build_qubo_matrix()

        Q = self._Q
        n = self.n

        J = np.zeros((n, n))
        h = np.zeros(n)

        # Coupling coefficients: J[i,j] = Q[i,j] / 4
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = Q[i, j] / 4
                J[j, i] = J[i, j]

        # Local field coefficients: h[i] = Q[i,i]/2 + sum_{j!=i}(Q[i,j])/4
        for i in range(n):
            off_diagonal_sum = np.sum(Q[i, :]) - Q[i, i]
            h[i] = Q[i, i] / 2 + off_diagonal_sum / 4

        self._h = h
        self._J = J
        return h, J

    def normalize_for_qpu(
        self,
        h_range: Tuple[float, float] = (-6.0, 6.0),
        j_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Normalize Ising parameters to fit within D-Wave QPU limits.

        D-Wave QPU coefficient limits (per their API):
            Advantage (Pegasus):  h in [-4, 4],  J in [-1, 1]
            Advantage2 (Zephyr):  h in [-6, 6],  J in [-1, 1]

        NOTE: there's some finite precision (I think around 5 bits of resolution)
        when mapped to QPU hardware, but the ranges are continuous.
            
        We scales h and J to fit within the specified ranges while
        preserving the relative relationships between coefficients.

        Args:
            h_range: Target range for h values, default (-6.0, 6.0) for Advantage2
            j_range: Target range for J values, default (-1.0, 1.0)

        Returns:
            h_normalized: Scaled h vector
            J_normalized: Scaled J matrix
            scale_factor: The factor by which values were scaled (for energy conversion)

        Note:
            The scale factor can be used to convert normalized energies back to
            original scale: original_energy = normalized_energy / scale_factor
        """
        h, J = self.qubo_to_ising()

        # Find the maximum absolute values
        h_max = np.max(np.abs(h)) if len(h) > 0 else 1.0
        J_nonzero = J[J != 0]
        j_max = np.max(np.abs(J_nonzero)) if len(J_nonzero) > 0 else 1.0

        # Calculate scale factors needed to fit each range
        # h needs to fit in h_range
        h_limit = min(abs(h_range[0]), abs(h_range[1]))
        h_scale = h_limit / h_max if h_max > 0 else 1.0

        # J needs to fit in j_range
        j_limit = min(abs(j_range[0]), abs(j_range[1]))
        j_scale = j_limit / j_max if j_max > 0 else 1.0

        # Use the more restrictive scale factor to ensure both fit
        scale_factor = min(h_scale, j_scale)

        # Apply scaling
        h_normalized = h * scale_factor
        J_normalized = J * scale_factor

        return h_normalized, J_normalized, scale_factor

    def get_normalization_info(
        self,
        h_range: Tuple[float, float] = (-6.0, 6.0),
        j_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> Dict[str, Any]:
        """
        Get detailed information about the normalization process.

        Args:
            h_range: Target range for h values
            j_range: Target range for J values

        Returns:
            Dictionary with normalization statistics
        """
        h, J = self.qubo_to_ising()
        h_norm, J_norm, scale = self.normalize_for_qpu(h_range, j_range)

        J_nonzero = J[J != 0]
        J_norm_nonzero = J_norm[J_norm != 0]

        return {
            'original_h_range': (float(np.min(h)), float(np.max(h))),
            'original_j_range': (float(np.min(J_nonzero)), float(np.max(J_nonzero))) if len(J_nonzero) > 0 else (0, 0),
            'normalized_h_range': (float(np.min(h_norm)), float(np.max(h_norm))),
            'normalized_j_range': (float(np.min(J_norm_nonzero)), float(np.max(J_norm_nonzero))) if len(J_norm_nonzero) > 0 else (0, 0),
            'target_h_range': h_range,
            'target_j_range': j_range,
            'scale_factor': scale,
            'compression_ratio': 1.0 / scale if scale > 0 else float('inf'),
            'h_utilization': np.max(np.abs(h_norm)) / min(abs(h_range[0]), abs(h_range[1])),
            'j_utilization': np.max(np.abs(J_norm_nonzero)) / min(abs(j_range[0]), abs(j_range[1])) if len(J_norm_nonzero) > 0 else 0,
        }

    def to_bqm_normalized(
        self,
        h_range: Tuple[float, float] = (-6.0, 6.0),
        j_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> Tuple[dimod.BinaryQuadraticModel, float]:
        """
        Convert to normalized BQM suitable for D-Wave QPU submission.

        Args:
            h_range: Target range for h values
            j_range: Target range for J values

        Returns:
            Tuple of (normalized BQM, scale_factor)

        Example:
            bqm, scale = optimizer.to_bqm_normalized()
            # Submit to QPU...
            # Convert energy back: original_energy = qpu_energy / scale
        """
        h_norm, J_norm, scale_factor = self.normalize_for_qpu(h_range, j_range)

        # Build BQM from normalized Ising parameters
        linear = {self.assets[i].id: h_norm[i] for i in range(self.n)}
        quadratic = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if J_norm[i, j] != 0:
                    quadratic[(self.assets[i].id, self.assets[j].id)] = J_norm[i, j]

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.SPIN)

        return bqm, scale_factor

    def evaluate_energy(self, selection: np.ndarray) -> float:
        """
        Evaluate the QUBO energy for a given selection vector.

        Args:
            selection: Binary vector x in {0,1}^n where x_i=1 means asset i is selected

        Returns:
            Energy E(x) = sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j
        """
        if self._Q is None:
            self.build_qubo_matrix()

        x = np.asarray(selection)
        Q = self._Q

        # E(x) = sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j
        # Using matrix form: E(x) = x^T * Q_upper * x where Q_upper has zeros below diagonal
        # But since Q is symmetric and we want upper triangle only for off-diagonal:
        # E = sum of diagonal terms + sum of upper triangle interactions

        energy = 0.0

        # Diagonal terms
        for i in range(self.n):
            energy += Q[i, i] * x[i]

        # Off-diagonal terms (upper triangle only)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                energy += Q[i, j] * x[i] * x[j]

        return energy

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        """
        Convert to dimod BinaryQuadraticModel for use with D-Wave solvers.

        Returns:
            BQM with asset IDs as variable names
        """
        if self._Q is None:
            self.build_qubo_matrix()

        Q = self._Q

        # Create QUBO dictionary with asset IDs as keys
        qubo_dict = {}

        # Diagonal (linear) terms
        for i, asset in enumerate(self.assets):
            qubo_dict[(asset.id, asset.id)] = Q[i, i]

        # Off-diagonal (quadratic) terms - upper triangle only
        for i in range(self.n):
            for j in range(i + 1, self.n):
                qubo_dict[(self.assets[i].id, self.assets[j].id)] = Q[i, j]

        return dimod.BinaryQuadraticModel.from_qubo(qubo_dict)

    def solve(
        self,
        num_reads: int = 1000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Solve using simulated annealing (default solver).

        Args:
            num_reads: Number of annealing runs
            seed: Random seed for reproducibility

        Returns:
            Dictionary with:
                - 'selection': Binary selection vector
                - 'selected_assets': List of selected asset IDs
                - 'energy': QUBO energy of best solution
                - 'total_price': Sum of selected asset prices
                - 'total_duration': Sum of selected asset durations
                - 'total_score': Sum of selected asset scores
                - 'num_selected': Number of assets selected
                - 'sampleset': Full dimod SampleSet for analysis
        """
        bqm = self.to_bqm()

        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)

        # Get best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy

        # Convert to selection vector
        selection = np.array([best_sample[asset.id] for asset in self.assets])
        selected_indices = np.where(selection == 1)[0]
        selected_assets = [self.assets[i].id for i in selected_indices]

        # Compute metrics for selected assets
        total_price = np.sum(self.prices[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_duration = np.sum(self.durations[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_score = np.sum(self.scores[selected_indices]) if len(selected_indices) > 0 else 0.0

        return {
            'selection': selection,
            'selected_assets': selected_assets,
            'energy': best_energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_assets),
            'sampleset': sampleset
        }

    def solve_qpu(
        self,
        num_reads: int = 1000,
        solver: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Solve on a D-Wave QPU via DWaveCliqueSampler.

        Submits the same BQM that solve() runs through simulated
        annealing to real quantum-annealing hardware. The QUBO penalty
        matrix is fully connected (every asset couples to every other),
        so DWaveCliqueSampler — which uses pre-computed clique
        embeddings — is the natural fit. Coefficients are auto-scaled
        into the QPU's h/J ranges by the sampler (auto_scale=True).

        Requires D-Wave Leap credentials. The token is read from
        D-Wave's standard config chain (DWAVE_API_TOKEN env var or
        dwave.conf); the solver name defaults to DWAVE_API_SOLVER or
        'Advantage2_system1.6'.

        Args:
            num_reads: Number of QPU anneals.
            solver: Explicit solver name. Defaults to the
                DWAVE_API_SOLVER env var, then 'Advantage2_system1.6'.

        Returns:
            Same result dict shape as solve(), with 'sampleset' being
            the QPU SampleSet (includes timing info in .info).
        """
        from dwave.system import DWaveCliqueSampler

        bqm = self.to_bqm()
        solver_name = solver or os.environ.get(
            'DWAVE_API_SOLVER', 'Advantage2_system1.6'
        )
        sampler = DWaveCliqueSampler(solver=solver_name)
        sampleset = sampler.sample(bqm, num_reads=num_reads)

        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy

        selection = np.array([best_sample[asset.id] for asset in self.assets])
        selected_indices = np.where(selection == 1)[0]
        selected_assets = [self.assets[i].id for i in selected_indices]

        total_price = np.sum(self.prices[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_duration = np.sum(self.durations[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_score = np.sum(self.scores[selected_indices]) if len(selected_indices) > 0 else 0.0

        # Actual QPU time (microseconds) reported by the sampler, converted
        # to ms. This is the hardware cost — far smaller than wall-clock,
        # which is dominated by network round-trip and Leap queue time.
        qpu_access_us = sampleset.info.get('timing', {}).get('qpu_access_time')
        qpu_access_ms = qpu_access_us / 1000.0 if qpu_access_us is not None else None

        return {
            'selection': selection,
            'selected_assets': selected_assets,
            'energy': best_energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_assets),
            'qpu_access_ms': qpu_access_ms,
            'sampleset': sampleset
        }

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

        total_price = np.sum(self.prices[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_duration = np.sum(self.durations[selected_indices]) if len(selected_indices) > 0 else 0.0
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

    def print_qubo_matrix(self, precision: int = 2) -> None:
        """Print the QUBO matrix in a formatted way."""
        if self._Q is None:
            self.build_qubo_matrix()

        Q = self._Q
        asset_ids = [a.id for a in self.assets]

        print("\nQUBO Matrix Q:")
        print("     " + "  ".join(f"{aid:>10}" for aid in asset_ids))
        for i, aid in enumerate(asset_ids):
            row = "  ".join(f"{Q[i,j]:>10.{precision}f}" for j in range(self.n))
            print(f"{aid:>4} {row}")

    def print_ising_model(self, precision: int = 2) -> None:
        """Print the Ising model parameters."""
        h, J = self.qubo_to_ising()
        asset_ids = [a.id for a in self.assets]

        print("\nIsing h vector (local fields):")
        for i, aid in enumerate(asset_ids):
            print(f"  h[{aid}] = {h[i]:.{precision}f}")

        print("\nIsing J matrix (couplings):")
        print("     " + "  ".join(f"{aid:>10}" for aid in asset_ids))
        for i, aid in enumerate(asset_ids):
            row = "  ".join(f"{J[i,j]:>10.{precision}f}" for j in range(self.n))
            print(f"{aid:>4} {row}")


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
        'lambda_budget': 2.0,
        'lambda_duration': 10.0,
        'lambda_cardinality': 5.0
    }


def main():
    """Demo using the example problem."""
    print("=" * 80)
    print("Simple Portfolio QUBO Optimizer Demo")
    print("=" * 80)

    # Create optimizer with example problem
    assets = get_example_assets()
    constraints = get_example_constraints()

    print("\nAsset Data:")
    print("-" * 50)
    for a in assets:
        print(f"  {a['id']}: price={a['price']:.2f}, duration={a['duration']}, score={a['score']}")

    print("\nConstraints:")
    print(f"  Budget: ${constraints['budget']:.2f}")
    print(f"  Max Duration: {constraints['max_duration']}")
    print(f"  Max Cardinality: {constraints['max_cardinality']}")
    print(f"  Lambda (budget): {constraints['lambda_budget']}")
    print(f"  Lambda (duration): {constraints['lambda_duration']}")
    print(f"  Lambda (cardinality): {constraints['lambda_cardinality']}")

    optimizer = SimplePortfolioQUBO(
        assets=assets,
        **constraints
    )

    # Build and display QUBO matrix
    Q = optimizer.build_qubo_matrix()
    optimizer.print_qubo_matrix()

    # Verify Q[0,0] matches expected value
    print(f"\nVerification: Q[0,0] = {Q[0,0]:.2f} (expected -1483.00)")
    print(f"Verification: Q[0,1] = {Q[0,1]:.2f} (expected 853.96)")

    # Convert to Ising and display
    optimizer.print_ising_model()

    # Test energy calculation for selection [1,0,1,0,0]
    test_selection = np.array([1, 0, 1, 0, 0])
    energy = optimizer.evaluate_energy(test_selection)
    print(f"\nEnergy for selection [A, C]: {energy:.2f} (expected -2032.26)")

    # Solve using simulated annealing
    print("\n" + "=" * 80)
    print("Solving with Simulated Annealing...")
    print("=" * 80)

    result = optimizer.solve(num_reads=1000, seed=42)

    print(f"\nBest Solution:")
    print(f"  Selected Assets: {result['selected_assets']}")
    print(f"  Energy: {result['energy']:.2f}")
    print(f"  Total Price: ${result['total_price']:.2f}")
    print(f"  Total Duration: {result['total_duration']}")
    print(f"  Total Score: {result['total_score']}")
    print(f"  Num Selected: {result['num_selected']}")

    # Check constraints
    constraint_check = optimizer.check_constraints(result['selection'])
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


if __name__ == "__main__":
    main()
