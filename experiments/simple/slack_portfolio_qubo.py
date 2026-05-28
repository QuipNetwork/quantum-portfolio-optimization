#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Slack-Variable Portfolio QUBO Optimizer.

Extends the simple portfolio QUBO formulation to support true inequality constraints
using slack variables. This eliminates the penalty for under-budget/under-duration
solutions that exists in the equality-based formulation.

Inequality Constraint Encoding:
    For constraint: Σ p_i x_i ≤ B

    1. Introduce slack variable s ≥ 0 encoded in binary:
       s = Σ_{k=0}^{K} 2^k · s_k

    2. Convert to equality:
       Σ p_i x_i + s = B

    3. Penalty term becomes:
       λ · (Σ p_i x_i + s - B)²

    This has zero penalty when sum ≤ B (slack absorbs the difference).

Trade-offs vs Equality Formulation:
    + True inequality semantics (no under-budget penalty)
    + Better alignment with practical constraints
    - More qubits needed (log₂(max_slack) per constraint)
    - Larger QUBO matrix
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


class SlackPortfolioQUBO:
    """
    QUBO-based portfolio optimizer with slack variables for inequality constraints.

    Uses slack variables to encode ≤ constraints:
    - Budget: Σ p_i x_i ≤ B
    - Duration: Σ d_i x_i ≤ D
    - Cardinality: Σ x_i ≤ K
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
        budget_precision: float = 0.1,
        duration_precision: float = 1.0
    ):
        """
        Initialize the portfolio optimizer with slack variables.

        Args:
            assets: List of asset dictionaries with keys: 'id', 'price', 'duration', 'score'
            budget: Maximum total price constraint (≤)
            max_duration: Maximum total duration constraint (≤)
            max_cardinality: Maximum number of assets to select (≤)
            lambda_budget: Penalty weight for budget constraint
            lambda_duration: Penalty weight for duration constraint
            lambda_cardinality: Penalty weight for cardinality constraint
            budget_precision: Precision for budget slack encoding
            duration_precision: Precision for duration slack encoding
        """
        self.assets = [Asset(**a) for a in assets]
        self.n = len(self.assets)
        self.budget = budget
        self.max_duration = max_duration
        self.max_cardinality = max_cardinality
        self.lambda_b = lambda_budget
        self.lambda_d = lambda_duration
        self.lambda_c = lambda_cardinality
        self.budget_precision = budget_precision
        self.duration_precision = duration_precision

        # Extract arrays for vectorized operations
        self.prices = np.array([a.price for a in self.assets])
        self.durations = np.array([a.duration for a in self.assets])
        self.scores = np.array([a.score for a in self.assets])

        # Calculate slack variable sizes
        # Budget slack: need to represent values from 0 to budget
        self.budget_slack_bits = self._calc_slack_bits(budget, budget_precision)
        # Duration slack: need to represent values from 0 to max_duration
        self.duration_slack_bits = self._calc_slack_bits(max_duration, duration_precision)
        # Cardinality slack: need to represent values from 0 to max_cardinality
        self.cardinality_slack_bits = self._calc_slack_bits(max_cardinality, 1)

        # Total variables: n assets + slack bits for each constraint
        self.n_slack = (self.budget_slack_bits +
                       self.duration_slack_bits +
                       self.cardinality_slack_bits)
        self.n_total = self.n + self.n_slack

        # Variable index ranges
        self.asset_range = (0, self.n)
        self.budget_slack_range = (self.n, self.n + self.budget_slack_bits)
        self.duration_slack_range = (self.n + self.budget_slack_bits,
                                     self.n + self.budget_slack_bits + self.duration_slack_bits)
        self.cardinality_slack_range = (self.n + self.budget_slack_bits + self.duration_slack_bits,
                                        self.n_total)

        # Cache
        self._Q: Optional[np.ndarray] = None

    def _calc_slack_bits(self, max_value: float, precision: float) -> int:
        """Calculate number of bits needed to encode slack up to max_value."""
        max_slack = int(np.ceil(max_value / precision))
        if max_slack <= 0:
            return 1
        return int(np.ceil(np.log2(max_slack + 1)))

    def _get_slack_coefficients(self, n_bits: int, precision: float) -> np.ndarray:
        """Get the coefficients for binary-encoded slack: [1, 2, 4, 8, ...] * precision."""
        return np.array([precision * (2 ** k) for k in range(n_bits)])

    def get_variable_info(self) -> Dict[str, Any]:
        """Get information about the variable structure."""
        return {
            'n_assets': self.n,
            'n_budget_slack': self.budget_slack_bits,
            'n_duration_slack': self.duration_slack_bits,
            'n_cardinality_slack': self.cardinality_slack_bits,
            'n_total': self.n_total,
            'asset_range': self.asset_range,
            'budget_slack_range': self.budget_slack_range,
            'duration_slack_range': self.duration_slack_range,
            'cardinality_slack_range': self.cardinality_slack_range,
            'budget_precision': self.budget_precision,
            'duration_precision': self.duration_precision
        }

    def build_qubo_matrix(self) -> np.ndarray:
        """
        Build the QUBO matrix Q with slack variables.

        The QUBO encodes:
        - Objective: maximize score (minimize -score)
        - Budget constraint: Σ p_i x_i + s_b = B
        - Duration constraint: Σ d_i x_i + s_d = D
        - Cardinality constraint: Σ x_i + s_c = K

        Returns:
            Q: n_total x n_total symmetric QUBO matrix
        """
        n_total = self.n_total
        Q = np.zeros((n_total, n_total))

        # Get slack coefficients
        budget_slack_coef = self._get_slack_coefficients(
            self.budget_slack_bits, self.budget_precision)
        duration_slack_coef = self._get_slack_coefficients(
            self.duration_slack_bits, self.duration_precision)
        cardinality_slack_coef = self._get_slack_coefficients(
            self.cardinality_slack_bits, 1)

        # ===== OBJECTIVE: -score for each asset =====
        for i in range(self.n):
            Q[i, i] -= self.scores[i]

        # ===== BUDGET CONSTRAINT: (Σ p_i x_i + s_b - B)² =====
        # Expand: Σ p_i² x_i + 2 Σ_{i<j} p_i p_j x_i x_j
        #       + 2 Σ_i Σ_k p_i s_bk · c_k x_i
        #       + Σ_k c_k² s_bk + 2 Σ_{k<l} c_k c_l s_bk s_bl
        #       - 2B Σ p_i x_i - 2B Σ_k c_k s_bk + B²

        lb = self.lambda_b
        B = self.budget

        # Diagonal for assets: p_i² - 2B·p_i
        for i in range(self.n):
            Q[i, i] += lb * (self.prices[i]**2 - 2 * B * self.prices[i])

        # Off-diagonal for asset pairs: 2·p_i·p_j
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[i, j] += 2 * lb * self.prices[i] * self.prices[j]
                Q[j, i] = Q[i, j]

        # Budget slack diagonal: c_k² - 2B·c_k
        for k, c_k in enumerate(budget_slack_coef):
            idx = self.budget_slack_range[0] + k
            Q[idx, idx] += lb * (c_k**2 - 2 * B * c_k)

        # Budget slack off-diagonal: 2·c_k·c_l
        for k in range(len(budget_slack_coef)):
            for l in range(k + 1, len(budget_slack_coef)):
                idx_k = self.budget_slack_range[0] + k
                idx_l = self.budget_slack_range[0] + l
                Q[idx_k, idx_l] += 2 * lb * budget_slack_coef[k] * budget_slack_coef[l]
                Q[idx_l, idx_k] = Q[idx_k, idx_l]

        # Cross terms: asset × budget slack: 2·p_i·c_k
        for i in range(self.n):
            for k, c_k in enumerate(budget_slack_coef):
                idx_k = self.budget_slack_range[0] + k
                Q[i, idx_k] += 2 * lb * self.prices[i] * c_k
                Q[idx_k, i] = Q[i, idx_k]

        # ===== DURATION CONSTRAINT: (Σ d_i x_i + s_d - D)² =====
        ld = self.lambda_d
        D = self.max_duration

        # Diagonal for assets: d_i² - 2D·d_i
        for i in range(self.n):
            Q[i, i] += ld * (self.durations[i]**2 - 2 * D * self.durations[i])

        # Off-diagonal for asset pairs: 2·d_i·d_j
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[i, j] += 2 * ld * self.durations[i] * self.durations[j]
                Q[j, i] = Q[i, j]

        # Duration slack diagonal: c_k² - 2D·c_k
        for k, c_k in enumerate(duration_slack_coef):
            idx = self.duration_slack_range[0] + k
            Q[idx, idx] += ld * (c_k**2 - 2 * D * c_k)

        # Duration slack off-diagonal: 2·c_k·c_l
        for k in range(len(duration_slack_coef)):
            for l in range(k + 1, len(duration_slack_coef)):
                idx_k = self.duration_slack_range[0] + k
                idx_l = self.duration_slack_range[0] + l
                Q[idx_k, idx_l] += 2 * ld * duration_slack_coef[k] * duration_slack_coef[l]
                Q[idx_l, idx_k] = Q[idx_k, idx_l]

        # Cross terms: asset × duration slack: 2·d_i·c_k
        for i in range(self.n):
            for k, c_k in enumerate(duration_slack_coef):
                idx_k = self.duration_slack_range[0] + k
                Q[i, idx_k] += 2 * ld * self.durations[i] * c_k
                Q[idx_k, i] = Q[i, idx_k]

        # ===== CARDINALITY CONSTRAINT: (Σ x_i + s_c - K)² =====
        lc = self.lambda_c
        K = self.max_cardinality

        # Diagonal for assets: 1 - 2K
        for i in range(self.n):
            Q[i, i] += lc * (1 - 2 * K)

        # Off-diagonal for asset pairs: 2
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[i, j] += 2 * lc
                Q[j, i] = Q[i, j]

        # Cardinality slack diagonal: c_k² - 2K·c_k
        for k, c_k in enumerate(cardinality_slack_coef):
            idx = self.cardinality_slack_range[0] + k
            Q[idx, idx] += lc * (c_k**2 - 2 * K * c_k)

        # Cardinality slack off-diagonal: 2·c_k·c_l
        for k in range(len(cardinality_slack_coef)):
            for l in range(k + 1, len(cardinality_slack_coef)):
                idx_k = self.cardinality_slack_range[0] + k
                idx_l = self.cardinality_slack_range[0] + l
                Q[idx_k, idx_l] += 2 * lc * cardinality_slack_coef[k] * cardinality_slack_coef[l]
                Q[idx_l, idx_k] = Q[idx_k, idx_l]

        # Cross terms: asset × cardinality slack: 2·1·c_k = 2·c_k
        for i in range(self.n):
            for k, c_k in enumerate(cardinality_slack_coef):
                idx_k = self.cardinality_slack_range[0] + k
                Q[i, idx_k] += 2 * lc * c_k
                Q[idx_k, i] = Q[i, idx_k]

        self._Q = Q
        return Q

    def evaluate_energy(self, full_selection: np.ndarray) -> float:
        """
        Evaluate the QUBO energy for a given selection vector (including slack).

        Args:
            full_selection: Binary vector of length n_total

        Returns:
            Energy E(x) = x^T Q x (upper triangular form)
        """
        if self._Q is None:
            self.build_qubo_matrix()

        x = np.asarray(full_selection)
        Q = self._Q

        energy = 0.0
        for i in range(self.n_total):
            energy += Q[i, i] * x[i]
            for j in range(i + 1, self.n_total):
                energy += Q[i, j] * x[i] * x[j]

        return energy

    def decode_slack(self, full_selection: np.ndarray) -> Dict[str, float]:
        """Decode the slack variable values from a full selection vector."""
        x = np.asarray(full_selection)

        # Budget slack
        budget_slack_coef = self._get_slack_coefficients(
            self.budget_slack_bits, self.budget_precision)
        budget_slack_bits = x[self.budget_slack_range[0]:self.budget_slack_range[1]]
        budget_slack = np.dot(budget_slack_bits, budget_slack_coef)

        # Duration slack
        duration_slack_coef = self._get_slack_coefficients(
            self.duration_slack_bits, self.duration_precision)
        duration_slack_bits = x[self.duration_slack_range[0]:self.duration_slack_range[1]]
        duration_slack = np.dot(duration_slack_bits, duration_slack_coef)

        # Cardinality slack
        cardinality_slack_coef = self._get_slack_coefficients(
            self.cardinality_slack_bits, 1)
        cardinality_slack_bits = x[self.cardinality_slack_range[0]:self.cardinality_slack_range[1]]
        cardinality_slack = np.dot(cardinality_slack_bits, cardinality_slack_coef)

        return {
            'budget_slack': budget_slack,
            'duration_slack': duration_slack,
            'cardinality_slack': cardinality_slack
        }

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        """Convert to dimod BinaryQuadraticModel."""
        if self._Q is None:
            self.build_qubo_matrix()

        Q = self._Q

        # Create variable names
        var_names = []
        for a in self.assets:
            var_names.append(a.id)
        for k in range(self.budget_slack_bits):
            var_names.append(f'sb{k}')
        for k in range(self.duration_slack_bits):
            var_names.append(f'sd{k}')
        for k in range(self.cardinality_slack_bits):
            var_names.append(f'sc{k}')

        # Build QUBO dict
        qubo_dict = {}
        for i in range(self.n_total):
            qubo_dict[(var_names[i], var_names[i])] = Q[i, i]
            for j in range(i + 1, self.n_total):
                if Q[i, j] != 0:
                    qubo_dict[(var_names[i], var_names[j])] = Q[i, j]

        return dimod.BinaryQuadraticModel.from_qubo(qubo_dict)

    def solve(
        self,
        num_reads: int = 1000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Solve using simulated annealing.

        Returns:
            Dictionary with solution details including slack values
        """
        bqm = self.to_bqm()
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)

        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy

        # Extract asset selection
        asset_selection = np.array([best_sample[a.id] for a in self.assets])
        selected_indices = np.where(asset_selection == 1)[0]
        selected_assets = [self.assets[i].id for i in selected_indices]

        # Extract full selection for slack decoding
        var_names = [a.id for a in self.assets]
        for k in range(self.budget_slack_bits):
            var_names.append(f'sb{k}')
        for k in range(self.duration_slack_bits):
            var_names.append(f'sd{k}')
        for k in range(self.cardinality_slack_bits):
            var_names.append(f'sc{k}')

        full_selection = np.array([best_sample[v] for v in var_names])
        slack_values = self.decode_slack(full_selection)

        # Compute metrics
        total_price = np.sum(self.prices[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_duration = np.sum(self.durations[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_score = np.sum(self.scores[selected_indices]) if len(selected_indices) > 0 else 0.0

        return {
            'selection': asset_selection,
            'selected_assets': selected_assets,
            'energy': best_energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_assets),
            'slack_values': slack_values,
            'sampleset': sampleset
        }

    def solve_qpu(
        self,
        num_reads: int = 1000,
        solver: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Solve on a D-Wave QPU via DWaveCliqueSampler.

        Submits the slack-variable BQM (assets + budget/duration/
        cardinality slack bits) to real quantum-annealing hardware.
        DWaveCliqueSampler handles the dense embedding; coefficients
        are auto-scaled into the QPU's h/J ranges (auto_scale=True).

        Requires D-Wave Leap credentials (DWAVE_API_TOKEN via env or
        dwave.conf); solver defaults to DWAVE_API_SOLVER or
        'Advantage2_system1.6'. Note the slack encoding uses more
        qubits than the simple QUBO, so larger problems may exceed
        the largest clique the QPU can embed.

        Args:
            num_reads: Number of QPU anneals.
            solver: Explicit solver name. Defaults to the
                DWAVE_API_SOLVER env var, then 'Advantage2_system1.6'.

        Returns:
            Same result dict shape as solve(), including slack_values
            and the QPU SampleSet.
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

        asset_selection = np.array([best_sample[a.id] for a in self.assets])
        selected_indices = np.where(asset_selection == 1)[0]
        selected_assets = [self.assets[i].id for i in selected_indices]

        var_names = [a.id for a in self.assets]
        for k in range(self.budget_slack_bits):
            var_names.append(f'sb{k}')
        for k in range(self.duration_slack_bits):
            var_names.append(f'sd{k}')
        for k in range(self.cardinality_slack_bits):
            var_names.append(f'sc{k}')

        full_selection = np.array([best_sample[v] for v in var_names])
        slack_values = self.decode_slack(full_selection)

        total_price = np.sum(self.prices[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_duration = np.sum(self.durations[selected_indices]) if len(selected_indices) > 0 else 0.0
        total_score = np.sum(self.scores[selected_indices]) if len(selected_indices) > 0 else 0.0

        # QPU timing (us -> ms); see SimplePortfolioQUBO.solve_qpu.
        timing = sampleset.info.get('timing', {})
        qpu_access_us = timing.get('qpu_access_time')
        qpu_anneal_us = timing.get('qpu_anneal_time_per_sample')
        qpu_access_ms = qpu_access_us / 1000.0 if qpu_access_us is not None else None
        qpu_anneal_ms = qpu_anneal_us / 1000.0 if qpu_anneal_us is not None else None

        return {
            'selection': asset_selection,
            'selected_assets': selected_assets,
            'energy': best_energy,
            'total_price': total_price,
            'total_duration': total_duration,
            'total_score': total_score,
            'num_selected': len(selected_assets),
            'slack_values': slack_values,
            'qpu_access_ms': qpu_access_ms,
            'qpu_anneal_ms': qpu_anneal_ms,
            'sampleset': sampleset
        }

    def check_constraints(self, asset_selection: np.ndarray) -> Dict[str, Any]:
        """Check if an asset selection satisfies all constraints."""
        x = np.asarray(asset_selection)
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


def main():
    """Demo using the example problem with slack variables."""
    print("=" * 80)
    print("Slack-Variable Portfolio QUBO Optimizer Demo")
    print("=" * 80)

    assets = [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        {'id': 'C', 'price': 0.89, 'duration': 4, 'score': 5},
        {'id': 'D', 'price': 1.05, 'duration': 3, 'score': 1},
        {'id': 'E', 'price': 1.02, 'duration': 9, 'score': 9},
    ]

    optimizer = SlackPortfolioQUBO(
        assets=assets,
        budget=3.0,
        max_duration=15,
        max_cardinality=3,
        lambda_budget=2.0,
        lambda_duration=10.0,
        lambda_cardinality=5.0,
        budget_precision=0.1,
        duration_precision=1.0
    )

    var_info = optimizer.get_variable_info()
    print(f"\nVariable Structure:")
    print(f"  Assets: {var_info['n_assets']}")
    print(f"  Budget slack bits: {var_info['n_budget_slack']}")
    print(f"  Duration slack bits: {var_info['n_duration_slack']}")
    print(f"  Cardinality slack bits: {var_info['n_cardinality_slack']}")
    print(f"  Total variables: {var_info['n_total']}")

    print("\nSolving with Simulated Annealing...")
    result = optimizer.solve(num_reads=1000, seed=42)

    print(f"\nBest Solution:")
    print(f"  Selected Assets: {result['selected_assets']}")
    print(f"  Energy: {result['energy']:.2f}")
    print(f"  Total Price: ${result['total_price']:.2f}")
    print(f"  Total Duration: {result['total_duration']}")
    print(f"  Total Score: {result['total_score']}")

    print(f"\nSlack Values:")
    print(f"  Budget slack: {result['slack_values']['budget_slack']:.2f}")
    print(f"  Duration slack: {result['slack_values']['duration_slack']:.1f}")
    print(f"  Cardinality slack: {result['slack_values']['cardinality_slack']:.0f}")

    check = optimizer.check_constraints(result['selection'])
    print(f"\nConstraint Satisfaction:")
    print(f"  Budget (≤${check['budget_limit']:.2f}): ${check['total_price']:.2f} - "
          f"{'SATISFIED' if check['budget_satisfied'] else 'VIOLATED'}")
    print(f"  Duration (≤{check['duration_limit']}): {check['total_duration']:.0f} - "
          f"{'SATISFIED' if check['duration_satisfied'] else 'VIOLATED'}")
    print(f"  Cardinality (≤{check['cardinality_limit']}): {check['num_selected']} - "
          f"{'SATISFIED' if check['cardinality_satisfied'] else 'VIOLATED'}")


if __name__ == "__main__":
    main()
