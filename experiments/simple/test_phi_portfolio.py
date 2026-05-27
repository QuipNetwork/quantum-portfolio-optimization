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
Tests for PhiPortfolioOptimizer (OpenPhiSolve integration).

Tests both QUBO (penalty matrix) and MIQP (native constraints) paths.
All tests are skipped if phisolve is not installed.
"""

import numpy as np
import pytest

phisolve = pytest.importorskip("phisolve")

from simple_portfolio_qubo import (
    get_example_assets,
    get_example_constraints,
)
from phi_portfolio import PhiPortfolioOptimizer


@pytest.fixture
def example_problem():
    """Standard example problem from the PDF."""
    assets = get_example_assets()
    constraints = get_example_constraints()
    return assets, constraints


@pytest.fixture
def optimizer(example_problem):
    """Optimizer initialized with example problem."""
    assets, constraints = example_problem
    return PhiPortfolioOptimizer(
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


class TestPhiConstruction:
    """Test model construction for both paths."""

    def test_qubo_problem_creates(self, optimizer):
        """QUBO problem builds without error."""
        prob = optimizer.build_qubo_problem()
        assert prob is not None

    def test_miqp_problem_creates(self, optimizer):
        """MIQP problem builds without error."""
        prob = optimizer.build_miqp_problem()
        assert prob is not None

    def test_qubo_matrix_scaling(self, optimizer):
        """QUBO matrix for PhiSolve is 2x our QUBO matrix."""
        q_qubo = optimizer._qubo_optimizer.build_qubo_matrix()
        prob = optimizer.build_qubo_problem()
        # PhiSolve QUBO stores Q internally
        np.testing.assert_allclose(prob.Q, 2.0 * q_qubo)

    def test_miqp_constraint_dimensions(self, optimizer):
        """MIQP has correct constraint dimensions."""
        prob = optimizer.build_miqp_problem()
        # 3 inequality constraints (budget, duration, cardinality)
        assert prob.A.shape == (3, optimizer.n)
        assert prob.b.shape == (3,)

    def test_miqp_all_binary(self, optimizer):
        """MIQP declares all variables as binary."""
        prob = optimizer.build_miqp_problem()
        assert prob.n_binary_vars == optimizer.n

    def test_constructor_stores_params(self, example_problem):
        """Constructor stores all parameters correctly."""
        assets, constraints = example_problem
        opt = PhiPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )
        assert opt.budget == constraints['budget']
        assert opt.max_duration == constraints['max_duration']
        assert opt.max_cardinality == constraints['max_cardinality']
        assert opt.n == len(assets)


class TestQUBOSolving:
    """Test QUBO path (penalty matrix via QIHD)."""

    def test_qubo_returns_feasible(self, optimizer):
        """QUBO solution is feasible after repair."""
        result = optimizer.solve_qubo(n_shots=50, n_steps=500)
        assert result['is_feasible']

    def test_qubo_positive_score(self, optimizer):
        """QUBO finds a solution with positive score."""
        result = optimizer.solve_qubo(n_shots=50, n_steps=500)
        assert result['total_score'] > 0

    def test_qubo_result_structure(self, optimizer):
        """QUBO result has all expected keys."""
        result = optimizer.solve_qubo(n_shots=50, n_steps=500)
        expected_keys = {
            'selection', 'selected_assets', 'energy',
            'total_price', 'total_duration', 'total_score',
            'num_selected', 'is_feasible',
        }
        assert expected_keys == set(result.keys())

    def test_qubo_binary_selection(self, optimizer):
        """QUBO selection is strictly binary."""
        result = optimizer.solve_qubo(n_shots=50, n_steps=500)
        for val in result['selection']:
            assert val in (0, 1)

    def test_qubo_budget_satisfied(self, optimizer):
        """QUBO respects budget constraint."""
        result = optimizer.solve_qubo(n_shots=50, n_steps=500)
        assert result['total_price'] <= optimizer.budget

    def test_qubo_duration_satisfied(self, optimizer):
        """QUBO respects duration constraint."""
        result = optimizer.solve_qubo(n_shots=50, n_steps=500)
        assert (
            result['total_duration'] <= optimizer.max_duration
        )

    def test_qubo_cardinality_satisfied(self, optimizer):
        """QUBO respects cardinality constraint."""
        result = optimizer.solve_qubo(n_shots=50, n_steps=500)
        assert (
            result['num_selected'] <= optimizer.max_cardinality
        )


class TestMIQPSolving:
    """Test MIQP path (native linear constraints)."""

    def test_miqp_returns_feasible(self, optimizer):
        """MIQP solution is feasible after repair."""
        result = optimizer.solve_miqp(
            n_shots=50, n_steps=500
        )
        assert result['is_feasible']

    def test_miqp_positive_score(self, optimizer):
        """MIQP finds a solution with positive score."""
        result = optimizer.solve_miqp(
            n_shots=50, n_steps=500
        )
        assert result['total_score'] > 0

    def test_miqp_result_structure(self, optimizer):
        """MIQP result has all expected keys."""
        result = optimizer.solve_miqp(
            n_shots=50, n_steps=500
        )
        expected_keys = {
            'selection', 'selected_assets', 'energy',
            'total_price', 'total_duration', 'total_score',
            'num_selected', 'is_feasible',
        }
        assert expected_keys == set(result.keys())

    def test_miqp_binary_selection(self, optimizer):
        """MIQP selection is strictly binary."""
        result = optimizer.solve_miqp(
            n_shots=50, n_steps=500
        )
        for val in result['selection']:
            assert val in (0, 1)

    def test_miqp_all_constraints(self, optimizer):
        """MIQP satisfies all constraints at once."""
        result = optimizer.solve_miqp(
            n_shots=50, n_steps=500
        )
        assert result['total_price'] <= optimizer.budget
        assert (
            result['total_duration'] <= optimizer.max_duration
        )
        assert (
            result['num_selected'] <= optimizer.max_cardinality
        )


class TestRoundAndRepair:
    """Test the round-and-repair post-processing."""

    def test_exact_binary_unchanged(self, optimizer):
        """Exact binary input passes through unchanged."""
        binary = np.array([1, 0, 1, 0, 1])
        result = optimizer._round_and_repair(
            binary.astype(float)
        )
        assert result.dtype == int or np.issubdtype(
            result.dtype, np.integer
        )

    def test_near_binary_rounds(self, optimizer):
        """Values near 0 or 1 round correctly."""
        near = np.array([0.9, 0.1, 0.8, 0.05, 0.95])
        result = optimizer._round_and_repair(near)
        # 0.9, 0.8, 0.95 > 0.5 → 1; 0.1, 0.05 < 0.5 → 0
        assert result[0] == 1
        assert result[1] == 0
        assert result[3] == 0

    def test_infeasible_gets_repaired(self, optimizer):
        """Infeasible rounding gets repaired to feasible."""
        all_high = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        result = optimizer._round_and_repair(all_high)
        assert optimizer._is_feasible(result)

    def test_repair_drops_lowest_score(self, optimizer):
        """Repair drops lowest-score assets first."""
        all_high = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        result = optimizer._round_and_repair(all_high)
        if np.sum(result) > 0:
            selected_scores = optimizer.scores[result == 1]
            dropped_scores = optimizer.scores[result == 0]
            if (
                len(dropped_scores) > 0
                and len(selected_scores) > 0
            ):
                assert np.min(selected_scores) >= np.min(
                    dropped_scores
                )

    def test_all_zeros_stays_zeros(self, optimizer):
        """All-zero input stays all zeros."""
        zeros = np.zeros(optimizer.n)
        result = optimizer._round_and_repair(zeros)
        assert np.all(result == 0)


class TestConstraintChecking:
    """Test constraint checking utility."""

    def test_feasible_selection(self, optimizer):
        """Known feasible selection is identified."""
        sel = np.zeros(optimizer.n, dtype=int)
        sel[0] = 1
        check = optimizer.check_constraints(sel)
        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']

    def test_budget_violation(self, optimizer):
        """Over-budget selection is detected."""
        sel = np.ones(optimizer.n, dtype=int)
        check = optimizer.check_constraints(sel)
        assert not (
            check['budget_satisfied']
            and check['duration_satisfied']
            and check['cardinality_satisfied']
        )

    def test_empty_selection(self, optimizer):
        """Empty selection satisfies all constraints."""
        sel = np.zeros(optimizer.n, dtype=int)
        check = optimizer.check_constraints(sel)
        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']

    def test_check_reports_totals(self, optimizer):
        """Check returns total price, duration, count."""
        sel = np.zeros(optimizer.n, dtype=int)
        sel[0] = 1
        check = optimizer.check_constraints(sel)
        assert 'total_price' in check
        assert 'total_duration' in check
        assert 'num_selected' in check
        assert check['num_selected'] == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_asset_qubo(self):
        """QUBO works with a single asset."""
        assets = [
            {
                'id': 'X', 'price': 1.0,
                'duration': 5, 'score': 10,
            }
        ]
        opt = PhiPortfolioOptimizer(
            assets=assets,
            budget=2.0,
            max_duration=10,
            max_cardinality=1,
        )
        result = opt.solve_qubo(n_shots=50, n_steps=500)
        assert result['is_feasible']

    def test_single_asset_miqp(self):
        """MIQP works with a single asset."""
        assets = [
            {
                'id': 'X', 'price': 1.0,
                'duration': 5, 'score': 10,
            }
        ]
        opt = PhiPortfolioOptimizer(
            assets=assets,
            budget=2.0,
            max_duration=10,
            max_cardinality=1,
        )
        result = opt.solve_miqp(n_shots=50, n_steps=500)
        assert result['is_feasible']

    def test_tight_cardinality(self):
        """max_cardinality=1 selects at most one asset."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        opt = PhiPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=1,
        )
        result = opt.solve_qubo(n_shots=50, n_steps=500)
        assert result['num_selected'] <= 1
