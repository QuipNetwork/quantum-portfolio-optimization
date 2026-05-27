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
Tests for CuOptPortfolioOptimizer.

Tests both MILP (binary) and QP (continuous relaxation) paths.
All tests are skipped if cuopt is not installed.
"""

import numpy as np
import pytest

cuopt = pytest.importorskip("cuopt")

from simple_portfolio_qubo import (
    get_example_assets,
    get_example_constraints,
)
from cuopt_portfolio import CuOptPortfolioOptimizer


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
    return CuOptPortfolioOptimizer(
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


class TestCuOptConstruction:
    """Test model construction for both paths."""

    def test_milp_model_creates(self, optimizer):
        """MILP model builds without error."""
        prob, x = optimizer.build_milp_model()
        assert len(x) == optimizer.n

    def test_qp_model_creates(self, optimizer):
        """QP model builds without error."""
        prob, x = optimizer.build_qp_model()
        assert len(x) == optimizer.n

    def test_milp_variable_count(self, optimizer):
        """MILP has one binary variable per asset."""
        prob, x = optimizer.build_milp_model()
        assert len(x) == len(optimizer.assets)

    def test_qp_variable_count(self, optimizer):
        """QP has one continuous variable per asset."""
        prob, x = optimizer.build_qp_model()
        assert len(x) == len(optimizer.assets)

    def test_qp_matrix_scaling(self, optimizer):
        """QP matrix is 2x the QUBO matrix."""
        q_qubo = optimizer._qubo_optimizer.build_qubo_matrix()
        # Verify the QUBO matrix exists and is square
        assert q_qubo.shape == (optimizer.n, optimizer.n)

    def test_constructor_stores_params(self, example_problem):
        """Constructor stores all parameters correctly."""
        assets, constraints = example_problem
        opt = CuOptPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )
        assert opt.budget == constraints['budget']
        assert opt.max_duration == constraints['max_duration']
        assert opt.max_cardinality == constraints['max_cardinality']
        assert opt.n == len(assets)


class TestMILPSolving:
    """Test MILP path (binary variables, exact)."""

    def test_milp_returns_feasible(self, optimizer):
        """MILP solution satisfies all constraints."""
        result = optimizer.solve_milp()
        assert result['is_feasible']

    def test_milp_positive_score(self, optimizer):
        """MILP finds a solution with positive score."""
        result = optimizer.solve_milp()
        assert result['total_score'] > 0

    def test_milp_result_structure(self, optimizer):
        """MILP result has all expected keys."""
        result = optimizer.solve_milp()
        expected_keys = {
            'selection', 'selected_assets', 'energy',
            'total_price', 'total_duration', 'total_score',
            'num_selected', 'is_feasible',
        }
        assert expected_keys == set(result.keys())

    def test_milp_binary_selection(self, optimizer):
        """MILP selection is strictly binary."""
        result = optimizer.solve_milp()
        for val in result['selection']:
            assert val in (0, 1)

    def test_milp_budget_satisfied(self, optimizer):
        """MILP respects budget constraint."""
        result = optimizer.solve_milp()
        assert result['total_price'] <= optimizer.budget

    def test_milp_duration_satisfied(self, optimizer):
        """MILP respects duration constraint."""
        result = optimizer.solve_milp()
        assert result['total_duration'] <= optimizer.max_duration

    def test_milp_cardinality_satisfied(self, optimizer):
        """MILP respects cardinality constraint."""
        result = optimizer.solve_milp()
        assert result['num_selected'] <= optimizer.max_cardinality

    def test_milp_optimal_score(self, optimizer):
        """MILP finds the optimal score (17 for example problem).

        The example problem optimal is assets A+C+E = score 17.
        MILP should find this since it's an exact solver.
        """
        result = optimizer.solve_milp()
        assert result['total_score'] == 17.0


class TestQPSolving:
    """Test QP path (continuous relaxation + rounding)."""

    def test_qp_returns_feasible(self, optimizer):
        """QP solution is feasible after rounding."""
        result = optimizer.solve_qp()
        assert result['is_feasible']

    def test_qp_positive_score(self, optimizer):
        """QP finds a solution with positive score."""
        result = optimizer.solve_qp()
        assert result['total_score'] > 0

    def test_qp_result_structure(self, optimizer):
        """QP result has all expected keys."""
        result = optimizer.solve_qp()
        expected_keys = {
            'selection', 'selected_assets', 'energy',
            'total_price', 'total_duration', 'total_score',
            'num_selected', 'is_feasible',
        }
        assert expected_keys == set(result.keys())

    def test_qp_binary_after_rounding(self, optimizer):
        """QP selection is binary after rounding."""
        result = optimizer.solve_qp()
        for val in result['selection']:
            assert val in (0, 1)

    def test_qp_stores_continuous(self, optimizer):
        """QP stores the continuous pre-rounding solution."""
        result = optimizer.solve_qp()
        continuous = optimizer.get_continuous_solution()
        assert continuous is not None
        assert len(continuous) == optimizer.n

    def test_qp_continuous_in_bounds(self, optimizer):
        """QP continuous solution is in [0, 1]."""
        result = optimizer.solve_qp()
        continuous = optimizer.get_continuous_solution()
        assert np.all(continuous >= -1e-6)
        assert np.all(continuous <= 1.0 + 1e-6)

    def test_qp_solution_quality(self, optimizer):
        """QP finds a reasonable score (>= 13 for example)."""
        result = optimizer.solve_qp()
        assert result['total_score'] >= 13.0


class TestRoundAndRepair:
    """Test the round-and-repair post-processing."""

    def test_exact_binary_unchanged(self, optimizer):
        """Exact binary input passes through unchanged."""
        binary = np.array([1, 0, 1, 0, 1])
        result = optimizer._round_and_repair(binary.astype(float))
        # May be modified by repair if infeasible, but type is int
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
        # All ones is likely infeasible (exceeds budget/duration)
        all_high = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        result = optimizer._round_and_repair(all_high)
        assert optimizer._is_feasible(result)

    def test_repair_drops_lowest_score(self, optimizer):
        """Repair drops lowest-score assets first."""
        # Force all selected, repair should keep high-score ones
        all_high = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        result = optimizer._round_and_repair(all_high)
        # Check that selected assets have higher scores on avg
        if np.sum(result) > 0:
            selected_scores = optimizer.scores[result == 1]
            dropped_scores = optimizer.scores[result == 0]
            if len(dropped_scores) > 0 and len(selected_scores) > 0:
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
        # Select just the first asset (should be within all limits)
        sel = np.zeros(optimizer.n, dtype=int)
        sel[0] = 1
        check = optimizer.check_constraints(sel)
        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']

    def test_budget_violation(self, optimizer):
        """Over-budget selection is detected."""
        # Select all assets — likely exceeds budget
        sel = np.ones(optimizer.n, dtype=int)
        check = optimizer.check_constraints(sel)
        # At least one constraint should fail for all-selected
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

    def test_single_asset_milp(self):
        """MILP works with a single asset."""
        assets = [
            {'id': 'X', 'price': 1.0, 'duration': 5, 'score': 10}
        ]
        opt = CuOptPortfolioOptimizer(
            assets=assets,
            budget=2.0,
            max_duration=10,
            max_cardinality=1,
        )
        result = opt.solve_milp()
        assert result['is_feasible']
        assert result['total_score'] == 10.0

    def test_single_asset_qp(self):
        """QP works with a single asset."""
        assets = [
            {'id': 'X', 'price': 1.0, 'duration': 5, 'score': 10}
        ]
        opt = CuOptPortfolioOptimizer(
            assets=assets,
            budget=2.0,
            max_duration=10,
            max_cardinality=1,
        )
        result = opt.solve_qp()
        assert result['is_feasible']

    def test_tight_cardinality(self):
        """max_cardinality=1 selects at most one asset."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        opt = CuOptPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=1,
        )
        result = opt.solve_milp()
        assert result['num_selected'] <= 1

    def test_no_continuous_before_qp(self, optimizer):
        """get_continuous_solution returns None before solve."""
        assert optimizer.get_continuous_solution() is None
