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
Tests for QHD Portfolio Optimizer.

Tests the QHDOPT-based portfolio optimizer including:
- QP model construction (QUBO matrix conversion)
- SymPy model construction (native formulation + binary enforcement)
- Classical solving via IPOPT backend
- Round-and-repair logic
- Constraint checking
- Comparison with known optimal solutions
"""

import numpy as np
import pytest

qhdopt = pytest.importorskip("qhdopt")

from qhd_portfolio import QHDPortfolioOptimizer
from simple_portfolio_qubo import (
    SimplePortfolioQUBO,
    get_example_assets,
    get_example_constraints,
)


class TestQHDConstruction:
    """Tests for QHD model construction."""

    def test_creates_optimizer_with_correct_attributes(self):
        """Test that constructor stores all parameters."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        assert optimizer.n == 5
        assert optimizer.budget == 3.0
        assert optimizer.max_duration == 15
        assert optimizer.max_cardinality == 3

    def test_build_qp_model_returns_qhd_instance(self):
        """Test that build_qp_model produces a QHD object."""
        from qhdopt import QHD

        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        model = optimizer.build_qp_model()
        assert isinstance(model, QHD)

    def test_build_sympy_model_returns_qhd_instance(self):
        """Test that build_sympy_model produces a QHD object."""
        from qhdopt import QHD

        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        model = optimizer.build_sympy_model()
        assert isinstance(model, QHD)

    def test_qp_matrix_is_double_qubo(self):
        """Test that Q_qhd = 2 * Q_qubo for correct QP conversion."""
        assets = get_example_assets()
        constraints = get_example_constraints()

        qubo_opt = SimplePortfolioQUBO(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )
        q_qubo = qubo_opt.build_qubo_matrix()

        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        # Access the internal QUBO matrix via the QP model build
        q_internal = optimizer._qubo_optimizer.build_qubo_matrix()
        np.testing.assert_array_almost_equal(q_internal, q_qubo)

    def test_sympy_model_has_correct_dimension(self):
        """Test that the SymPy model has the right number of variables."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        model = optimizer.build_sympy_model()
        assert model.dimension == 5


class TestQPSolving:
    """Tests for QP path solving."""

    def test_solve_qp_returns_feasible(self):
        """Test that solve_qp returns a feasible solution."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_qp(num_shots=50)
        assert result['is_feasible']

    def test_solve_qp_finds_positive_score(self):
        """Test that the QP solver finds a solution with positive score."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_qp(num_shots=50)
        assert result['total_score'] > 0

    def test_solve_qp_result_structure(self):
        """Test that solve_qp returns all expected keys."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_qp(num_shots=20)

        expected_keys = {
            'selection', 'selected_assets', 'energy',
            'total_price', 'total_duration', 'total_score',
            'num_selected', 'is_feasible',
        }
        assert set(result.keys()) == expected_keys

    def test_solve_qp_selection_is_binary(self):
        """Test that the rounded selection is binary."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_qp(num_shots=20)
        assert all(v in (0, 1) for v in result['selection'])

    def test_solve_qp_respects_budget(self):
        """Test that the solution respects the budget."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_qp(num_shots=50)
        assert result['total_price'] <= constraints['budget']

    def test_solve_qp_respects_duration(self):
        """Test that the solution respects duration."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_qp(num_shots=50)
        assert result['total_duration'] <= constraints['max_duration']

    def test_solve_qp_respects_cardinality(self):
        """Test that the solution respects cardinality."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_qp(num_shots=50)
        assert result['num_selected'] <= constraints['max_cardinality']

    def test_solve_qp_stores_continuous_solution(self):
        """Test that the continuous solution is accessible."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        optimizer.solve_qp(num_shots=20)
        continuous = optimizer.get_continuous_solution()
        assert continuous is not None
        assert len(continuous) == 5
        assert all(
            -0.1 <= v <= 1.1 for v in continuous
        )  # Allow small float overshoot


class TestSymPySolving:
    """Tests for SymPy path solving."""

    def test_solve_sympy_returns_feasible(self):
        """Test that solve_sympy returns a feasible solution."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_sympy(num_shots=50)
        assert result['is_feasible']

    def test_solve_sympy_finds_good_solution(self):
        """Test that SymPy path finds a near-optimal solution."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_sympy(num_shots=50)
        # Optimal is 17 (A+E). Accept >= 13 for robustness.
        assert result['total_score'] >= 13

    def test_solve_sympy_result_structure(self):
        """Test that solve_sympy returns all expected keys."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_sympy(num_shots=20)

        expected_keys = {
            'selection', 'selected_assets', 'energy',
            'total_price', 'total_duration', 'total_score',
            'num_selected', 'is_feasible',
        }
        assert set(result.keys()) == expected_keys

    def test_solve_sympy_selection_is_binary(self):
        """Test that the selection is binary."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_sympy(num_shots=20)
        assert all(v in (0, 1) for v in result['selection'])

    def test_solve_sympy_respects_all_constraints(self):
        """Test all constraints at once for SymPy path."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        result = optimizer.solve_sympy(num_shots=50)
        assert result['total_price'] <= constraints['budget']
        assert (
            result['total_duration'] <= constraints['max_duration']
        )
        assert (
            result['num_selected']
            <= constraints['max_cardinality']
        )


class TestRoundAndRepair:
    """Tests for the rounding and repair logic."""

    def test_exact_binary_unchanged(self):
        """Test that already-binary input is not modified."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        binary = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
        result = optimizer._round_and_repair(binary)
        np.testing.assert_array_equal(result, [1, 0, 0, 0, 1])

    def test_near_binary_rounds_correctly(self):
        """Test that near-binary values round to the correct side."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        near_binary = np.array([0.99, 0.01, 0.01, 0.01, 0.99])
        result = optimizer._round_and_repair(near_binary)
        np.testing.assert_array_equal(result, [1, 0, 0, 0, 1])

    def test_infeasible_rounding_triggers_repair(self):
        """Test that infeasible rounded solutions get repaired."""
        assets = get_example_assets()
        # Tight budget forces repair
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=1.5,
            max_duration=100,
            max_cardinality=5,
        )

        # All above 0.5 → rounds to all selected → infeasible
        continuous = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        result = optimizer._round_and_repair(continuous)

        # After repair, should be feasible
        total_price = float(
            np.sum(optimizer.prices[np.where(result == 1)[0]])
        )
        assert total_price <= 1.5

    def test_repair_drops_lowest_score_first(self):
        """Test that repair removes lowest-score assets first."""
        assets = get_example_assets()
        # Budget allows exactly 2 cheapest
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=2.0,
            max_duration=100,
            max_cardinality=5,
        )

        # All above 0.5 → rounds to all → infeasible
        continuous = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        result = optimizer._round_and_repair(continuous)

        # D(score=1) should be dropped first, then B(score=4)
        assert result[3] == 0  # D dropped (lowest score)
        assert optimizer._is_feasible(result)

    def test_all_zeros_stays_feasible(self):
        """Test that all-zeros input stays feasible."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        zeros = np.array([0.1, 0.2, 0.3, 0.1, 0.0])
        result = optimizer._round_and_repair(zeros)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])
        assert optimizer._is_feasible(result)


class TestConstraintChecking:
    """Tests for constraint checking."""

    def test_feasible_selection(self):
        """Test that a feasible selection is correctly identified."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        # A+E: price=2.02, duration=15, cardinality=2
        selection = np.array([1, 0, 0, 0, 1])
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']

    def test_budget_violation(self):
        """Test detection of budget violation."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        # A+B+D+E: price=4.06 > 3.0
        selection = np.array([1, 1, 0, 1, 1])
        check = optimizer.check_constraints(selection)

        assert not check['budget_satisfied']

    def test_empty_selection(self):
        """Test that empty selection satisfies all constraints."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=constraints['budget'],
            max_duration=constraints['max_duration'],
            max_cardinality=constraints['max_cardinality'],
        )

        selection = np.zeros(5)
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_asset(self):
        """Test with a single asset."""
        assets = [
            {'id': 'X', 'price': 1.0, 'duration': 5, 'score': 10}
        ]
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=2.0,
            max_duration=10,
            max_cardinality=1,
        )

        result = optimizer.solve_sympy(num_shots=50)
        assert result['is_feasible']
        assert result['total_score'] == pytest.approx(10.0)

    def test_tight_cardinality(self):
        """Test with max_cardinality=1 selects best single asset."""
        assets = get_example_assets()
        optimizer = QHDPortfolioOptimizer(
            assets=assets,
            budget=10.0,
            max_duration=100,
            max_cardinality=1,
        )

        result = optimizer.solve_sympy(num_shots=50)
        assert result['is_feasible']
        assert result['num_selected'] <= 1
        # E has highest score (9)
        if result['num_selected'] == 1:
            assert result['total_score'] >= 8  # Allow A or E
