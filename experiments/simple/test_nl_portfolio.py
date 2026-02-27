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
Tests for NL Model Portfolio Optimizer.

Tests the dwave-optimization based portfolio optimizer including:
- NL model construction with binary variables and constraints
- Brute-force exact solver
- Constraint checking
- Comparison with CQM exact solver
"""

import numpy as np
import pytest

from nl_portfolio import (
    NLPortfolioOptimizer,
    get_example_assets,
    get_example_constraints
)
from cqm_portfolio import CQMPortfolioOptimizer


class TestNLModelConstruction:
    """Tests for NL model construction."""

    def test_build_model_creates_valid_model(self):
        """Test that build_model creates a valid dwave-optimization Model."""
        from dwave.optimization import Model

        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        model = optimizer.build_model()

        assert isinstance(model, Model)

    def test_model_has_correct_num_decisions(self):
        """Test that the model has exactly one decision variable (binary array)."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        optimizer.build_model()
        info = optimizer.get_model_info()

        # One binary array covers all n assets
        assert info['num_decisions'] == 1

    def test_model_has_three_constraints(self):
        """Test that the model has budget, duration, and cardinality constraints."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        optimizer.build_model()
        info = optimizer.get_model_info()

        assert info['num_constraints'] == 3

    def test_model_info_matches_inputs(self):
        """Test that get_model_info reflects the input parameters."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        optimizer.build_model()
        info = optimizer.get_model_info()

        assert info['num_assets'] == len(assets)
        assert info['budget'] == constraints['budget']
        assert info['max_duration'] == constraints['max_duration']
        assert info['max_cardinality'] == constraints['max_cardinality']

    def test_model_feasibility_evaluation(self):
        """Test that the model correctly identifies feasible/infeasible states."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        model = optimizer.build_model()
        model.lock()

        decisions = list(model.iter_decisions())
        x_var = decisions[0]
        model.states.resize(1)

        # A+E: price=2.02, duration=15, score=17 — feasible
        x_var.set_state(0, np.array([1.0, 0.0, 0.0, 0.0, 1.0]))
        assert model.feasible(0)

        # All 5 selected: price=4.95, duration=29 — infeasible
        x_var.set_state(0, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        assert not model.feasible(0)

        model.unlock()

    def test_model_objective_evaluation(self):
        """Test that the model computes the correct objective (negated score)."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        model = optimizer.build_model()
        model.lock()

        decisions = list(model.iter_decisions())
        x_var = decisions[0]
        model.states.resize(1)

        # A+E: score = 8+9 = 17, objective (negated) = -17
        x_var.set_state(0, np.array([1.0, 0.0, 0.0, 0.0, 1.0]))
        assert model.objective.state(0) == pytest.approx(-17.0)

        # C only: score = 5, objective = -5
        x_var.set_state(0, np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
        assert model.objective.state(0) == pytest.approx(-5.0)

        model.unlock()


class TestExactSolver:
    """Tests for brute-force exact solver."""

    def test_solve_exact_returns_feasible_solution(self):
        """Test that solve_exact returns a feasible solution."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        assert result['is_feasible']

    def test_solve_exact_maximizes_score(self):
        """Test that the exact solver finds the optimal score."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        # Optimal solution is A+E with score 17
        assert result['total_score'] == pytest.approx(17.0)

    def test_solve_exact_selects_correct_assets(self):
        """Test that the exact solver selects the optimal assets."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        assert set(result['selected_assets']) == {'A', 'E'}

    def test_solve_exact_respects_budget(self):
        """Test that the solution respects the budget constraint."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        assert result['total_price'] <= constraints['budget']

    def test_solve_exact_respects_duration(self):
        """Test that the solution respects the duration constraint."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        assert result['total_duration'] <= constraints['max_duration']

    def test_solve_exact_respects_cardinality(self):
        """Test that the solution respects the cardinality constraint."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        assert result['num_selected'] <= constraints['max_cardinality']

    def test_solve_exact_result_structure(self):
        """Test that solve_exact returns all expected keys."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        expected_keys = {
            'selection', 'selected_assets', 'energy',
            'total_price', 'total_duration', 'total_score',
            'num_selected', 'is_feasible'
        }
        assert set(result.keys()) == expected_keys

    def test_solve_exact_selection_is_binary(self):
        """Test that the selection vector is binary."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        assert all(v in (0.0, 1.0) for v in result['selection'])

    def test_solve_exact_metrics_consistent(self):
        """Test that reported metrics are consistent with selection."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()
        selected = result['selection']

        prices = np.array([a['price'] for a in assets])
        durations = np.array([a['duration'] for a in assets])
        scores = np.array([a['score'] for a in assets])

        assert result['total_price'] == pytest.approx(float(np.dot(selected, prices)))
        assert result['total_duration'] == pytest.approx(float(np.dot(selected, durations)))
        assert result['total_score'] == pytest.approx(float(np.dot(selected, scores)))
        assert result['num_selected'] == int(np.sum(selected))


class TestConstraintChecking:
    """Tests for constraint checking."""

    def test_feasible_selection(self):
        """Test that a feasible selection is correctly identified."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        # A+C: price=1.89, duration=10, cardinality=2
        selection = np.array([1, 0, 1, 0, 0])
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']

    def test_budget_violation(self):
        """Test detection of budget constraint violation."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        # A+B+D+E: price=4.06 > 3.0
        selection = np.array([1, 1, 0, 1, 1])
        check = optimizer.check_constraints(selection)

        assert not check['budget_satisfied']

    def test_duration_violation(self):
        """Test detection of duration constraint violation."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        # A+B+E: duration=6+7+9=22 > 15
        selection = np.array([1, 1, 0, 0, 1])
        check = optimizer.check_constraints(selection)

        assert not check['duration_satisfied']

    def test_cardinality_violation(self):
        """Test detection of cardinality constraint violation."""
        assets = get_example_assets()
        # Use tight cardinality to force violation
        optimizer = NLPortfolioOptimizer(
            assets=assets, budget=10.0, max_duration=100, max_cardinality=1
        )

        selection = np.array([1, 1, 0, 0, 0])
        check = optimizer.check_constraints(selection)

        assert not check['cardinality_satisfied']

    def test_empty_selection(self):
        """Test that empty selection satisfies all constraints."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = NLPortfolioOptimizer(assets=assets, **constraints)

        selection = np.zeros(len(assets))
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']


class TestComparisonWithCQM:
    """Tests comparing NL model results with CQM exact solver."""

    def test_same_optimal_score(self):
        """Test that NL and CQM exact solvers find the same optimal score."""
        assets = get_example_assets()
        constraints = get_example_constraints()

        nl_optimizer = NLPortfolioOptimizer(assets=assets, **constraints)
        cqm_optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        nl_result = nl_optimizer.solve_exact()
        cqm_result = cqm_optimizer.solve_exact()

        assert nl_result['total_score'] == pytest.approx(cqm_result['total_score'])

    def test_same_selected_assets(self):
        """Test that NL and CQM exact solvers select the same assets."""
        assets = get_example_assets()
        constraints = get_example_constraints()

        nl_optimizer = NLPortfolioOptimizer(assets=assets, **constraints)
        cqm_optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        nl_result = nl_optimizer.solve_exact()
        cqm_result = cqm_optimizer.solve_exact()

        assert set(nl_result['selected_assets']) == set(cqm_result['selected_assets'])

    def test_both_feasible(self):
        """Test that both solvers return feasible solutions."""
        assets = get_example_assets()
        constraints = get_example_constraints()

        nl_optimizer = NLPortfolioOptimizer(assets=assets, **constraints)
        cqm_optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        nl_result = nl_optimizer.solve_exact()
        cqm_result = cqm_optimizer.solve_exact()

        assert nl_result['is_feasible']
        assert cqm_result['is_feasible']


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_asset(self):
        """Test with a single asset."""
        assets = [{'id': 'X', 'price': 1.0, 'duration': 5, 'score': 10}]
        optimizer = NLPortfolioOptimizer(
            assets=assets, budget=2.0, max_duration=10, max_cardinality=1
        )

        result = optimizer.solve_exact()

        assert result['selected_assets'] == ['X']
        assert result['total_score'] == pytest.approx(10.0)

    def test_tight_budget_excludes_expensive(self):
        """Test that tight budget forces cheaper assets."""
        assets = [
            {'id': 'cheap', 'price': 0.5, 'duration': 1, 'score': 5},
            {'id': 'expensive', 'price': 2.0, 'duration': 1, 'score': 6},
        ]
        optimizer = NLPortfolioOptimizer(
            assets=assets, budget=1.0, max_duration=10, max_cardinality=2
        )

        result = optimizer.solve_exact()

        assert 'cheap' in result['selected_assets']
        assert 'expensive' not in result['selected_assets']

    def test_tight_cardinality(self):
        """Test with max_cardinality=1 selects the best single asset."""
        assets = get_example_assets()
        optimizer = NLPortfolioOptimizer(
            assets=assets, budget=10.0, max_duration=100, max_cardinality=1
        )

        result = optimizer.solve_exact()

        # E has the highest score (9)
        assert result['num_selected'] == 1
        assert result['selected_assets'] == ['E']
        assert result['total_score'] == pytest.approx(9.0)

    def test_impossible_constraints_selects_nothing(self):
        """Test that impossible constraints result in empty selection."""
        assets = get_example_assets()
        optimizer = NLPortfolioOptimizer(
            assets=assets, budget=0.0, max_duration=0, max_cardinality=0
        )

        result = optimizer.solve_exact()

        assert result['num_selected'] == 0
        assert result['total_score'] == pytest.approx(0.0)
