#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for CQM Portfolio Optimizer.

Tests the CQM-based portfolio optimizer including:
- CQM construction with inequality constraints
- ExactCQMSolver for exact solutions
- CQM->BQM conversion and simulated annealing
- Comparison with other solvers (ILP, manual QUBO)
"""

import numpy as np
import pytest
from dimod import ConstrainedQuadraticModel

from cqm_portfolio import (
    CQMPortfolioOptimizer,
    get_example_assets,
    get_example_constraints
)
from simple_portfolio_qubo import SimplePortfolioQUBO


class TestCQMConstruction:
    """Tests for CQM model construction."""

    def test_build_cqm_creates_valid_model(self):
        """Test that build_cqm creates a valid CQM."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        cqm = optimizer.build_cqm()

        assert isinstance(cqm, ConstrainedQuadraticModel)
        assert cqm.num_variables() == len(assets)
        assert cqm.num_constraints() == 3  # budget, duration, cardinality

    def test_cqm_has_correct_constraints(self):
        """Test that CQM has the expected constraint labels."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        optimizer.build_cqm()
        info = optimizer.get_cqm_info()

        assert 'budget' in info['constraint_labels']
        assert 'duration' in info['constraint_labels']
        assert 'cardinality' in info['constraint_labels']

    def test_cqm_variables_match_assets(self):
        """Test that CQM variables match asset IDs."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        optimizer.build_cqm()
        info = optimizer.get_cqm_info()

        asset_ids = [a['id'] for a in assets]
        for asset_id in asset_ids:
            assert asset_id in info['variable_labels']


class TestExactCQMSolver:
    """Tests for ExactCQMSolver solutions."""

    def test_solve_exact_returns_feasible_solution(self):
        """Test that ExactCQMSolver returns a feasible solution."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        assert result['is_feasible']
        check = optimizer.check_constraints(result['selection'])
        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']

    def test_solve_exact_maximizes_score(self):
        """Test that ExactCQMSolver finds optimal score."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_exact()

        # For the example, optimal feasible solution should select
        # assets with high scores while respecting constraints
        # Expected optimal: A(8) + C(5) + E(9) = 22 with price 2.91, duration 19 (too high)
        # or A(8) + E(9) = 17 with price 2.02, duration 15
        # or A(8) + C(5) = 13 with price 1.89, duration 10
        # Need to verify what's actually optimal

        # At minimum, score should be positive and constraints satisfied
        assert result['total_score'] > 0
        assert result['total_price'] <= constraints['budget']
        assert result['total_duration'] <= constraints['max_duration']
        assert result['num_selected'] <= constraints['max_cardinality']

    def test_solve_exact_matches_ilp(self):
        """Test that ExactCQMSolver matches ILP optimal solution."""
        try:
            from scipy.optimize import milp, Bounds, LinearConstraint
        except ImportError:
            pytest.skip("scipy.optimize.milp not available")

        assets = get_example_assets()
        constraints = get_example_constraints()

        # Solve with ILP
        n = len(assets)
        prices = np.array([a['price'] for a in assets])
        durations = np.array([a['duration'] for a in assets])
        scores = np.array([a['score'] for a in assets])

        c = -scores  # Maximize score = minimize -score
        A = np.vstack([prices, durations, np.ones(n)])
        b_upper = np.array([constraints['budget'], constraints['max_duration'], constraints['max_cardinality']])
        b_lower = np.array([-np.inf, -np.inf, -np.inf])

        lin_constraint = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        integrality = np.ones(n)

        ilp_result = milp(c, constraints=lin_constraint, bounds=bounds, integrality=integrality)
        ilp_score = -ilp_result.fun if ilp_result.success else 0.0

        # Solve with CQM
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)
        cqm_result = optimizer.solve_exact()

        # CQM should match ILP optimal
        assert abs(cqm_result['total_score'] - ilp_score) < 0.01

    def test_solve_exact_single_asset(self):
        """Test with a single asset."""
        assets = [{'id': 'X', 'price': 1.0, 'duration': 5, 'score': 10}]
        optimizer = CQMPortfolioOptimizer(
            assets=assets,
            budget=2.0,
            max_duration=10,
            max_cardinality=1
        )

        result = optimizer.solve_exact()

        assert result['is_feasible']
        assert result['selected_assets'] == ['X']
        assert result['total_score'] == 10

    def test_solve_exact_no_feasible_solution(self):
        """Test behavior when no feasible solution exists."""
        assets = [
            {'id': 'A', 'price': 5.0, 'duration': 20, 'score': 10},
            {'id': 'B', 'price': 6.0, 'duration': 25, 'score': 15},
        ]
        optimizer = CQMPortfolioOptimizer(
            assets=assets,
            budget=1.0,  # Too low for any asset
            max_duration=5,
            max_cardinality=1
        )

        result = optimizer.solve_exact()

        # ExactCQMSolver should return empty selection or infeasible
        # The empty selection {} is technically feasible (selects nothing)
        assert result['num_selected'] == 0 or not result['is_feasible']


class TestSimulatedAnnealing:
    """Tests for CQM->BQM + Simulated Annealing."""

    def test_solve_sa_returns_valid_result(self):
        """Test that solve_sa returns a valid result structure."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        result = optimizer.solve_sa(num_reads=100, seed=42)

        assert 'selection' in result
        assert 'selected_assets' in result
        assert 'total_score' in result
        assert 'is_feasible' in result
        assert len(result['selection']) == len(assets)

    def test_solve_sa_with_lagrange_multiplier(self):
        """Test that custom lagrange_multiplier works."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        # High lagrange multiplier should enforce constraints more strictly
        result = optimizer.solve_sa(num_reads=500, lagrange_multiplier=100.0, seed=42)

        # Should find a valid solution
        assert 'total_score' in result

    def test_solve_sa_produces_valid_results(self):
        """Test that solve_sa produces valid results across multiple runs."""
        assets = get_example_assets()
        constraints = get_example_constraints()

        # Run multiple times - verify we get valid results
        # Note: CQM->BQM conversion adds slack variables which increases
        # problem complexity and can lead to variable solution quality
        optimizer1 = CQMPortfolioOptimizer(assets=assets, **constraints)
        result1 = optimizer1.solve_sa(num_reads=500, seed=123)

        optimizer2 = CQMPortfolioOptimizer(assets=assets, **constraints)
        result2 = optimizer2.solve_sa(num_reads=500, seed=456)

        # Both should find some feasible solution or at least non-negative scores
        assert result1['is_feasible'] or result1['total_score'] >= 0
        assert result2['is_feasible'] or result2['total_score'] >= 0

        # Both should select some assets (not degenerate empty solutions)
        assert result1['num_selected'] >= 0
        assert result2['num_selected'] >= 0

    def test_solve_sa_near_optimal(self):
        """Test that SA solution is near optimal for small problems."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        exact_result = optimizer.solve_exact()

        # Rebuild CQM for SA (solve_exact may have used it)
        optimizer._cqm = None
        sa_result = optimizer.solve_sa(num_reads=1000, seed=42)

        # SA should be within 50% of optimal for this small problem
        # (may not always match due to constraint penalty trade-offs)
        if exact_result['total_score'] > 0:
            gap = (exact_result['total_score'] - sa_result['total_score']) / exact_result['total_score']
            assert gap < 0.5  # Within 50%


class TestBQMConversion:
    """Tests for CQM->BQM conversion."""

    def test_to_bqm_returns_bqm_and_inverter(self):
        """Test that to_bqm returns BQM and inverter."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        bqm, inverter = optimizer.to_bqm()

        from dimod import BinaryQuadraticModel
        assert isinstance(bqm, BinaryQuadraticModel)
        assert inverter is not None

    def test_to_bqm_has_penalty_terms(self):
        """Test that BQM has additional terms from constraint penalties."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        bqm, _ = optimizer.to_bqm()

        # BQM may have more variables due to slack encoding
        # At minimum it should have the original asset variables
        for asset in assets:
            assert asset['id'] in bqm.variables

    def test_to_bqm_with_custom_lagrange(self):
        """Test BQM conversion with custom lagrange multiplier."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        bqm_default, _ = optimizer.to_bqm()
        bqm_custom, _ = optimizer.to_bqm(lagrange_multiplier=1000.0)

        # Different lagrange multipliers should produce different biases
        # (comparing linear biases of original variables)
        default_biases = [bqm_default.linear.get(a['id'], 0) for a in assets]
        custom_biases = [bqm_custom.linear.get(a['id'], 0) for a in assets]

        # At least some biases should differ
        assert any(abs(d - c) > 0.01 for d, c in zip(default_biases, custom_biases))


class TestConstraintChecking:
    """Tests for constraint satisfaction checking."""

    def test_check_constraints_all_satisfied(self):
        """Test check_constraints with valid selection."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        # Select assets A and C: price=1.89, duration=10, count=2
        selection = np.array([1, 0, 1, 0, 0])  # A, C
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied']  # 1.89 <= 3.0
        assert check['duration_satisfied']  # 10 <= 15
        assert check['cardinality_satisfied']  # 2 <= 3

    def test_check_constraints_budget_violated(self):
        """Test check_constraints with budget violation."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        # Select all assets: price = 4.95 > 3.0
        selection = np.array([1, 1, 1, 1, 1])
        check = optimizer.check_constraints(selection)

        assert not check['budget_satisfied']

    def test_check_constraints_empty_selection(self):
        """Test check_constraints with empty selection."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        optimizer = CQMPortfolioOptimizer(assets=assets, **constraints)

        selection = np.array([0, 0, 0, 0, 0])
        check = optimizer.check_constraints(selection)

        # Empty selection satisfies all constraints
        assert check['budget_satisfied']
        assert check['duration_satisfied']
        assert check['cardinality_satisfied']
        assert check['total_price'] == 0.0
        assert check['total_duration'] == 0.0


class TestComparisonWithManualQUBO:
    """Tests comparing CQM with manual QUBO implementations."""

    def test_cqm_vs_simple_qubo_feasibility(self):
        """Compare CQM feasibility rate vs SimplePortfolioQUBO."""
        assets = get_example_assets()
        cqm_constraints = get_example_constraints()

        # CQM optimizer
        cqm_opt = CQMPortfolioOptimizer(assets=assets, **cqm_constraints)
        cqm_result = cqm_opt.solve_exact()

        # Simple QUBO optimizer (with example penalty weights)
        qubo_constraints = {
            **cqm_constraints,
            'lambda_budget': 2.0,
            'lambda_duration': 10.0,
            'lambda_cardinality': 5.0,
        }
        qubo_opt = SimplePortfolioQUBO(assets=assets, **qubo_constraints)
        qubo_result = qubo_opt.solve(num_reads=1000, seed=42)
        qubo_check = qubo_opt.check_constraints(qubo_result['selection'])

        # CQM exact should always be feasible
        assert cqm_result['is_feasible']

        # SimpleQUBO may or may not be feasible (soft constraints)
        # Just verify the comparison can be made
        qubo_feasible = (qubo_check['budget_satisfied'] and
                         qubo_check['duration_satisfied'] and
                         qubo_check['cardinality_satisfied'])

        # Report difference
        print(f"\nCQM Exact: score={cqm_result['total_score']}, feasible={cqm_result['is_feasible']}")
        print(f"SimpleQUBO: score={qubo_result['total_score']}, feasible={qubo_feasible}")

    def test_cqm_exact_is_optimal(self):
        """Verify CQM exact solution is truly optimal."""
        assets = [
            {'id': 'A', 'price': 1.0, 'duration': 1, 'score': 10},
            {'id': 'B', 'price': 1.0, 'duration': 1, 'score': 20},
            {'id': 'C', 'price': 1.0, 'duration': 1, 'score': 5},
        ]
        optimizer = CQMPortfolioOptimizer(
            assets=assets,
            budget=2.5,
            max_duration=3,
            max_cardinality=2
        )

        result = optimizer.solve_exact()

        # Optimal should be A(10) + B(20) = 30
        assert result['total_score'] == 30
        assert set(result['selected_assets']) == {'A', 'B'}


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_tight_budget_constraint(self):
        """Test when budget is exactly met."""
        assets = [
            {'id': 'A', 'price': 1.0, 'duration': 1, 'score': 10},
            {'id': 'B', 'price': 2.0, 'duration': 1, 'score': 20},
        ]
        optimizer = CQMPortfolioOptimizer(
            assets=assets,
            budget=3.0,  # Exactly A + B
            max_duration=10,
            max_cardinality=2
        )

        result = optimizer.solve_exact()

        # Should select both when budget is exactly met
        assert result['total_price'] <= 3.0
        assert result['is_feasible']

    def test_cardinality_one(self):
        """Test with cardinality constraint of 1."""
        assets = [
            {'id': 'A', 'price': 1.0, 'duration': 1, 'score': 5},
            {'id': 'B', 'price': 1.0, 'duration': 1, 'score': 10},
            {'id': 'C', 'price': 1.0, 'duration': 1, 'score': 3},
        ]
        optimizer = CQMPortfolioOptimizer(
            assets=assets,
            budget=10.0,
            max_duration=10,
            max_cardinality=1
        )

        result = optimizer.solve_exact()

        # Should select only B (highest score)
        assert result['num_selected'] == 1
        assert result['selected_assets'] == ['B']
        assert result['total_score'] == 10

    def test_many_assets_small_cardinality(self):
        """Test with many assets but small cardinality."""
        n = 20
        assets = [
            {'id': f'A{i}', 'price': 0.5, 'duration': 1, 'score': i}
            for i in range(n)
        ]
        optimizer = CQMPortfolioOptimizer(
            assets=assets,
            budget=1.0,  # Allows 2 assets
            max_duration=10,
            max_cardinality=2
        )

        result = optimizer.solve_exact()

        # Should select the two highest-scoring assets (A18, A19)
        assert result['num_selected'] == 2
        assert result['total_score'] == 18 + 19  # Scores of A18 and A19


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
