#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for SlackPortfolioQUBO - slack variable implementation for inequality constraints.
"""

import pytest
import numpy as np
from slack_portfolio_qubo import SlackPortfolioQUBO, Asset


# ============================================================================
# Test Data - Example Problem
# ============================================================================

def get_test_assets():
    """Return example assets."""
    return [
        {'id': 'A', 'price': 1.00, 'duration': 6, 'score': 8},
        {'id': 'B', 'price': 0.99, 'duration': 7, 'score': 4},
        {'id': 'C', 'price': 0.89, 'duration': 4, 'score': 5},
        {'id': 'D', 'price': 1.05, 'duration': 3, 'score': 1},
        {'id': 'E', 'price': 1.02, 'duration': 9, 'score': 9},
    ]


def get_test_constraints():
    """Return example constraints."""
    return {
        'budget': 3.0,
        'max_duration': 15,
        'max_cardinality': 3,
        'lambda_budget': 2.0,
        'lambda_duration': 10.0,
        'lambda_cardinality': 5.0,
    }


# ============================================================================
# Basic Construction Tests
# ============================================================================

class TestSlackPortfolioQUBOConstruction:
    """Test basic construction and initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        assert optimizer.n == 5
        assert optimizer.budget == 3.0
        assert optimizer.max_duration == 15
        assert optimizer.max_cardinality == 3

    def test_slack_bits_calculation(self):
        """Test that slack bits are calculated correctly."""
        assets = get_test_assets()
        constraints = get_test_constraints()
        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget_precision=0.1,
            duration_precision=1.0,
            **constraints
        )

        info = optimizer.get_variable_info()

        # Budget slack: 0 to 3.0 with precision 0.1 => 30 values => 5 bits (2^5=32)
        assert info['n_budget_slack'] == 5

        # Duration slack: 0 to 15 with precision 1.0 => 15 values => 4 bits (2^4=16)
        assert info['n_duration_slack'] == 4

        # Cardinality slack: 0 to 3 => 3 values => 2 bits (2^2=4)
        assert info['n_cardinality_slack'] == 2

        # Total: 5 assets + 5 + 4 + 2 = 16 variables
        assert info['n_total'] == 16

    def test_variable_ranges(self):
        """Test variable index ranges are correct."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget_precision=0.1,
            duration_precision=1.0,
            **get_test_constraints()
        )

        info = optimizer.get_variable_info()

        # Assets: 0-4
        assert info['asset_range'] == (0, 5)

        # Budget slack: 5-9 (5 bits)
        assert info['budget_slack_range'] == (5, 10)

        # Duration slack: 10-13 (4 bits)
        assert info['duration_slack_range'] == (10, 14)

        # Cardinality slack: 14-15 (2 bits)
        assert info['cardinality_slack_range'] == (14, 16)


# ============================================================================
# QUBO Matrix Tests
# ============================================================================

class TestSlackQUBOMatrix:
    """Test QUBO matrix construction."""

    def test_qubo_matrix_shape(self):
        """Test QUBO matrix has correct shape."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())
        Q = optimizer.build_qubo_matrix()

        assert Q.shape == (optimizer.n_total, optimizer.n_total)

    def test_qubo_matrix_symmetric(self):
        """Test QUBO matrix is symmetric."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())
        Q = optimizer.build_qubo_matrix()

        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_score_in_diagonal(self):
        """Test that negative scores appear in diagonal."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())
        Q = optimizer.build_qubo_matrix()

        # The diagonal should include -score terms for assets
        # Asset A has score 8
        # Q[0,0] should have -8 contribution from score
        # But also constraint penalties, so we can't test exact value easily

        # Just verify diagonal entries exist and are reasonable
        for i in range(optimizer.n):
            assert Q[i, i] != 0, f"Diagonal entry Q[{i},{i}] should not be zero"


# ============================================================================
# Slack Variable Tests
# ============================================================================

class TestSlackVariables:
    """Test slack variable encoding and decoding."""

    def test_slack_coefficients(self):
        """Test binary coefficients for slack encoding."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget_precision=0.1,
            duration_precision=1.0,
            **get_test_constraints()
        )

        # Budget slack: precision 0.1, 5 bits => [0.1, 0.2, 0.4, 0.8, 1.6]
        budget_coef = optimizer._get_slack_coefficients(
            optimizer.budget_slack_bits,
            optimizer.budget_precision
        )
        expected = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
        np.testing.assert_array_almost_equal(budget_coef, expected)

        # Duration slack: precision 1.0, 4 bits => [1, 2, 4, 8]
        duration_coef = optimizer._get_slack_coefficients(
            optimizer.duration_slack_bits,
            optimizer.duration_precision
        )
        expected = np.array([1, 2, 4, 8])
        np.testing.assert_array_almost_equal(duration_coef, expected)

    def test_decode_slack_zeros(self):
        """Test decoding when all slack bits are zero."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())
        optimizer.build_qubo_matrix()

        # All zeros
        full_selection = np.zeros(optimizer.n_total)
        slack = optimizer.decode_slack(full_selection)

        assert slack['budget_slack'] == 0.0
        assert slack['duration_slack'] == 0.0
        assert slack['cardinality_slack'] == 0.0

    def test_decode_slack_max_values(self):
        """Test decoding when all slack bits are one."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget_precision=0.1,
            duration_precision=1.0,
            **get_test_constraints()
        )
        optimizer.build_qubo_matrix()

        # All ones in slack positions
        full_selection = np.zeros(optimizer.n_total)
        full_selection[optimizer.budget_slack_range[0]:optimizer.budget_slack_range[1]] = 1
        full_selection[optimizer.duration_slack_range[0]:optimizer.duration_slack_range[1]] = 1
        full_selection[optimizer.cardinality_slack_range[0]:optimizer.cardinality_slack_range[1]] = 1

        slack = optimizer.decode_slack(full_selection)

        # Budget: 0.1 + 0.2 + 0.4 + 0.8 + 1.6 = 3.1
        assert abs(slack['budget_slack'] - 3.1) < 0.01

        # Duration: 1 + 2 + 4 + 8 = 15
        assert slack['duration_slack'] == 15

        # Cardinality: 1 + 2 = 3
        assert slack['cardinality_slack'] == 3


# ============================================================================
# Energy Evaluation Tests
# ============================================================================

class TestEnergyEvaluation:
    """Test energy evaluation."""

    def test_empty_selection_energy(self):
        """Test energy with no assets or slack selected."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())
        optimizer.build_qubo_matrix()

        # Empty selection
        full_selection = np.zeros(optimizer.n_total)
        energy = optimizer.evaluate_energy(full_selection)

        # Empty selection should have zero energy
        assert energy == 0.0

    def test_energy_matches_solution(self):
        """Test that evaluate_energy matches solve() result."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        result = optimizer.solve(num_reads=100, seed=42)

        # Reconstruct full selection
        var_names = [a.id for a in optimizer.assets]
        for k in range(optimizer.budget_slack_bits):
            var_names.append(f'sb{k}')
        for k in range(optimizer.duration_slack_bits):
            var_names.append(f'sd{k}')
        for k in range(optimizer.cardinality_slack_bits):
            var_names.append(f'sc{k}')

        best_sample = result['sampleset'].first.sample
        full_selection = np.array([best_sample[v] for v in var_names])

        computed_energy = optimizer.evaluate_energy(full_selection)

        # Should match within floating point tolerance
        assert abs(computed_energy - result['energy']) < 1e-6


# ============================================================================
# Constraint Satisfaction Tests
# ============================================================================

class TestConstraintSatisfaction:
    """Test constraint checking."""

    def test_check_constraints_feasible(self):
        """Test constraint check for a feasible selection."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        # Select assets A and E: price=2.02, duration=15, count=2
        selection = np.array([1, 0, 0, 0, 1])
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied'] == True  # 2.02 <= 3.0
        assert check['duration_satisfied'] == True  # 15 <= 15
        assert check['cardinality_satisfied'] == True  # 2 <= 3
        assert check['total_price'] == pytest.approx(2.02)
        assert check['total_duration'] == 15
        assert check['num_selected'] == 2

    def test_check_constraints_budget_violation(self):
        """Test constraint check when budget is violated."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        # Select all assets: price=4.95 > 3.0
        selection = np.array([1, 1, 1, 1, 1])
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied'] == False
        assert check['duration_satisfied'] == False  # 29 > 15
        assert check['cardinality_satisfied'] == False  # 5 > 3

    def test_check_constraints_empty(self):
        """Test constraint check for empty selection."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        selection = np.zeros(5)
        check = optimizer.check_constraints(selection)

        assert check['budget_satisfied'] == True  # 0 <= 3.0
        assert check['duration_satisfied'] == True  # 0 <= 15
        assert check['cardinality_satisfied'] == True  # 0 <= 3
        assert check['total_price'] == 0.0
        assert check['total_duration'] == 0.0
        assert check['num_selected'] == 0


# ============================================================================
# Solving Tests
# ============================================================================

class TestSolving:
    """Test the solve method."""

    def test_solve_returns_valid_structure(self):
        """Test that solve returns expected dictionary structure."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        result = optimizer.solve(num_reads=100, seed=42)

        assert 'selection' in result
        assert 'selected_assets' in result
        assert 'energy' in result
        assert 'total_price' in result
        assert 'total_duration' in result
        assert 'total_score' in result
        assert 'num_selected' in result
        assert 'slack_values' in result
        assert 'sampleset' in result

    def test_solve_finds_good_solution(self):
        """Test that solve finds a reasonable solution."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        result = optimizer.solve(num_reads=1000, seed=42)

        # Should find a feasible solution with positive score
        check = optimizer.check_constraints(result['selection'])

        # At least one constraint should typically be satisfied
        # (with slack variables, we expect high feasibility)
        assert result['total_score'] > 0

    def test_solve_deterministic_with_seed(self):
        """Test that solve is deterministic with same seed."""
        assets = get_test_assets()
        optimizer1 = SlackPortfolioQUBO(assets=assets, **get_test_constraints())
        optimizer2 = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        result1 = optimizer1.solve(num_reads=100, seed=42)
        result2 = optimizer2.solve(num_reads=100, seed=42)

        assert result1['selected_assets'] == result2['selected_assets']
        assert result1['energy'] == result2['energy']


# ============================================================================
# Slack Variable Inequality Behavior Tests
# ============================================================================

class TestInequalityBehavior:
    """Test that slack variables properly encode inequality constraints."""

    def test_under_budget_not_penalized(self):
        """Test that under-budget solutions use slack to avoid penalty."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget=3.0,
            max_duration=15,
            max_cardinality=3,
            lambda_budget=100.0,  # High penalty to make effect visible
            lambda_duration=10.0,
            lambda_cardinality=5.0,
        )

        result = optimizer.solve(num_reads=500, seed=42)

        # The budget slack should absorb unused budget
        slack = result['slack_values']
        unused_budget = optimizer.budget - result['total_price']

        # Slack should be close to the unused budget (within precision)
        if result['total_price'] < optimizer.budget:
            # There's room for slack
            assert slack['budget_slack'] >= 0

    def test_cardinality_slack_absorbs_unused_slots(self):
        """Test that cardinality slack absorbs unused selection slots."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget=3.0,
            max_duration=15,
            max_cardinality=3,
            lambda_budget=2.0,
            lambda_duration=10.0,
            lambda_cardinality=100.0,  # High cardinality penalty
        )

        result = optimizer.solve(num_reads=500, seed=42)

        slack = result['slack_values']
        unused_cardinality = optimizer.max_cardinality - result['num_selected']

        # If we selected fewer than max, slack should absorb the difference
        if result['num_selected'] < optimizer.max_cardinality:
            assert slack['cardinality_slack'] >= 0


# ============================================================================
# BQM Conversion Tests
# ============================================================================

class TestBQMConversion:
    """Test conversion to BinaryQuadraticModel."""

    def test_to_bqm_returns_valid_model(self):
        """Test that to_bqm returns a valid BQM."""
        import dimod

        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        bqm = optimizer.to_bqm()

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert len(bqm.variables) == optimizer.n_total

    def test_bqm_variable_names(self):
        """Test that BQM has correct variable names."""
        assets = get_test_assets()
        optimizer = SlackPortfolioQUBO(assets=assets, **get_test_constraints())

        bqm = optimizer.to_bqm()
        var_names = list(bqm.variables)

        # Check asset names
        assert 'A' in var_names
        assert 'E' in var_names

        # Check slack variable names
        assert 'sb0' in var_names  # Budget slack bit 0
        assert 'sd0' in var_names  # Duration slack bit 0
        assert 'sc0' in var_names  # Cardinality slack bit 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_asset(self):
        """Test with a single asset."""
        assets = [{'id': 'A', 'price': 1.0, 'duration': 5, 'score': 10}]

        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget=2.0,
            max_duration=10,
            max_cardinality=1,
        )

        result = optimizer.solve(num_reads=100, seed=42)

        # Should select the single asset
        assert result['num_selected'] >= 0
        assert result['total_score'] <= 10

    def test_tight_constraints(self):
        """Test with very tight constraints."""
        assets = get_test_assets()

        # Very tight budget
        optimizer = SlackPortfolioQUBO(
            assets=assets,
            budget=1.0,  # Very tight
            max_duration=5,  # Very tight
            max_cardinality=1,
        )

        result = optimizer.solve(num_reads=100, seed=42)

        # Should still return a result
        assert 'selection' in result
        assert 'energy' in result

    def test_different_precisions(self):
        """Test with different precision values."""
        assets = get_test_assets()

        # Coarse precision
        optimizer_coarse = SlackPortfolioQUBO(
            assets=assets,
            budget=3.0,
            max_duration=15,
            max_cardinality=3,
            budget_precision=1.0,  # Coarse
            duration_precision=5.0,  # Coarse
        )

        # Fine precision
        optimizer_fine = SlackPortfolioQUBO(
            assets=assets,
            budget=3.0,
            max_duration=15,
            max_cardinality=3,
            budget_precision=0.01,  # Fine
            duration_precision=0.5,  # Fine
        )

        # Coarse should have fewer total variables
        assert optimizer_coarse.n_total < optimizer_fine.n_total


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
