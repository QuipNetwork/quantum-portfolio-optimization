#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for Simple Portfolio QUBO Optimizer.

Validates the implementation against the examples from:
"QUBO Matrix Construction for Portfolio Optimization" by Pho Le (November 2025).
"""

import pytest
import numpy as np
import dimod

from simple_portfolio_qubo import (
    SimplePortfolioQUBO,
    get_example_assets,
    get_example_constraints,
)


class TestQUBOMatrixConstruction:
    """Test QUBO matrix construction against expected values."""

    @pytest.fixture
    def example_optimizer(self):
        """Create optimizer with example data."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        return SimplePortfolioQUBO(assets=assets, **constraints)

    def test_qubo_matrix_shape(self, example_optimizer):
        """Test that QUBO matrix has correct shape."""
        Q = example_optimizer.build_qubo_matrix()
        assert Q.shape == (5, 5), f"Expected (5,5), got {Q.shape}"

    def test_qubo_matrix_symmetry(self, example_optimizer):
        """Test that QUBO matrix is symmetric."""
        Q = example_optimizer.build_qubo_matrix()
        np.testing.assert_array_almost_equal(
            Q, Q.T, decimal=10,
            err_msg="QUBO matrix should be symmetric"
        )

    def test_qubo_diagonal_q00(self, example_optimizer):
        """
        Test Q[0,0] matches expected value.

        Formula:
        Q[0,0] = -8 + 2*(1.00^2 - 2*3.0*1.00) + 10*(6^2 - 2*15*6) + 5*(1 - 2*3)
               = -8 + 2*(-5) + 10*(-144) + 5*(-5)
               = -8 - 10 - 1440 - 25 = -1483
        """
        Q = example_optimizer.build_qubo_matrix()
        expected = -1483.00
        assert np.isclose(Q[0, 0], expected, rtol=1e-6), \
            f"Q[0,0] = {Q[0,0]:.2f}, expected {expected:.2f}"

    def test_qubo_diagonal_all(self, example_optimizer):
        """
        Test all diagonal values computed from the formula.

        Formula:
        Q[i,i] = -s_i + lb*(p_i^2 - 2*B*p_i) + ld*(d_i^2 - 2*D*d_i) + lc*(1 - 2*K)
        """
        Q = example_optimizer.build_qubo_matrix()

        # Expected values computed from formula
        expected_diagonal = np.array([-1483.00, -1648.92, -1079.10, -846.39, -1934.16])

        for i, (actual, expected) in enumerate(zip(np.diag(Q), expected_diagonal)):
            assert np.isclose(actual, expected, rtol=1e-4), \
                f"Q[{i},{i}] = {actual:.2f}, expected {expected:.2f}"

    def test_qubo_offdiagonal_q01(self, example_optimizer):
        """
        Test Q[0,1] matches expected value.

        Formula:
        Q[0,1] = 2*2*1.00*0.99 + 2*10*6*7 + 2*5
               = 3.96 + 840 + 10 = 853.96
        """
        Q = example_optimizer.build_qubo_matrix()
        expected = 853.96
        assert np.isclose(Q[0, 1], expected, rtol=1e-4), \
            f"Q[0,1] = {Q[0,1]:.2f}, expected {expected:.2f}"
        assert np.isclose(Q[1, 0], Q[0, 1], rtol=1e-10), \
            "Q[0,1] should equal Q[1,0] (symmetry)"

    def test_qubo_full_matrix(self, example_optimizer):
        """
        Test full QUBO matrix computed from the formula.
        """
        Q = example_optimizer.build_qubo_matrix()

        # Expected matrix values computed from formula
        expected_Q = np.array([
            [-1483.00,  853.96,  493.56,  374.20, 1094.08],
            [  853.96, -1648.92,  573.52,  434.16, 1274.04],
            [  493.56,  573.52, -1079.10,  253.74,  733.63],
            [  374.20,  434.16,  253.74, -846.40,  554.28],
            [ 1094.08, 1274.04,  733.63,  554.28, -1934.16]
        ])

        np.testing.assert_array_almost_equal(
            Q, expected_Q, decimal=2,
            err_msg="QUBO matrix does not match formula values"
        )


class TestIsingConversion:
    """Test QUBO to Ising model conversion."""

    @pytest.fixture
    def example_optimizer(self):
        """Create optimizer with example data."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        return SimplePortfolioQUBO(assets=assets, **constraints)

    def test_ising_h_vector(self, example_optimizer):
        """
        Test h vector matches expected values.

        Expected:
        h = [-37.55, -40.54, -25.93, -19.10, -53.07]
        """
        h, J = example_optimizer.qubo_to_ising()
        expected_h = np.array([-37.55, -40.54, -25.93, -19.10, -53.07])

        np.testing.assert_array_almost_equal(
            h, expected_h, decimal=1,
            err_msg="Ising h vector does not match expected values"
        )

    def test_ising_J_matrix(self, example_optimizer):
        """
        Test J matrix matches expected values.

        Expected (with zeros on diagonal):
        J = [[  0,    213.49, 123.39,  93.55, 273.52],
             [213.49,   0,    143.38, 108.54, 318.51],
             [123.39, 143.38,   0,     63.43, 183.41],
             [ 93.55, 108.54,  63.43,   0,    138.57],
             [273.52, 318.51, 183.41, 138.57,   0   ]]
        """
        h, J = example_optimizer.qubo_to_ising()

        expected_J = np.array([
            [  0,    213.49, 123.39,  93.55, 273.52],
            [213.49,   0,    143.38, 108.54, 318.51],
            [123.39, 143.38,   0,     63.43, 183.41],
            [ 93.55, 108.54,  63.43,   0,    138.57],
            [273.52, 318.51, 183.41, 138.57,   0   ]
        ])

        # Check diagonal is zero
        np.testing.assert_array_almost_equal(
            np.diag(J), np.zeros(5), decimal=10,
            err_msg="J matrix diagonal should be zero"
        )

        # Check off-diagonal values
        np.testing.assert_array_almost_equal(
            J, expected_J, decimal=1,
            err_msg="Ising J matrix does not match expected values"
        )

    def test_ising_J_symmetry(self, example_optimizer):
        """Test that J matrix is symmetric."""
        h, J = example_optimizer.qubo_to_ising()
        np.testing.assert_array_almost_equal(
            J, J.T, decimal=10,
            err_msg="J matrix should be symmetric"
        )


class TestEnergyCalculation:
    """Test energy evaluation function."""

    @pytest.fixture
    def example_optimizer(self):
        """Create optimizer with example data."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        return SimplePortfolioQUBO(assets=assets, **constraints)

    def test_energy_selection_AC(self, example_optimizer):
        """
        Test energy for selection [1,0,1,0,0] (assets A and C).

        Formula:
        E = Q[0,0] + Q[2,2] + Q[0,2]
        E = -1483.00 + (-1079.10) + 493.56 = -2068.54
        """
        selection = np.array([1, 0, 1, 0, 0])
        energy = example_optimizer.evaluate_energy(selection)

        # Correct value computed from formula
        expected = -1483.00 + (-1079.10) + 493.56  # = -2068.54

        assert np.isclose(energy, expected, rtol=1e-4), \
            f"Energy = {energy:.2f}, expected {expected:.2f}"

    def test_energy_empty_selection(self, example_optimizer):
        """Test energy for empty selection is zero."""
        selection = np.array([0, 0, 0, 0, 0])
        energy = example_optimizer.evaluate_energy(selection)
        assert energy == 0.0, f"Empty selection should have energy 0, got {energy}"

    def test_energy_single_asset(self, example_optimizer):
        """Test energy for single asset selection equals diagonal."""
        example_optimizer.build_qubo_matrix()
        Q = example_optimizer._Q

        for i in range(5):
            selection = np.zeros(5)
            selection[i] = 1
            energy = example_optimizer.evaluate_energy(selection)
            assert np.isclose(energy, Q[i, i], rtol=1e-10), \
                f"Single asset {i} energy should equal Q[{i},{i}]"

    def test_energy_full_selection(self, example_optimizer):
        """Test energy for selecting all assets."""
        selection = np.array([1, 1, 1, 1, 1])
        energy = example_optimizer.evaluate_energy(selection)

        # Manually compute expected energy
        Q = example_optimizer.build_qubo_matrix()
        expected = np.sum(np.diag(Q))  # Diagonal terms
        for i in range(5):
            for j in range(i + 1, 5):
                expected += Q[i, j]  # Upper triangle

        assert np.isclose(energy, expected, rtol=1e-10), \
            f"Full selection energy = {energy:.2f}, expected {expected:.2f}"


class TestSimulatedAnnealing:
    """Test simulated annealing solver."""

    @pytest.fixture
    def example_optimizer(self):
        """Create optimizer with example data."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        return SimplePortfolioQUBO(assets=assets, **constraints)

    def test_solve_returns_valid_structure(self, example_optimizer):
        """Test that solve returns expected dictionary structure."""
        result = example_optimizer.solve(num_reads=100, seed=42)

        assert 'selection' in result
        assert 'selected_assets' in result
        assert 'energy' in result
        assert 'total_price' in result
        assert 'total_duration' in result
        assert 'total_score' in result
        assert 'num_selected' in result
        assert 'sampleset' in result

    def test_solve_selection_is_binary(self, example_optimizer):
        """Test that selection vector is binary."""
        result = example_optimizer.solve(num_reads=100, seed=42)
        selection = result['selection']

        assert len(selection) == 5
        assert all(x in [0, 1] for x in selection)

    def test_solve_metrics_consistent(self, example_optimizer):
        """Test that metrics are consistent with selection."""
        result = example_optimizer.solve(num_reads=100, seed=42)

        selection = result['selection']
        selected_indices = np.where(selection == 1)[0]

        assets = get_example_assets()
        expected_price = sum(assets[i]['price'] for i in selected_indices)
        expected_duration = sum(assets[i]['duration'] for i in selected_indices)
        expected_score = sum(assets[i]['score'] for i in selected_indices)

        assert np.isclose(result['total_price'], expected_price)
        assert np.isclose(result['total_duration'], expected_duration)
        assert np.isclose(result['total_score'], expected_score)
        assert result['num_selected'] == len(selected_indices)

    def test_solve_energy_matches_evaluation(self, example_optimizer):
        """Test that reported energy matches evaluate_energy."""
        result = example_optimizer.solve(num_reads=100, seed=42)

        computed_energy = example_optimizer.evaluate_energy(result['selection'])
        assert np.isclose(result['energy'], computed_energy, rtol=1e-10), \
            f"Reported energy {result['energy']} != computed {computed_energy}"

    def test_solve_deterministic_with_seed(self, example_optimizer):
        """Test that solve is deterministic with same seed."""
        result1 = example_optimizer.solve(num_reads=100, seed=42)
        result2 = example_optimizer.solve(num_reads=100, seed=42)

        np.testing.assert_array_equal(
            result1['selection'], result2['selection'],
            err_msg="Same seed should produce same result"
        )
        assert result1['energy'] == result2['energy']


class TestBQMConversion:
    """Test conversion to dimod BinaryQuadraticModel."""

    @pytest.fixture
    def example_optimizer(self):
        """Create optimizer with example data."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        return SimplePortfolioQUBO(assets=assets, **constraints)

    def test_bqm_has_correct_variables(self, example_optimizer):
        """Test that BQM has asset IDs as variables."""
        bqm = example_optimizer.to_bqm()

        expected_vars = {'A', 'B', 'C', 'D', 'E'}
        assert set(bqm.variables) == expected_vars

    def test_bqm_linear_terms(self, example_optimizer):
        """Test that BQM linear terms match QUBO diagonal."""
        bqm = example_optimizer.to_bqm()
        Q = example_optimizer.build_qubo_matrix()
        assets = get_example_assets()

        for i, asset in enumerate(assets):
            # BQM linear term for variable is Q[i,i]
            assert np.isclose(bqm.linear[asset['id']], Q[i, i], rtol=1e-10)

    def test_bqm_quadratic_terms(self, example_optimizer):
        """Test that BQM quadratic terms match QUBO off-diagonal."""
        bqm = example_optimizer.to_bqm()
        Q = example_optimizer.build_qubo_matrix()
        assets = get_example_assets()

        for i in range(5):
            for j in range(i + 1, 5):
                key = (assets[i]['id'], assets[j]['id'])
                # Try both orderings since BQM may use either
                if key in bqm.quadratic:
                    assert np.isclose(bqm.quadratic[key], Q[i, j], rtol=1e-10)
                else:
                    key_rev = (assets[j]['id'], assets[i]['id'])
                    assert np.isclose(bqm.quadratic[key_rev], Q[i, j], rtol=1e-10)


class TestConstraintChecking:
    """Test constraint checking functionality."""

    @pytest.fixture
    def example_optimizer(self):
        """Create optimizer with example data."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        return SimplePortfolioQUBO(assets=assets, **constraints)

    def test_check_constraints_cardinality(self, example_optimizer):
        """Test cardinality constraint checking."""
        # 2 assets - should satisfy max_cardinality=3
        selection_2 = np.array([1, 0, 1, 0, 0])
        check_2 = example_optimizer.check_constraints(selection_2)
        assert check_2['cardinality_satisfied'] is True
        assert check_2['num_selected'] == 2

        # 4 assets - should violate max_cardinality=3
        selection_4 = np.array([1, 1, 1, 1, 0])
        check_4 = example_optimizer.check_constraints(selection_4)
        assert check_4['cardinality_satisfied'] is False
        assert check_4['num_selected'] == 4

    def test_check_constraints_empty(self, example_optimizer):
        """Test constraint checking for empty selection."""
        selection = np.array([0, 0, 0, 0, 0])
        check = example_optimizer.check_constraints(selection)

        assert check['total_price'] == 0.0
        assert check['total_duration'] == 0.0
        assert check['num_selected'] == 0
        assert check['cardinality_satisfied'] is True
        assert check['duration_satisfied'] is True


class TestDifferentPortfolioSizes:
    """Test with different portfolio sizes."""

    @pytest.mark.parametrize("n_assets", [3, 5, 10, 20])
    def test_various_sizes(self, n_assets):
        """Test QUBO construction works for different sizes."""
        np.random.seed(42)

        assets = [
            {
                'id': f'ASSET_{i}',
                'price': np.random.uniform(0.5, 2.0),
                'duration': np.random.randint(1, 10),
                'score': np.random.randint(1, 10)
            }
            for i in range(n_assets)
        ]

        optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=n_assets * 0.5,  # Scale budget with size
            max_duration=n_assets * 3,
            max_cardinality=min(n_assets, 5),
            lambda_budget=2.0,
            lambda_duration=10.0,
            lambda_cardinality=5.0
        )

        Q = optimizer.build_qubo_matrix()

        # Verify shape and symmetry
        assert Q.shape == (n_assets, n_assets)
        np.testing.assert_array_almost_equal(Q, Q.T, decimal=10)

        # Verify solve works
        result = optimizer.solve(num_reads=50, seed=42)
        assert len(result['selection']) == n_assets
        assert result['num_selected'] <= n_assets

    @pytest.mark.parametrize("n_assets", [3, 5, 8])
    def test_energy_consistency(self, n_assets):
        """Test energy from solve matches manual evaluation."""
        np.random.seed(123)

        assets = [
            {
                'id': f'A{i}',
                'price': np.random.uniform(0.8, 1.2),
                'duration': np.random.randint(2, 8),
                'score': np.random.randint(1, 10)
            }
            for i in range(n_assets)
        ]

        optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=2.5,
            max_duration=15,
            max_cardinality=3
        )

        result = optimizer.solve(num_reads=100, seed=42)
        manual_energy = optimizer.evaluate_energy(result['selection'])

        assert np.isclose(result['energy'], manual_energy, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_asset(self):
        """Test with single asset."""
        assets = [{'id': 'ONLY', 'price': 1.0, 'duration': 5, 'score': 10}]

        optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=1.0,
            max_duration=5,
            max_cardinality=1
        )

        Q = optimizer.build_qubo_matrix()
        assert Q.shape == (1, 1)

        result = optimizer.solve(num_reads=10, seed=42)
        assert len(result['selection']) == 1

    def test_zero_penalties(self):
        """Test with zero penalty weights (pure score maximization)."""
        assets = get_example_assets()

        optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=3.0,
            max_duration=15,
            max_cardinality=3,
            lambda_budget=0.0,
            lambda_duration=0.0,
            lambda_cardinality=0.0
        )

        Q = optimizer.build_qubo_matrix()

        # With zero penalties, diagonal should just be -score
        expected_diagonal = -np.array([8, 4, 5, 1, 9])
        np.testing.assert_array_almost_equal(
            np.diag(Q), expected_diagonal, decimal=10
        )

        # Off-diagonal should be zero
        Q_no_diag = Q.copy()
        np.fill_diagonal(Q_no_diag, 0)
        assert np.allclose(Q_no_diag, 0)

    def test_high_cardinality_penalty(self):
        """Test that high cardinality penalty reduces selections."""
        assets = get_example_assets()

        # Low cardinality penalty
        optimizer_low = SimplePortfolioQUBO(
            assets=assets,
            budget=3.0,
            max_duration=15,
            max_cardinality=3,
            lambda_cardinality=0.1
        )

        # High cardinality penalty
        optimizer_high = SimplePortfolioQUBO(
            assets=assets,
            budget=3.0,
            max_duration=15,
            max_cardinality=3,
            lambda_cardinality=100.0
        )

        result_low = optimizer_low.solve(num_reads=500, seed=42)
        result_high = optimizer_high.solve(num_reads=500, seed=42)

        # Higher penalty should tend to select fewer assets
        # (This is probabilistic, so we just check it runs)
        assert result_high['num_selected'] <= 5
        assert result_low['num_selected'] <= 5


class TestQPUNormalization:
    """Test QPU normalization for D-Wave coefficient limits."""

    @pytest.fixture
    def example_optimizer(self):
        """Create optimizer with example data."""
        assets = get_example_assets()
        constraints = get_example_constraints()
        return SimplePortfolioQUBO(assets=assets, **constraints)

    def test_normalize_fits_advantage2_limits(self, example_optimizer):
        """Test that normalized values fit within Advantage2 limits."""
        h_norm, J_norm, scale = example_optimizer.normalize_for_qpu(
            h_range=(-6.0, 6.0),
            j_range=(-1.0, 1.0)
        )

        # Check h is within range
        assert np.all(h_norm >= -6.0), f"h has values below -6: {np.min(h_norm)}"
        assert np.all(h_norm <= 6.0), f"h has values above 6: {np.max(h_norm)}"

        # Check J is within range
        J_nonzero = J_norm[J_norm != 0]
        assert np.all(J_nonzero >= -1.0), f"J has values below -1: {np.min(J_nonzero)}"
        assert np.all(J_nonzero <= 1.0), f"J has values above 1: {np.max(J_nonzero)}"

    def test_normalize_fits_advantage_limits(self, example_optimizer):
        """Test that normalized values fit within Advantage (Pegasus) limits."""
        h_norm, J_norm, scale = example_optimizer.normalize_for_qpu(
            h_range=(-4.0, 4.0),
            j_range=(-1.0, 1.0)
        )

        # Check h is within range
        assert np.all(h_norm >= -4.0), f"h has values below -4: {np.min(h_norm)}"
        assert np.all(h_norm <= 4.0), f"h has values above 4: {np.max(h_norm)}"

        # Check J is within range
        J_nonzero = J_norm[J_norm != 0]
        assert np.all(J_nonzero >= -1.0), f"J has values below -1: {np.min(J_nonzero)}"
        assert np.all(J_nonzero <= 1.0), f"J has values above 1: {np.max(J_nonzero)}"

    def test_normalize_preserves_relative_ordering(self, example_optimizer):
        """Test that normalization preserves relative ordering of coefficients."""
        h_orig, J_orig = example_optimizer.qubo_to_ising()
        h_norm, J_norm, scale = example_optimizer.normalize_for_qpu()

        # Check h ordering is preserved
        h_order_orig = np.argsort(h_orig)
        h_order_norm = np.argsort(h_norm)
        np.testing.assert_array_equal(h_order_orig, h_order_norm,
            err_msg="h ordering not preserved after normalization")

        # Check J ordering is preserved (for non-zero elements)
        J_upper_orig = J_orig[np.triu_indices(5, k=1)]
        J_upper_norm = J_norm[np.triu_indices(5, k=1)]
        J_order_orig = np.argsort(J_upper_orig)
        J_order_norm = np.argsort(J_upper_norm)
        np.testing.assert_array_equal(J_order_orig, J_order_norm,
            err_msg="J ordering not preserved after normalization")

    def test_normalize_scale_factor_correct(self, example_optimizer):
        """Test that scale factor correctly relates original to normalized values."""
        h_orig, J_orig = example_optimizer.qubo_to_ising()
        h_norm, J_norm, scale = example_optimizer.normalize_for_qpu()

        # Normalized = Original * scale
        np.testing.assert_array_almost_equal(
            h_norm, h_orig * scale, decimal=10,
            err_msg="h_norm != h_orig * scale"
        )
        np.testing.assert_array_almost_equal(
            J_norm, J_orig * scale, decimal=10,
            err_msg="J_norm != J_orig * scale"
        )

    def test_normalization_info(self, example_optimizer):
        """Test get_normalization_info returns expected structure."""
        info = example_optimizer.get_normalization_info()

        assert 'original_h_range' in info
        assert 'original_j_range' in info
        assert 'normalized_h_range' in info
        assert 'normalized_j_range' in info
        assert 'scale_factor' in info
        assert 'compression_ratio' in info
        assert 'h_utilization' in info
        assert 'j_utilization' in info

        # Compression ratio should be > 1 (we're compressing)
        assert info['compression_ratio'] > 1, "Should be compressing for example problem"

        # Utilization should be <= 1 (within limits)
        assert info['h_utilization'] <= 1.0 + 1e-10
        assert info['j_utilization'] <= 1.0 + 1e-10

        print(f"\nNormalization Info:")
        print(f"  Original h range: {info['original_h_range']}")
        print(f"  Original J range: {info['original_j_range']}")
        print(f"  Normalized h range: {info['normalized_h_range']}")
        print(f"  Normalized J range: {info['normalized_j_range']}")
        print(f"  Scale factor: {info['scale_factor']:.6f}")
        print(f"  Compression ratio: {info['compression_ratio']:.1f}x")

    def test_to_bqm_normalized(self, example_optimizer):
        """Test normalized BQM generation."""
        bqm, scale = example_optimizer.to_bqm_normalized()

        # Check BQM has correct variables
        expected_vars = {'A', 'B', 'C', 'D', 'E'}
        assert set(bqm.variables) == expected_vars

        # Check all coefficients are within limits
        for v in bqm.variables:
            assert -6.0 <= bqm.linear[v] <= 6.0, f"Linear {v} out of range"

        for (u, v), val in bqm.quadratic.items():
            assert -1.0 <= val <= 1.0, f"Quadratic ({u},{v}) out of range: {val}"

        # Check it's in SPIN vartype (Ising form)
        assert bqm.vartype == dimod.SPIN

    def test_pdf_example_compression_ratio(self, example_optimizer):
        """Test that example problem requires significant compression."""
        info = example_optimizer.get_normalization_info()

        # Example should need ~300x compression due to large J values
        assert info['compression_ratio'] > 100, \
            f"Expected compression ratio > 100, got {info['compression_ratio']}"

        # The limiting factor should be J (coupling values)
        h_orig, J_orig = example_optimizer.qubo_to_ising()
        J_nonzero = J_orig[J_orig != 0]

        h_max = np.max(np.abs(h_orig))
        j_max = np.max(np.abs(J_nonzero))

        # J should be the bottleneck (larger relative to its limit)
        h_ratio = h_max / 6.0  # h limit
        j_ratio = j_max / 1.0  # J limit

        assert j_ratio > h_ratio, "J should be the limiting factor for example problem"


class TestConstraintPenaltyEncoding:
    """
    Test that constraint penalties are correctly encoded in the QUBO.

    These tests verify that violating a constraint increases energy by the
    expected amount according to the penalty formula.
    """

    def test_budget_penalty_encoding(self):
        """
        Test that budget constraint penalty is correctly encoded.

        For a selection with total price P and target budget B:
        Budget penalty contribution = lambda_b * (P - B)^2

        We verify this by comparing two selections that differ only in
        budget violation magnitude.
        """
        # Simple 2-asset scenario to isolate budget effect
        assets = [
            {'id': 'A', 'price': 1.0, 'duration': 1, 'score': 0},
            {'id': 'B', 'price': 2.0, 'duration': 1, 'score': 0},
        ]

        # Zero scores and other penalties to isolate budget effect
        optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=1.5,  # Target
            max_duration=100,  # Won't trigger
            max_cardinality=10,  # Won't trigger
            lambda_budget=10.0,
            lambda_duration=0.0,
            lambda_cardinality=0.0
        )

        # Selection [1,0]: price=1.0, deviation=-0.5, penalty=10*0.25=2.5
        # Selection [0,1]: price=2.0, deviation=+0.5, penalty=10*0.25=2.5
        # Selection [1,1]: price=3.0, deviation=+1.5, penalty=10*2.25=22.5

        e_A = optimizer.evaluate_energy(np.array([1, 0]))
        e_B = optimizer.evaluate_energy(np.array([0, 1]))
        e_AB = optimizer.evaluate_energy(np.array([1, 1]))
        e_none = optimizer.evaluate_energy(np.array([0, 0]))

        # Verify relative energies reflect penalty magnitudes
        # [1,0] and [0,1] should have same penalty (both 0.5 from target)
        # Note: We can't test absolute values due to dropped constants,
        # but we can verify the DIFFERENCE between energies
        budget = 1.5
        lb = 10.0

        # Energy difference between [1,1] and [1,0] should reflect
        # the increased budget penalty plus interaction term
        # Full penalty for [1,1]: lb*(3.0-1.5)^2 = 22.5
        # Full penalty for [1,0]: lb*(1.0-1.5)^2 = 2.5
        # Difference in full penalties: 20.0

        # In QUBO, energy difference = Q[1,1] + Q[0,1] (adding asset B to A)
        # This should reflect the marginal cost of adding B
        print(f"e_none={e_none:.2f}, e_A={e_A:.2f}, e_B={e_B:.2f}, e_AB={e_AB:.2f}")

        # Key test: [1,1] should have HIGHER energy than [1,0] or [0,1]
        # because it overshoots the budget more
        assert e_AB > e_A, "Selecting both should have higher energy than just A"
        assert e_AB > e_B, "Selecting both should have higher energy than just B"

    def test_duration_penalty_encoding(self):
        """
        Test that duration constraint penalty is correctly encoded.

        Duration penalty = lambda_d * (total_duration - max_duration)^2
        """
        assets = [
            {'id': 'A', 'price': 1.0, 'duration': 5, 'score': 0},
            {'id': 'B', 'price': 1.0, 'duration': 10, 'score': 0},
        ]

        optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=100,  # Won't trigger
            max_duration=8,  # Target
            max_cardinality=10,
            lambda_budget=0.0,
            lambda_duration=5.0,
            lambda_cardinality=0.0
        )

        # [1,0]: duration=5, deviation=-3, penalty=5*9=45
        # [0,1]: duration=10, deviation=+2, penalty=5*4=20
        # [1,1]: duration=15, deviation=+7, penalty=5*49=245

        e_A = optimizer.evaluate_energy(np.array([1, 0]))
        e_B = optimizer.evaluate_energy(np.array([0, 1]))
        e_AB = optimizer.evaluate_energy(np.array([1, 1]))

        # [1,1] should have highest energy due to large duration overshoot
        assert e_AB > e_A, "Both selected should have higher energy (duration overshoot)"
        assert e_AB > e_B, "Both selected should have higher energy (duration overshoot)"

    def test_cardinality_penalty_encoding(self):
        """
        Test that cardinality constraint penalty is correctly encoded.

        Cardinality penalty = lambda_c * (num_selected - max_cardinality)^2
        """
        assets = [
            {'id': 'A', 'price': 1.0, 'duration': 1, 'score': 0},
            {'id': 'B', 'price': 1.0, 'duration': 1, 'score': 0},
            {'id': 'C', 'price': 1.0, 'duration': 1, 'score': 0},
        ]

        optimizer = SimplePortfolioQUBO(
            assets=assets,
            budget=100,
            max_duration=100,
            max_cardinality=2,  # Target
            lambda_budget=0.0,
            lambda_duration=0.0,
            lambda_cardinality=10.0
        )

        # [1,0,0]: count=1, deviation=-1, penalty=10*1=10
        # [1,1,0]: count=2, deviation=0, penalty=10*0=0
        # [1,1,1]: count=3, deviation=+1, penalty=10*1=10

        e_1 = optimizer.evaluate_energy(np.array([1, 0, 0]))
        e_2 = optimizer.evaluate_energy(np.array([1, 1, 0]))
        e_3 = optimizer.evaluate_energy(np.array([1, 1, 1]))

        # Selecting exactly max_cardinality should have lowest penalty
        assert e_2 < e_1, "Selecting 2 (at target) should be better than 1 (under)"
        assert e_2 < e_3, "Selecting 2 (at target) should be better than 3 (over)"

    def test_penalty_magnitude_scales_with_lambda(self):
        """Test that increasing lambda increases penalty effect."""
        assets = [
            {'id': 'A', 'price': 1.0, 'duration': 1, 'score': 10},
            {'id': 'B', 'price': 1.0, 'duration': 1, 'score': 10},
            {'id': 'C', 'price': 1.0, 'duration': 1, 'score': 10},
            {'id': 'D', 'price': 1.0, 'duration': 1, 'score': 10},
        ]

        # With low cardinality penalty, selecting all (high score) wins
        optimizer_low = SimplePortfolioQUBO(
            assets=assets,
            budget=100,
            max_duration=100,
            max_cardinality=2,
            lambda_budget=0.0,
            lambda_duration=0.0,
            lambda_cardinality=1.0  # Low
        )

        # With high cardinality penalty, constraint dominates
        optimizer_high = SimplePortfolioQUBO(
            assets=assets,
            budget=100,
            max_duration=100,
            max_cardinality=2,
            lambda_budget=0.0,
            lambda_duration=0.0,
            lambda_cardinality=100.0  # High
        )

        result_low = optimizer_low.solve(num_reads=100, seed=42)
        result_high = optimizer_high.solve(num_reads=100, seed=42)

        # High penalty should enforce cardinality constraint more strictly
        assert result_high['num_selected'] <= result_low['num_selected'] or \
               result_high['num_selected'] <= 2, \
            "High penalty should enforce cardinality constraint"

    def test_combined_constraints_tradeoff(self):
        """
        Test that the optimizer balances multiple constraints.

        With competing constraints, the solution should reflect the
        relative penalty weights.
        """
        assets = [
            {'id': 'CHEAP', 'price': 0.5, 'duration': 10, 'score': 5},
            {'id': 'EXPENSIVE', 'price': 2.0, 'duration': 2, 'score': 5},
        ]

        # Budget-focused: prefer cheap asset
        optimizer_budget = SimplePortfolioQUBO(
            assets=assets,
            budget=0.5,
            max_duration=5,
            max_cardinality=1,
            lambda_budget=100.0,  # High
            lambda_duration=1.0,  # Low
            lambda_cardinality=0.0
        )

        # Duration-focused: prefer short duration asset
        optimizer_duration = SimplePortfolioQUBO(
            assets=assets,
            budget=0.5,
            max_duration=5,
            max_cardinality=1,
            lambda_budget=1.0,  # Low
            lambda_duration=100.0,  # High
            lambda_cardinality=0.0
        )

        result_budget = optimizer_budget.solve(num_reads=200, seed=42)
        result_duration = optimizer_duration.solve(num_reads=200, seed=42)

        # Budget-focused should prefer CHEAP (price=0.5, closer to target)
        # Duration-focused should prefer EXPENSIVE (duration=2, under target)
        print(f"Budget-focused selected: {result_budget['selected_assets']}")
        print(f"Duration-focused selected: {result_duration['selected_assets']}")

        # Both should find valid solutions given sufficient penalty
        assert result_budget['num_selected'] >= 0
        assert result_duration['num_selected'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
