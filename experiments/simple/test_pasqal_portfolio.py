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

"""Tests for PasqalPortfolioOptimizer (qubosolver / MIS / Pulser paths)."""

import numpy as np
import pytest

from pasqal_portfolio import PasqalPortfolioOptimizer
from simple_portfolio_qubo import get_example_assets, get_example_constraints


@pytest.fixture
def pdf_optimizer():
    """PasqalPortfolioOptimizer initialized with the PDF example problem."""
    assets = get_example_assets()
    c = get_example_constraints()
    return PasqalPortfolioOptimizer(
        assets=assets,
        budget=c['budget'],
        max_duration=c['max_duration'],
        max_cardinality=c['max_cardinality'],
        lambda_budget=c.get('lambda_budget', 2.0),
        lambda_duration=c.get('lambda_duration', 10.0),
        lambda_cardinality=c.get('lambda_cardinality', 5.0),
    )


def test_constructor_stores_problem_data(pdf_optimizer):
    assert pdf_optimizer.n == len(get_example_assets())
    assert pdf_optimizer.budget == get_example_constraints()['budget']
    assert isinstance(pdf_optimizer.prices, np.ndarray)
    assert isinstance(pdf_optimizer.scores, np.ndarray)


def test_check_constraints_on_empty_selection(pdf_optimizer):
    result = pdf_optimizer.check_constraints(np.zeros(pdf_optimizer.n, dtype=int))
    assert result['num_selected'] == 0
    assert result['budget_satisfied'] is True
    assert result['duration_satisfied'] is True
    assert result['cardinality_satisfied'] is True


def test_round_and_repair_returns_feasible(pdf_optimizer):
    # All ones is infeasible for the PDF example (over budget/cardinality);
    # repair must yield a feasible binary vector.
    continuous = np.ones(pdf_optimizer.n)
    repaired = pdf_optimizer._round_and_repair(continuous)
    assert set(repaired.tolist()).issubset({0, 1})
    assert pdf_optimizer._is_feasible(repaired)


# --- solve_qubo ---


@pytest.fixture(autouse=False)
def require_qubosolver():
    """Skip any test that uses this fixture when qubosolver is absent."""
    pytest.importorskip("qubosolver")


def test_solve_qubo_runs_on_pdf_example(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo(seed=42)
    assert 'selection' in result
    assert result['selection'].shape == (pdf_optimizer.n,)


def test_solve_qubo_returns_binary_selection(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo(seed=42)
    assert set(result['selection'].tolist()).issubset({0, 1})


def test_solve_qubo_result_has_standard_shape(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo(seed=42)
    for key in (
        'selection', 'selected_assets', 'energy',
        'total_price', 'total_duration', 'total_score',
        'num_selected', 'is_feasible',
    ):
        assert key in result, f"missing key: {key}"


def test_solve_qubo_post_repair_yields_feasible(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo(seed=42)
    # After _round_and_repair, the solver MUST return a feasible selection.
    assert result['is_feasible'] is True


def test_solve_qubo_energy_matches_selection(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo(seed=42)
    expected = pdf_optimizer._compute_energy(result['selection'])
    assert abs(result['energy'] - expected) < 1e-6


# --- solve_mis ---


@pytest.fixture
def require_mis_libs():
    pytest.importorskip("mis")
    pytest.importorskip("networkx")


def test_build_conflict_graph_pairs_over_budget(pdf_optimizer, require_mis_libs):
    g = pdf_optimizer._build_conflict_graph()
    p = pdf_optimizer.prices
    b = pdf_optimizer.budget
    for i in range(pdf_optimizer.n):
        for j in range(i + 1, pdf_optimizer.n):
            if p[i] + p[j] > b:
                assert g.has_edge(i, j), f"missing budget edge ({i},{j})"


def test_build_conflict_graph_pairs_over_duration(pdf_optimizer, require_mis_libs):
    g = pdf_optimizer._build_conflict_graph()
    d = pdf_optimizer.durations
    md = pdf_optimizer.max_duration
    for i in range(pdf_optimizer.n):
        for j in range(i + 1, pdf_optimizer.n):
            if d[i] + d[j] > md:
                assert g.has_edge(i, j), f"missing duration edge ({i},{j})"


def test_solve_mis_runs_on_pdf_example(pdf_optimizer, require_mis_libs):
    result = pdf_optimizer.solve_mis(seed=42)
    assert result['selection'].shape == (pdf_optimizer.n,)


def test_solve_mis_returns_binary_selection(pdf_optimizer, require_mis_libs):
    result = pdf_optimizer.solve_mis(seed=42)
    assert set(result['selection'].tolist()).issubset({0, 1})


def test_solve_mis_returns_feasible_after_post_selection(pdf_optimizer, require_mis_libs):
    result = pdf_optimizer.solve_mis(seed=42)
    assert result['is_feasible'] is True


def test_solve_mis_result_has_standard_shape(pdf_optimizer, require_mis_libs):
    result = pdf_optimizer.solve_mis(seed=42)
    for key in (
        'selection', 'selected_assets', 'energy',
        'total_price', 'total_duration', 'total_score',
        'num_selected', 'is_feasible',
    ):
        assert key in result
