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


def test_pick_best_feasible_picks_highest_score_among_feasible(pdf_optimizer):
    # PDF: A+E (score 17) > E only (score 9) > all-zeros (score 0).
    # All three are feasible; helper must return A+E.
    candidates = [
        np.array([0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 0, 1]),
        np.array([1, 0, 0, 0, 1]),
    ]
    best = pdf_optimizer._pick_best_feasible(candidates)
    assert best.tolist() == [1, 0, 0, 0, 1]


def test_pick_best_feasible_falls_back_to_repair_when_none_feasible(pdf_optimizer):
    # If every candidate is infeasible, helper must repair and return
    # SOMETHING feasible (not crash, not return an infeasible result).
    all_ones = np.ones(pdf_optimizer.n, dtype=int)
    best = pdf_optimizer._pick_best_feasible([all_ones])
    assert pdf_optimizer._is_feasible(best)


def test_pick_best_feasible_handles_empty_input(pdf_optimizer):
    # Edge case: empty candidate list returns all-zeros (vacuously feasible).
    best = pdf_optimizer._pick_best_feasible([])
    assert best.tolist() == [0] * pdf_optimizer.n


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


# --- solve_qubo_slack ---


def test_solve_qubo_slack_runs_on_pdf_example(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo_slack(seed=42)
    # Selection bits are asset-only (slack bits stripped) so shape is n.
    assert result['selection'].shape == (pdf_optimizer.n,)


def test_solve_qubo_slack_returns_binary_selection(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo_slack(seed=42)
    assert set(result['selection'].tolist()).issubset({0, 1})


def test_solve_qubo_slack_post_repair_yields_feasible(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo_slack(seed=42)
    assert result['is_feasible'] is True


def test_solve_qubo_slack_result_has_standard_shape(pdf_optimizer, require_qubosolver):
    result = pdf_optimizer.solve_qubo_slack(seed=42)
    for key in (
        'selection', 'selected_assets', 'energy',
        'total_price', 'total_duration', 'total_score',
        'num_selected', 'is_feasible',
    ):
        assert key in result


def test_solve_qubo_slack_rejects_oversized_problem():
    # n=9 must trigger the slack-matrix runtime cap before any
    # qubosolver call. This test does NOT require qubosolver to be
    # installed — the guard runs before the lazy import.
    assets = [
        {'id': f'A{i}', 'price': 0.5, 'duration': 1, 'score': 1}
        for i in range(9)
    ]
    opt = PasqalPortfolioOptimizer(
        assets=assets, budget=2.0, max_duration=5, max_cardinality=3,
    )
    with pytest.raises(ValueError, match="9"):
        opt.solve_qubo_slack()


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


# --- solve_pulser ---


@pytest.fixture
def require_pulser():
    pytest.importorskip("pulser")
    pytest.importorskip("qutip")


def test_solve_pulser_runs_on_pdf_example(pdf_optimizer, require_pulser):
    result = pdf_optimizer.solve_pulser(n_shots=20, seed=42)
    assert result['selection'].shape == (pdf_optimizer.n,)


def test_solve_pulser_returns_binary_selection(pdf_optimizer, require_pulser):
    result = pdf_optimizer.solve_pulser(n_shots=20, seed=42)
    assert set(result['selection'].tolist()).issubset({0, 1})


def test_solve_pulser_returns_feasible_after_repair(pdf_optimizer, require_pulser):
    result = pdf_optimizer.solve_pulser(n_shots=20, seed=42)
    assert result['is_feasible'] is True


def test_solve_pulser_result_has_standard_shape(pdf_optimizer, require_pulser):
    result = pdf_optimizer.solve_pulser(n_shots=20, seed=42)
    for key in (
        'selection', 'selected_assets', 'energy',
        'total_price', 'total_duration', 'total_score',
        'num_selected', 'is_feasible',
    ):
        assert key in result


def test_solve_pulser_rejects_oversized_problem():
    # n=13 must trigger the Qutip-emulation hard cap before any
    # Pulser objects get constructed. This test does NOT require pulser
    # to be installed — the guard runs first.
    assets = [
        {'id': f'A{i}', 'price': 0.5, 'duration': 1, 'score': 1}
        for i in range(13)
    ]
    opt = PasqalPortfolioOptimizer(
        assets=assets, budget=2.0, max_duration=5, max_cardinality=3,
    )
    with pytest.raises(ValueError, match="13"):
        opt.solve_pulser()


# --- integration ---


def test_all_three_paths_run_on_pdf_example(pdf_optimizer):
    """All three Pasqal paths produce a feasible selection on the PDF example.

    Comparability note: only solve_qubo is expected to be competitive
    on score. MIS and Pulser only need to be feasible.
    """
    pytest.importorskip("qubosolver")
    pytest.importorskip("mis")
    pytest.importorskip("pulser")

    qubo = pdf_optimizer.solve_qubo(seed=42)
    mis = pdf_optimizer.solve_mis(seed=42)
    pulser_r = pdf_optimizer.solve_pulser(n_shots=20, seed=42)

    for label, result in (("qubo", qubo), ("mis", mis), ("pulser", pulser_r)):
        assert result['is_feasible'], f"{label} returned infeasible result"

    # Comparable row should match or beat trivial all-zeros (score 0).
    assert qubo['total_score'] > 0


# --- solve_pulser_dmm (weighted MIS) / solve_pulser_qubo (full QUBO) ---
#
# The epsilon-weight and size-cap tests do NOT require pulser to be
# installed (pure numpy / guard logic). The embedding + solve tests do.


def test_score_only_epsilon_is_score_ordering(pdf_optimizer):
    # The score-only DMM weighting solve_pulser_dmm uses
    # (_normalize_epsilon(-score), DEVNOTES §3a) must map the
    # highest-score asset to epsilon=0 (most encouraged) and the
    # lowest-score asset to epsilon=1 (low epsilon = desirable).
    eps = pdf_optimizer._normalize_epsilon(-pdf_optimizer.scores)
    scores = pdf_optimizer.scores
    assert eps.shape == (pdf_optimizer.n,)
    assert 0.0 <= eps.min() <= eps.max() <= 1.0
    assert np.argmin(eps) == np.argmax(scores)
    assert np.argmax(eps) == np.argmin(scores)


def test_normalize_epsilon_handles_constant_diagonal(pdf_optimizer):
    # A flat diagonal has zero range -> all-zero epsilon (no NaNs).
    eps = pdf_optimizer._normalize_epsilon(np.full(4, 3.0))
    assert np.all(eps == 0.0)


def test_solve_pulser_dmm_rejects_oversized_problem():
    # n=11 must trip the DMM cap (10) before any pulser import.
    assets = [
        {'id': f'A{i}', 'price': 0.5, 'duration': 1, 'score': 1}
        for i in range(11)
    ]
    opt = PasqalPortfolioOptimizer(
        assets=assets, budget=2.0, max_duration=5, max_cardinality=3,
    )
    with pytest.raises(ValueError, match="11"):
        opt.solve_pulser_dmm()


def test_solve_pulser_qubo_rejects_oversized_problem():
    # n=11 must trip the QUBO-embedding cap (10) before any pulser import.
    assets = [
        {'id': f'A{i}', 'price': 0.5, 'duration': 1, 'score': 1}
        for i in range(11)
    ]
    opt = PasqalPortfolioOptimizer(
        assets=assets, budget=2.0, max_duration=5, max_cardinality=3,
    )
    with pytest.raises(ValueError, match="11"):
        opt.solve_pulser_qubo()


def test_solve_pulser_dmm_runs_on_pdf_example(pdf_optimizer, require_pulser):
    result = pdf_optimizer.solve_pulser_dmm(seed=42)
    assert result['is_feasible']
    sel = np.asarray(result['selection'])
    assert set(np.unique(sel)).issubset({0, 1})
    assert sel.shape == (pdf_optimizer.n,)


def test_solve_pulser_dmm_result_has_standard_shape(pdf_optimizer, require_pulser):
    result = pdf_optimizer.solve_pulser_dmm(seed=42)
    for key in (
        'selection', 'selected_assets', 'energy',
        'total_price', 'total_duration', 'total_score',
        'num_selected', 'is_feasible',
    ):
        assert key in result


def test_solve_pulser_qubo_runs_on_pdf_example(pdf_optimizer, require_pulser):
    # Manual full-QUBO embedding: off-diagonal -> positions, diagonal ->
    # DMM, pulse energy tied to the register's U_max. Must return a
    # feasible binary selection over the assets.
    result = pdf_optimizer.solve_pulser_qubo(seed=42)
    assert result['is_feasible']
    sel = np.asarray(result['selection'])
    assert sel.shape == (pdf_optimizer.n,)
    assert set(np.unique(sel)).issubset({0, 1})


def test_solve_pulser_qubo_embedding_adapts_energy_scale(pdf_optimizer, require_pulser):
    # The register embeds the full off-diagonal; the embedding must be
    # physical (all pairs >= min atom spacing, within radial limit) so
    # the U_max-tied pulse is well-defined.
    from scipy.spatial.distance import pdist
    Q = pdf_optimizer._qubo_optimizer.build_qubo_matrix()
    q_off = (Q + Q.T) / 2.0
    np.fill_diagonal(q_off, 0.0)
    register, device, coords = pdf_optimizer._embed_register_2d(q_off)
    d = pdist(coords)
    assert d.min() >= device.min_atom_distance - 1e-6
    assert np.linalg.norm(coords, axis=1).max() <= device.max_radial_distance


# --- weighted-MIS DMM internals (the recommended solve_pulser_dmm) ---


def test_conflict_pairs_identifies_expected_conflicts(pdf_optimizer):
    # On the PDF example the only infeasible pair is B,E (durations
    # 7+9=16 > 15). A,E (15 <= 15) must NOT be a conflict — it is the
    # optimum and must stay co-selectable.
    idx = {aid: i for i, aid in enumerate(pdf_optimizer.asset_ids)}
    pairs = set(pdf_optimizer._conflict_pairs())
    be = tuple(sorted((idx['B'], idx['E'])))
    ae = tuple(sorted((idx['A'], idx['E'])))
    assert be in pairs
    assert ae not in pairs


def test_dmm_conflict_register_frees_optimal_pair(pdf_optimizer, require_pulser):
    # The core fix: in the conflict-graph register the optimal pair A,E
    # (not a conflict) must sit OUTSIDE the blockade so both can be
    # excited, while the conflict pair B,E sits INSIDE it. This is the
    # geometric property the full-QUBO embedding violated (A,E blockaded).
    idx = {aid: i for i, aid in enumerate(pdf_optimizer.asset_ids)}
    register, device = pdf_optimizer._embed_conflict_register_2d(seed=42)
    coords = {i: np.array(register.qubits[f"q{i}"])
              for i in range(pdf_optimizer.n)}

    def dist(a, b):
        return float(np.linalg.norm(coords[idx[a]] - coords[idx[b]]))

    omega = 0.5 * device.channels["rydberg_global"].max_amp
    rb = device.rydberg_blockade_radius(omega)
    assert dist('A', 'E') > rb, "optimal pair A,E must be free (outside blockade)"
    assert dist('B', 'E') < rb, "conflict pair B,E must be blockaded"


def test_solve_pulser_dmm_returns_feasible(pdf_optimizer, require_pulser):
    # _pick_best_feasible is fallback-only (it reports the best
    # already-feasible sample, no repair-masking), so the exact score is
    # stochastic across emulator runs. The honest guarantee is that the
    # result is feasible with a valid selection.
    result = pdf_optimizer.solve_pulser_dmm(seed=42)
    assert result['is_feasible']
    sel = np.asarray(result['selection'])
    assert sel.shape == (pdf_optimizer.n,)
    assert set(np.unique(sel)).issubset({0, 1})
