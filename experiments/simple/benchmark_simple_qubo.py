#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Benchmark tool for comparing SimplePortfolioQUBO against classical optimizers.

Compares solution quality, constraint satisfaction, and runtime across:
- QUBO (Simulated Annealing)
- Brute Force (exact, for small problems)
- Greedy (score-per-price ratio)
- Random Sampling
- Branch and Bound (ILP via scipy if available)

Usage:
    python experiments/simple/benchmark_simple_qubo.py
    python experiments/simple/benchmark_simple_qubo.py --num-assets 10 --num-trials 5
    python experiments/simple/benchmark_simple_qubo.py --problem-set random --seed 42
"""

import argparse
import json
import math
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from itertools import combinations

# Load D-Wave Leap credentials from the repo-root .env (DWAVE_API_TOKEN,
# DWAVE_API_SOLVER, DWAVE_REGION_URL) so the QPU rows can authenticate.
# Explicit path so it works regardless of the cwd the benchmark runs from.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

from simple_portfolio_qubo import SimplePortfolioQUBO, get_example_assets, get_example_constraints
from slack_portfolio_qubo import SlackPortfolioQUBO
from cqm_portfolio import CQMPortfolioOptimizer
from nl_portfolio import NLPortfolioOptimizer

try:
    from qhd_portfolio import QHDPortfolioOptimizer
    _HAS_QHDOPT = True
except ImportError:
    _HAS_QHDOPT = False

try:
    from cuopt_portfolio import CuOptPortfolioOptimizer
    _HAS_CUOPT = True
except ImportError:
    _HAS_CUOPT = False

try:
    from phi_portfolio import PhiPortfolioOptimizer
    _HAS_PHISOLVE = True
except ImportError:
    _HAS_PHISOLVE = False

try:
    from pasqal_portfolio import PasqalPortfolioOptimizer
    _HAS_PASQAL = True
except ImportError:
    _HAS_PASQAL = False

# D-Wave embedding packages (optional — falls back to topology_cache.json)
try:
    import dwave.embedding.pegasus as pegasus_emb
    import dwave.embedding.zephyr as zephyr_emb
    import dwave_networkx as dnx
    from minorminer import find_embedding as _find_embedding
    _HAS_DWAVE_EMBEDDING = True
except ImportError:
    _HAS_DWAVE_EMBEDDING = False

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "topology_cache.json")
_topology_cache: Optional[Dict] = None


def _load_topology_cache() -> Dict:
    """Load pre-computed clique embeddings from cache file."""
    global _topology_cache
    if _topology_cache is None:
        with open(_CACHE_PATH) as f:
            _topology_cache = json.load(f)
    return _topology_cache


@dataclass
class BenchmarkResult:
    """Result from a single solver run."""
    solver_name: str
    selected_assets: List[str]
    total_score: float
    total_price: float
    total_duration: float
    num_selected: int
    energy: float
    runtime_ms: float
    budget_satisfied: bool
    duration_satisfied: bool
    cardinality_satisfied: bool
    is_feasible: bool


@dataclass
class QubitInfo:
    """Qubit requirements for a solver formulation."""
    solver_name: str
    asset_qubits: Optional[int] = None
    constraint_qubits: Optional[int] = None
    logical_qubits: Optional[int] = None
    # Clique embedding (K_n upper bound)
    pegasus_physical: Optional[int] = None
    pegasus_chain: Optional[int] = None
    zephyr_physical: Optional[int] = None
    zephyr_chain: Optional[int] = None
    # Actual BQM graph embedding (tighter, graph-aware)
    pegasus_actual: Optional[int] = None
    pegasus_actual_chain: Optional[int] = None
    zephyr_actual: Optional[int] = None
    zephyr_actual_chain: Optional[int] = None


# Cached topology graphs (built once)
_pegasus_graph = None
_zephyr_graph = None


def _get_topology_graph(topology: str):
    """Get or build a cached topology graph.

    Requires dwave_networkx. Returns None if not available.
    """
    if not _HAS_DWAVE_EMBEDDING:
        return None
    global _pegasus_graph, _zephyr_graph
    if topology == "pegasus":
        if _pegasus_graph is None:
            _pegasus_graph = dnx.pegasus_graph(16)
        return _pegasus_graph
    elif topology == "zephyr":
        if _zephyr_graph is None:
            _zephyr_graph = dnx.zephyr_graph(12)
        return _zephyr_graph
    raise ValueError(f"Unknown topology: {topology}")


def embed_bqm(bqm, topology: str = "pegasus", seed: int = 42):
    """Compute actual minor embedding for a BQM's interaction graph.

    Requires minorminer and dwave_networkx. Returns (None, None) if
    these packages are not installed.

    Args:
        bqm: A dimod BinaryQuadraticModel.
        topology: "pegasus" or "zephyr".
        seed: Random seed for reproducible embeddings.

    Returns:
        (total_physical_qubits, max_chain_length) or (None, None) if
        the embedding fails or packages are not available.
    """
    target = _get_topology_graph(topology)
    if target is None:
        return (None, None)
    source_edges = list(bqm.quadratic)
    if not source_edges:
        return (bqm.num_variables, 1)
    try:
        emb = _find_embedding(
            source_edges, target.edges(), random_seed=seed,
        )
    except (ValueError, RuntimeError):
        return (None, None)
    if not emb:
        return (None, None)
    physical = sum(len(chain) for chain in emb.values())
    max_chain = max(len(chain) for chain in emb.values())
    return (physical, max_chain)


# Cache for clique embeddings to avoid recomputation
_embedding_cache: Dict[Tuple[str, int], Tuple[int, int]] = {}


def get_clique_embedding(
    logical_qubits: int,
    topology: str = "pegasus",
) -> Tuple[int, int]:
    """Compute physical qubits and max chain length for a clique embedding.

    Uses D-Wave embedding packages if available, otherwise falls back to
    pre-computed results in topology_cache.json.

    Args:
        logical_qubits: Number of logical qubits (clique size).
        topology: "pegasus" (Advantage P16) or "zephyr" (Advantage2 Z12).

    Returns:
        (total_physical_qubits, max_chain_length).
        Returns (None, None) if the clique doesn't fit.
    """
    cache_key = (topology, logical_qubits)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # Try live computation first
    if _HAS_DWAVE_EMBEDDING:
        try:
            if topology == "pegasus":
                emb = pegasus_emb.find_clique_embedding(logical_qubits, 16)
            elif topology == "zephyr":
                emb = zephyr_emb.find_clique_embedding(logical_qubits, 12)
            else:
                raise ValueError(f"Unknown topology: {topology}")
        except ValueError:
            _embedding_cache[cache_key] = (None, None)
            return (None, None)

        if not emb:
            _embedding_cache[cache_key] = (None, None)
            return (None, None)

        physical = sum(len(chain) for chain in emb.values())
        max_chain = max(len(chain) for chain in emb.values())
        _embedding_cache[cache_key] = (physical, max_chain)
        return (physical, max_chain)

    # Fall back to cached results
    try:
        tc = _load_topology_cache()
        file_key = f"{topology}:{logical_qubits}"
        entry = tc["clique_embeddings"].get(file_key)
        if entry is not None:
            result = (entry[0], entry[1])
        else:
            result = (None, None)
        _embedding_cache[cache_key] = result
        return result
    except (FileNotFoundError, KeyError):
        _embedding_cache[cache_key] = (None, None)
        return (None, None)


def compute_qubit_requirements(
    assets: List[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> List[QubitInfo]:
    """Compute qubit requirements for all solver formulations.

    Args:
        assets: List of asset dictionaries.
        constraints: Problem constraints.

    Returns:
        List of QubitInfo for each solver type.
    """
    n = len(assets)
    budget = constraints['budget']
    max_duration = constraints['max_duration']
    max_cardinality = constraints['max_cardinality']

    results = []

    # Simple QUBO: n asset vars, 0 slack (soft penalties)
    simple_logical = n
    p_phys, p_chain = get_clique_embedding(simple_logical, "pegasus")
    z_phys, z_chain = get_clique_embedding(simple_logical, "zephyr")
    simple_opt = SimplePortfolioQUBO(assets=assets, **{
        'budget': budget,
        'max_duration': max_duration,
        'max_cardinality': max_cardinality,
        'lambda_budget': constraints.get('lambda_budget', 2.0),
        'lambda_duration': constraints.get('lambda_duration', 10.0),
        'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
    })
    simple_bqm = simple_opt.to_bqm()
    pa_phys, pa_chain = embed_bqm(simple_bqm, "pegasus")
    za_phys, za_chain = embed_bqm(simple_bqm, "zephyr")
    results.append(QubitInfo(
        solver_name="QUBO (SA/Filt/HighPen)",
        asset_qubits=n,
        constraint_qubits=0,
        logical_qubits=simple_logical,
        pegasus_physical=p_phys,
        pegasus_chain=p_chain,
        zephyr_physical=z_phys,
        zephyr_chain=z_chain,
        pegasus_actual=pa_phys,
        pegasus_actual_chain=pa_chain,
        zephyr_actual=za_phys,
        zephyr_actual_chain=za_chain,
    ))

    # Slack QUBO: n asset vars + slack bits
    slack_constraints = {
        'budget': budget,
        'max_duration': max_duration,
        'max_cardinality': max_cardinality,
        'lambda_budget': constraints.get('lambda_budget', 2.0),
        'lambda_duration': constraints.get('lambda_duration', 10.0),
        'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
    }
    slack_opt = SlackPortfolioQUBO(assets=assets, **slack_constraints)
    var_info = slack_opt.get_variable_info()
    slack_logical = var_info['n_total']
    slack_bits = (var_info['n_budget_slack']
                  + var_info['n_duration_slack']
                  + var_info['n_cardinality_slack'])
    p_phys, p_chain = get_clique_embedding(slack_logical, "pegasus")
    z_phys, z_chain = get_clique_embedding(slack_logical, "zephyr")
    slack_bqm = slack_opt.to_bqm()
    pa_phys, pa_chain = embed_bqm(slack_bqm, "pegasus")
    za_phys, za_chain = embed_bqm(slack_bqm, "zephyr")
    results.append(QubitInfo(
        solver_name="QUBO (Slack/Slack+Filt)",
        asset_qubits=n,
        constraint_qubits=slack_bits,
        logical_qubits=slack_logical,
        pegasus_physical=p_phys,
        pegasus_chain=p_chain,
        zephyr_physical=z_phys,
        zephyr_chain=z_chain,
        pegasus_actual=pa_phys,
        pegasus_actual_chain=pa_chain,
        zephyr_actual=za_phys,
        zephyr_actual_chain=za_chain,
    ))

    # CQM (SA->BQM): produces a real BQM that can be QPU-embedded
    cqm_opt = CQMPortfolioOptimizer(
        assets=assets, budget=budget,
        max_duration=max_duration, max_cardinality=max_cardinality,
    )
    bqm, _ = cqm_opt.to_bqm()
    cqm_bqm_vars = bqm.num_variables
    p_phys, p_chain = get_clique_embedding(cqm_bqm_vars, "pegasus")
    z_phys, z_chain = get_clique_embedding(cqm_bqm_vars, "zephyr")
    pa_phys, pa_chain = embed_bqm(bqm, "pegasus")
    za_phys, za_chain = embed_bqm(bqm, "zephyr")
    results.append(QubitInfo(
        solver_name="CQM (SA->BQM)",
        asset_qubits=n,
        constraint_qubits=cqm_bqm_vars - n,
        logical_qubits=cqm_bqm_vars,
        pegasus_physical=p_phys,
        pegasus_chain=p_chain,
        zephyr_physical=z_phys,
        zephyr_chain=z_chain,
        pegasus_actual=pa_phys,
        pegasus_actual_chain=pa_chain,
        zephyr_actual=za_phys,
        zephyr_actual_chain=za_chain,
    ))

    # Weight-encoding reference (QPO main project comparison)
    for n_levels, label in [(4, "Weight 4-level"), (8, "Weight 8-level")]:
        weight_logical = n * n_levels
        p_phys, p_chain = get_clique_embedding(
            weight_logical, "pegasus"
        )
        z_phys, z_chain = get_clique_embedding(
            weight_logical, "zephyr"
        )
        results.append(QubitInfo(
            solver_name=f"[ref] {label}",
            asset_qubits=n * n_levels,
            constraint_qubits=0,
            logical_qubits=weight_logical,
            pegasus_physical=p_phys,
            pegasus_chain=p_chain,
            zephyr_physical=z_phys,
            zephyr_chain=z_chain,
        ))

    return results


def print_qubit_table(qubit_info: List[QubitInfo], n_assets: int):
    """Print qubit requirements table with clique and actual embeddings."""
    actual = [q for q in qubit_info if not q.solver_name.startswith("[ref]")]
    refs = [q for q in qubit_info if q.solver_name.startswith("[ref]")]

    width = 135
    print(f"\n{'=' * width}")
    print(f" Qubit Requirements (n={n_assets} assets, binary selection encoding)")
    print(f"{'=' * width}")

    header = (
        f"{'Solver':<24} "
        f"{'Assets':>6} "
        f"{'Slack':>5} "
        f"{'Logical':>7}  "
        f"{'Peg Clique':>10} "
        f"{'Peg Actual':>10} "
        f"{'Zep Clique':>10} "
        f"{'Zep Actual':>10}  "
        f"{'Fits Adv?':>9} "
        f"{'Fits Adv2?':>10}"
    )
    print(header)
    print("-" * width)

    def fmt_emb(phys, chain):
        """Format embedding as 'phys / chain' or dash."""
        if phys is not None:
            return f"{phys} / {chain}"
        return "\u2014"

    def fmt_row(q: QubitInfo):
        asset_s = f"{q.asset_qubits:>6}" if q.asset_qubits is not None else f"{'\u2014':>6}"
        slack_s = f"{q.constraint_qubits:>5}" if q.constraint_qubits is not None else f"{'\u2014':>5}"
        logical_s = f"{q.logical_qubits:>7}" if q.logical_qubits is not None else f"{'\u2014':>7}"

        peg_cliq = fmt_emb(q.pegasus_physical, q.pegasus_chain)
        peg_act = fmt_emb(q.pegasus_actual, q.pegasus_actual_chain)
        zep_cliq = fmt_emb(q.zephyr_physical, q.zephyr_chain)
        zep_act = fmt_emb(q.zephyr_actual, q.zephyr_actual_chain)

        # Use best (smallest) embedding for "fits" check
        peg_candidates = [v for v in [q.pegasus_actual, q.pegasus_physical]
                          if v is not None]
        peg_best = min(peg_candidates) if peg_candidates else None
        zep_candidates = [v for v in [q.zephyr_actual, q.zephyr_physical]
                          if v is not None]
        zep_best = min(zep_candidates) if zep_candidates else None

        if peg_best is not None:
            fits_adv = "YES" if peg_best <= 5600 else "NO"
        else:
            fits_adv = "\u2014"

        if zep_best is not None:
            fits_adv2 = "YES" if zep_best <= 4400 else "NO"
        else:
            fits_adv2 = "\u2014"

        return (
            f"{q.solver_name:<24} "
            f"{asset_s} {slack_s} {logical_s}  "
            f"{peg_cliq:>10} {peg_act:>10} "
            f"{zep_cliq:>10} {zep_act:>10}  "
            f"{fits_adv:>9} {fits_adv2:>10}"
        )

    for q in actual:
        print(fmt_row(q))

    if refs:
        print("-" * width)
        print("  Reference: weight-encoded formulation "
              "(QPO main project)")
        for q in refs:
            print(fmt_row(q))

    print("-" * width)
    print("  Pegasus = Advantage P16 (~5,600 qubits) | "
          "Zephyr = Advantage2 Z12 (~4,400 qubits)")
    print("  Clique = K_n embedding (upper bound, fully-connected) | "
          "Actual = minorminer on BQM graph (tighter)")
    print("  Format: physical_qubits / max_chain_length")
    print()


def print_qubo_info(
    assets: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    max_full_matrix_n: int = 16,
):
    """Print the QUBO matrix and its key statistics.

    This is the exact problem object the penalty-matrix solvers consume:
    dwave-neal SA, cuOpt-QP, Phi-QUBO, and Pasqal-QUBO all receive this
    same Q. Pasqal's qubosolver in particular receives np.triu(Q) (the
    upper triangle) and embeds it onto a Rydberg atom register, mapping
    diagonal terms to per-atom detuning and off-diagonal terms to atom
    spacings. Inspecting Q here lets you reason about that embedding
    before it disappears into the solver.

    Args:
        assets: Asset dicts (id, price, duration, score).
        constraints: Dict with budget, max_duration, max_cardinality, and
            optional lambda_budget/duration/cardinality.
        max_full_matrix_n: Print the full matrix only up to this size;
            above it, print summary statistics only.
    """
    optimizer = SimplePortfolioQUBO(
        assets=assets,
        budget=constraints["budget"],
        max_duration=constraints["max_duration"],
        max_cardinality=constraints["max_cardinality"],
        lambda_budget=constraints.get("lambda_budget", 2.0),
        lambda_duration=constraints.get("lambda_duration", 10.0),
        lambda_cardinality=constraints.get("lambda_cardinality", 5.0),
    )

    Q = optimizer.build_qubo_matrix()
    Q_upper = np.triu(Q)
    n = optimizer.n
    asset_ids = [a.id for a in optimizer.assets]

    width = 100
    print(f"\n{'=' * width}")
    print(f" QUBO Matrix (n={n} assets, soft-penalty formulation)")
    print(f"{'=' * width}")
    print(f"  Penalty weights: lambda_budget={optimizer.lambda_b}, "
          f"lambda_duration={optimizer.lambda_d}, "
          f"lambda_cardinality={optimizer.lambda_c}")

    diag = np.diag(Q)
    off = Q_upper[np.triu_indices(n, k=1)]
    print(f"\n  Diagonal terms Q[i,i] (per-asset: -score + penalties):")
    for i, aid in enumerate(asset_ids):
        print(f"    {aid:<6} {diag[i]:>14.4f}")
    print(f"\n  Diagonal range:     [{diag.min():.4f}, {diag.max():.4f}]")
    if off.size:
        print(f"  Off-diagonal range: [{off.min():.4f}, {off.max():.4f}]  "
              f"(pairwise couplings, upper triangle)")

    if n <= max_full_matrix_n:
        print(f"\n  Full symmetric matrix Q:")
        header = "        " + "".join(f"{aid:>11}" for aid in asset_ids)
        print(header)
        for i, aid in enumerate(asset_ids):
            row = "".join(f"{Q[i, j]:>11.3f}" for j in range(n))
            print(f"    {aid:<4}{row}")
    else:
        print(f"\n  (Full matrix suppressed for n > {max_full_matrix_n}; "
              f"showing statistics only.)")

    # Ising view + QPU compression ratio: same numbers the README's
    # D-Wave normalization section discusses, and a proxy for how hard
    # the off-diagonal magnitudes are to realize physically (large
    # compression => couplings span a wide dynamic range).
    info = optimizer.get_normalization_info()
    print(f"\n  Ising (h, J) view:")
    print(f"    h range (original):  "
          f"[{info['original_h_range'][0]:.4f}, "
          f"{info['original_h_range'][1]:.4f}]")
    print(f"    J range (original):  "
          f"[{info['original_j_range'][0]:.4f}, "
          f"{info['original_j_range'][1]:.4f}]")
    print(f"    Compression to fit Advantage2 (h in [-6,6], J in [-1,1]): "
          f"{info['compression_ratio']:.1f}x")
    print()


def print_slack_qubo_info(
    assets: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    budget_precision: float = 1.0,
    duration_precision: float = 5.0,
    max_full_matrix_n: int = 16,
):
    """Print the slack-variable QUBO matrix and its qubit budget.

    This is the matrix the Pasqal-Slack path (solve_qubo_slack) feeds to
    qubosolver. Unlike the soft-penalty QUBO, it adds binary slack
    variables that turn each <= constraint into an exact equality, so
    the variable count grows beyond the n assets. The default
    precisions here (budget 1.0, duration 5.0) match solve_qubo_slack's
    coarse defaults so the qubit count shown equals what Pasqal-Slack
    actually builds. (The dwave-neal SA path uses finer 0.1 / 1.0
    precision, which produces more slack bits.)

    Args:
        assets: Asset dicts (id, price, duration, score).
        constraints: Dict with budget, max_duration, max_cardinality, and
            optional lambda_budget/duration/cardinality.
        budget_precision: Slack discretization step for budget.
        duration_precision: Slack discretization step for duration.
        max_full_matrix_n: Print the full matrix only up to this size.
    """
    optimizer = SlackPortfolioQUBO(
        assets=assets,
        budget=constraints["budget"],
        max_duration=constraints["max_duration"],
        max_cardinality=constraints["max_cardinality"],
        lambda_budget=constraints.get("lambda_budget", 2.0),
        lambda_duration=constraints.get("lambda_duration", 10.0),
        lambda_cardinality=constraints.get("lambda_cardinality", 5.0),
        budget_precision=budget_precision,
        duration_precision=duration_precision,
    )

    vi = optimizer.get_variable_info()
    Q = optimizer.build_qubo_matrix()
    nt = optimizer.n_total

    width = 100
    print(f"\n{'=' * width}")
    print(f" Slack-Variable QUBO Matrix (Pasqal-Slack path)")
    print(f"{'=' * width}")
    print(f"  Slack precision: budget={budget_precision}, "
          f"duration={duration_precision}, cardinality=1 "
          f"(matches solve_qubo_slack defaults)")
    print(f"\n  Qubit (variable) budget:")
    print(f"    Asset bits:            {vi['n_assets']:>3}   "
          f"indices {vi['asset_range']}")
    print(f"    Budget slack bits:     {vi['n_budget_slack']:>3}   "
          f"indices {vi['budget_slack_range']}")
    print(f"    Duration slack bits:   {vi['n_duration_slack']:>3}   "
          f"indices {vi['duration_slack_range']}")
    print(f"    Cardinality slack bits:{vi['n_cardinality_slack']:>3}   "
          f"indices {vi['cardinality_slack_range']}")
    print(f"    {'-' * 40}")
    print(f"    TOTAL qubits:          {nt:>3}   "
          f"({vi['n_assets']} asset + "
          f"{nt - vi['n_assets']} slack)")

    diag = np.diag(Q)
    off = np.triu(Q, k=1)
    off_nz = off[off != 0]
    print(f"\n  Matrix is {nt}x{nt}.")
    print(f"  Diagonal range:     [{diag.min():.4f}, {diag.max():.4f}]")
    if off_nz.size:
        print(f"  Off-diagonal range: [{off_nz.min():.4f}, "
              f"{off_nz.max():.4f}]  (nonzero couplings, upper triangle)")

    if nt <= max_full_matrix_n:
        print(f"\n  Full symmetric matrix Q ({nt}x{nt}):")
        for i in range(nt):
            row = "".join(f"{Q[i, j]:>9.2f}" for j in range(nt))
            print(f"    {i:>3}{row}")
    else:
        print(f"\n  (Full matrix suppressed for size > {max_full_matrix_n}; "
              f"showing statistics only.)")
    print()


def brute_force_solve(
    assets: List[Dict[str, Any]],
    budget: float,
    max_duration: float,
    max_cardinality: int
) -> Tuple[List[int], float]:
    """
    Brute force solver - enumerate all 2^n combinations.

    Returns the selection that maximizes score while satisfying ALL constraints.
    Only feasible for small n (< 20).
    """
    n = len(assets)
    best_selection = []
    best_score = -float('inf')

    prices = np.array([a['price'] for a in assets])
    durations = np.array([a['duration'] for a in assets])
    scores = np.array([a['score'] for a in assets])

    # Try all possible selections
    for r in range(max_cardinality + 1):
        for combo in combinations(range(n), r):
            if len(combo) == 0:
                continue

            indices = list(combo)
            total_price = np.sum(prices[indices])
            total_duration = np.sum(durations[indices])
            total_score = np.sum(scores[indices])

            # Check hard constraints (all are <= inequalities)
            budget_ok = total_price <= budget
            duration_ok = total_duration <= max_duration
            cardinality_ok = len(indices) <= max_cardinality

            if budget_ok and duration_ok and cardinality_ok:
                if total_score > best_score:
                    best_score = total_score
                    best_selection = indices

    return best_selection, best_score


def greedy_solve(
    assets: List[Dict[str, Any]],
    budget: float,
    max_duration: float,
    max_cardinality: int
) -> Tuple[List[int], float]:
    """
    Greedy solver - select assets by score/price ratio.

    Selects assets greedily until constraints are violated.
    """
    n = len(assets)

    prices = np.array([a['price'] for a in assets])
    durations = np.array([a['duration'] for a in assets])
    scores = np.array([a['score'] for a in assets])

    # Compute score-per-price ratio (higher is better)
    ratios = scores / (prices + 1e-10)

    # Sort by ratio descending
    sorted_indices = np.argsort(-ratios)

    selected = []
    total_price = 0.0
    total_duration = 0.0

    for idx in sorted_indices:
        new_price = total_price + prices[idx]
        new_duration = total_duration + durations[idx]
        new_count = len(selected) + 1

        # Check if adding this asset would violate constraints
        if new_count > max_cardinality:
            continue
        if new_duration > max_duration:
            continue
        if new_price > budget:
            continue

        selected.append(idx)
        total_price = new_price
        total_duration = new_duration

    total_score = np.sum(scores[selected]) if selected else 0.0
    return selected, total_score


def random_solve(
    assets: List[Dict[str, Any]],
    budget: float,
    max_duration: float,
    max_cardinality: int,
    num_samples: int = 1000,
    seed: Optional[int] = None
) -> Tuple[List[int], float]:
    """
    Random sampling solver - try random selections and keep best feasible.
    """
    rng = np.random.default_rng(seed)
    n = len(assets)

    prices = np.array([a['price'] for a in assets])
    durations = np.array([a['duration'] for a in assets])
    scores = np.array([a['score'] for a in assets])

    best_selection = []
    best_score = -float('inf')

    for _ in range(num_samples):
        # Random number of assets to select
        k = rng.integers(1, max_cardinality + 1)
        indices = list(rng.choice(n, size=min(k, n), replace=False))

        total_price = np.sum(prices[indices])
        total_duration = np.sum(durations[indices])
        total_score = np.sum(scores[indices])

        # Check constraints (all are <= inequalities)
        budget_ok = total_price <= budget
        duration_ok = total_duration <= max_duration
        cardinality_ok = len(indices) <= max_cardinality

        if budget_ok and duration_ok and cardinality_ok:
            if total_score > best_score:
                best_score = total_score
                best_selection = indices

    return best_selection, best_score if best_score > -float('inf') else 0.0


def scipy_ilp_solve(
    assets: List[Dict[str, Any]],
    budget: float,
    max_duration: float,
    max_cardinality: int
) -> Tuple[List[int], float]:
    """
    Integer Linear Programming solver using scipy.optimize.milp.

    Maximizes score subject to:
    - sum(price_i * x_i) <= budget
    - sum(duration_i * x_i) <= max_duration
    - sum(x_i) <= max_cardinality
    - x_i in {0, 1}
    """
    try:
        from scipy.optimize import milp, Bounds, LinearConstraint
    except ImportError:
        return [], 0.0

    n = len(assets)

    prices = np.array([a['price'] for a in assets])
    durations = np.array([a['duration'] for a in assets])
    scores = np.array([a['score'] for a in assets])

    # Objective: minimize -score (to maximize score)
    c = -scores

    # Constraints matrix (all <= inequalities)
    A = np.vstack([
        prices,           # Budget: sum(p_i * x_i) <= budget
        durations,        # Duration: sum(d_i * x_i) <= max_duration
        np.ones(n)        # Cardinality: sum(x_i) <= max_cardinality
    ])

    b_upper = np.array([budget, max_duration, max_cardinality])
    b_lower = np.array([-np.inf, -np.inf, -np.inf])

    constraints = LinearConstraint(A, b_lower, b_upper)

    # Variable bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

    # Integer constraints
    integrality = np.ones(n)

    try:
        result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)

        if result.success:
            selection = np.round(result.x).astype(int)
            selected = list(np.where(selection == 1)[0])
            total_score = np.sum(scores[selected]) if selected else 0.0
            return selected, total_score
    except Exception:
        pass

    return [], 0.0


def qubo_solve_filtered(
    optimizer: SimplePortfolioQUBO,
    num_reads: int = 1000,
    seed: Optional[int] = None
) -> Tuple[List[int], float, float]:
    """
    QUBO solver with post-filtering to only accept feasible solutions.

    Returns the best feasible solution from the sample set.
    """
    result = optimizer.solve(num_reads=num_reads, seed=seed)
    sampleset = result['sampleset']

    best_selection = []
    best_score = -float('inf')
    best_energy = 0.0

    for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy']):
        selection = np.array([sample[a.id] for a in optimizer.assets])
        check = optimizer.check_constraints(selection)

        if (check['cardinality_satisfied'] and
            check['duration_satisfied'] and
            check['budget_satisfied']):
            score = check['num_selected']  # Count selected
            indices = list(np.where(selection == 1)[0])
            total_score = np.sum(optimizer.scores[indices]) if indices else 0.0

            if total_score > best_score:
                best_score = total_score
                best_selection = indices
                best_energy = energy

    return best_selection, best_score if best_score > -float('inf') else 0.0, best_energy


def slack_qubo_solve_filtered(
    optimizer: SlackPortfolioQUBO,
    num_reads: int = 1000,
    seed: Optional[int] = None
) -> Tuple[List[int], float, float]:
    """
    Slack-variable QUBO solver with post-filtering for feasibility.

    Returns the best feasible solution from the sample set.
    """
    result = optimizer.solve(num_reads=num_reads, seed=seed)
    sampleset = result['sampleset']

    best_selection = []
    best_score = -float('inf')
    best_energy = 0.0

    for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy']):
        selection = np.array([sample[a.id] for a in optimizer.assets])
        check = optimizer.check_constraints(selection)

        if (check['cardinality_satisfied'] and
            check['duration_satisfied'] and
            check['budget_satisfied']):
            indices = list(np.where(selection == 1)[0])
            total_score = np.sum(optimizer.scores[indices]) if indices else 0.0

            if total_score > best_score:
                best_score = total_score
                best_selection = indices
                best_energy = energy

    return best_selection, best_score if best_score > -float('inf') else 0.0, best_energy


def run_benchmark(
    assets: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    qubo_num_reads: int = 1000,
    random_samples: int = 1000,
    seed: Optional[int] = None
) -> List[BenchmarkResult]:
    """
    Run all solvers on a single problem instance.
    """
    results = []

    budget = constraints['budget']
    max_duration = constraints['max_duration']
    max_cardinality = constraints['max_cardinality']

    prices = np.array([a['price'] for a in assets])
    durations = np.array([a['duration'] for a in assets])
    scores = np.array([a['score'] for a in assets])
    asset_ids = [a['id'] for a in assets]
    n = len(assets)

    def make_result(name: str, selected_indices: List[int], runtime_ms: float, energy: float = 0.0) -> BenchmarkResult:
        if not selected_indices:
            return BenchmarkResult(
                solver_name=name,
                selected_assets=[],
                total_score=0.0,
                total_price=0.0,
                total_duration=0.0,
                num_selected=0,
                energy=energy,
                runtime_ms=runtime_ms,
                budget_satisfied=False,
                duration_satisfied=True,
                cardinality_satisfied=True,
                is_feasible=False
            )

        total_price = float(np.sum(prices[selected_indices]))
        total_duration = float(np.sum(durations[selected_indices]))
        total_score = float(np.sum(scores[selected_indices]))

        # All constraints are <= inequalities
        budget_ok = total_price <= budget
        duration_ok = total_duration <= max_duration
        cardinality_ok = len(selected_indices) <= max_cardinality

        return BenchmarkResult(
            solver_name=name,
            selected_assets=[asset_ids[i] for i in selected_indices],
            total_score=total_score,
            total_price=total_price,
            total_duration=total_duration,
            num_selected=len(selected_indices),
            energy=energy,
            runtime_ms=runtime_ms,
            budget_satisfied=budget_ok,
            duration_satisfied=duration_ok,
            cardinality_satisfied=cardinality_ok,
            is_feasible=budget_ok and duration_ok and cardinality_ok
        )

    # 1. QUBO (Simulated Annealing) - raw best solution
    start = time.perf_counter()
    optimizer = SimplePortfolioQUBO(assets=assets, **constraints)
    qubo_result = optimizer.solve(num_reads=qubo_num_reads, seed=seed)
    qubo_time = (time.perf_counter() - start) * 1000

    selected_indices = [i for i, x in enumerate(qubo_result['selection']) if x == 1]
    results.append(make_result("QUBO (SA)", selected_indices, qubo_time, qubo_result['energy']))

    # 1a-qpu. QUBO (QPU) - same BQM as QUBO (SA), on real D-Wave hardware
    # via DWaveCliqueSampler. Only attempted when DWAVE_API_TOKEN is set
    # (via .env or the shell) so routine/multi-trial runs don't submit to
    # the QPU and consume Leap budget unless credentials are configured.
    if os.environ.get('DWAVE_API_TOKEN'):
        start = time.perf_counter()
        try:
            qpu_optimizer = SimplePortfolioQUBO(assets=assets, **constraints)
            qpu_result = qpu_optimizer.solve_qpu(num_reads=qubo_num_reads)
            qpu_time = (time.perf_counter() - start) * 1000
            qpu_selection = [i for i, x in enumerate(qpu_result['selection']) if x == 1]
            results.append(make_result(
                "QUBO (QPU)", qpu_selection, qpu_time, qpu_result['energy']
            ))
        except Exception:
            qpu_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="QUBO (QPU)", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=qpu_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="QUBO (QPU)", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 1b. QUBO (Filtered) - best feasible solution from sample set
    start = time.perf_counter()
    optimizer2 = SimplePortfolioQUBO(assets=assets, **constraints)
    filtered_selection, filtered_score, filtered_energy = qubo_solve_filtered(
        optimizer2, num_reads=qubo_num_reads, seed=seed
    )
    filtered_time = (time.perf_counter() - start) * 1000
    results.append(make_result("QUBO (Filtered)", filtered_selection, filtered_time, filtered_energy))

    # 1c. QUBO (High Penalty) - use higher lambda values for harder constraints
    start = time.perf_counter()
    high_penalty_constraints = constraints.copy()
    high_penalty_constraints['lambda_budget'] = constraints.get('lambda_budget', 2.0) * 50
    high_penalty_constraints['lambda_duration'] = constraints.get('lambda_duration', 10.0) * 50
    high_penalty_constraints['lambda_cardinality'] = constraints.get('lambda_cardinality', 5.0) * 50
    optimizer3 = SimplePortfolioQUBO(assets=assets, **high_penalty_constraints)
    hp_selection, hp_score, hp_energy = qubo_solve_filtered(
        optimizer3, num_reads=qubo_num_reads, seed=seed
    )
    hp_time = (time.perf_counter() - start) * 1000
    results.append(make_result("QUBO (HighPen)", hp_selection, hp_time, hp_energy))

    # 1d. QUBO (Slack) - slack variable encoding for true inequality constraints
    start = time.perf_counter()
    slack_constraints = {
        'budget': budget,
        'max_duration': max_duration,
        'max_cardinality': max_cardinality,
        'lambda_budget': constraints.get('lambda_budget', 2.0),
        'lambda_duration': constraints.get('lambda_duration', 10.0),
        'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
    }
    slack_optimizer = SlackPortfolioQUBO(assets=assets, **slack_constraints)
    slack_result = slack_optimizer.solve(num_reads=qubo_num_reads, seed=seed)
    slack_time = (time.perf_counter() - start) * 1000
    slack_selection = [i for i, x in enumerate(slack_result['selection']) if x == 1]
    results.append(make_result("QUBO (Slack)", slack_selection, slack_time, slack_result['energy']))

    # 1d-qpu. QUBO (Slack QPU) - slack BQM on real D-Wave hardware via
    # DWaveCliqueSampler. Uses more qubits than the simple QUBO, so may
    # exceed the largest embeddable clique on larger problems. Only
    # attempted when DWAVE_API_TOKEN is set (see QUBO (QPU) above).
    if os.environ.get('DWAVE_API_TOKEN'):
        start = time.perf_counter()
        try:
            slack_qpu_optimizer = SlackPortfolioQUBO(assets=assets, **slack_constraints)
            slack_qpu_result = slack_qpu_optimizer.solve_qpu(num_reads=qubo_num_reads)
            slack_qpu_time = (time.perf_counter() - start) * 1000
            slack_qpu_selection = [
                i for i, x in enumerate(slack_qpu_result['selection']) if x == 1
            ]
            results.append(make_result(
                "QUBO (Slack QPU)", slack_qpu_selection, slack_qpu_time,
                slack_qpu_result['energy']
            ))
        except Exception:
            slack_qpu_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="QUBO (Slack QPU)", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=slack_qpu_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="QUBO (Slack QPU)", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 1e. QUBO (Slack+Filter) - best feasible from slack QUBO sample set
    start = time.perf_counter()
    slack_optimizer2 = SlackPortfolioQUBO(assets=assets, **slack_constraints)
    slack_filt_selection, slack_filt_score, slack_filt_energy = slack_qubo_solve_filtered(
        slack_optimizer2, num_reads=qubo_num_reads, seed=seed
    )
    slack_filt_time = (time.perf_counter() - start) * 1000
    results.append(make_result("QUBO (Slack+Filt)", slack_filt_selection, slack_filt_time, slack_filt_energy))

    # 2. Brute Force (only for small problems)
    if n <= 15:
        start = time.perf_counter()
        bf_selection, bf_score = brute_force_solve(assets, budget, max_duration, max_cardinality)
        bf_time = (time.perf_counter() - start) * 1000
        results.append(make_result("Brute Force", bf_selection, bf_time))
    else:
        results.append(BenchmarkResult(
            solver_name="Brute Force",
            selected_assets=[],
            total_score=0.0,
            total_price=0.0,
            total_duration=0.0,
            num_selected=0,
            energy=0.0,
            runtime_ms=0.0,
            budget_satisfied=False,
            duration_satisfied=False,
            cardinality_satisfied=False,
            is_feasible=False
        ))

    # 3. Greedy
    start = time.perf_counter()
    greedy_selection, greedy_score = greedy_solve(assets, budget, max_duration, max_cardinality)
    greedy_time = (time.perf_counter() - start) * 1000
    results.append(make_result("Greedy", greedy_selection, greedy_time))

    # 4. Random Sampling
    start = time.perf_counter()
    random_selection, random_score = random_solve(
        assets, budget, max_duration, max_cardinality,
        num_samples=random_samples, seed=seed
    )
    random_time = (time.perf_counter() - start) * 1000
    results.append(make_result("Random", random_selection, random_time))

    # 5. ILP (scipy)
    start = time.perf_counter()
    ilp_selection, ilp_score = scipy_ilp_solve(assets, budget, max_duration, max_cardinality)
    ilp_time = (time.perf_counter() - start) * 1000
    results.append(make_result("ILP (scipy)", ilp_selection, ilp_time))

    # 6. CQM (Exact) - ExactCQMSolver (small problems only)
    if n <= 20:
        start = time.perf_counter()
        cqm_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
        }
        cqm_optimizer = CQMPortfolioOptimizer(assets=assets, **cqm_constraints)
        cqm_result = cqm_optimizer.solve_exact()
        cqm_time = (time.perf_counter() - start) * 1000
        cqm_selection = [i for i, x in enumerate(cqm_result['selection']) if x == 1]
        results.append(make_result("CQM (Exact)", cqm_selection, cqm_time, cqm_result['energy']))
    else:
        results.append(BenchmarkResult(
            solver_name="CQM (Exact)",
            selected_assets=[],
            total_score=0.0,
            total_price=0.0,
            total_duration=0.0,
            num_selected=0,
            energy=0.0,
            runtime_ms=0.0,
            budget_satisfied=False,
            duration_satisfied=False,
            cardinality_satisfied=False,
            is_feasible=False
        ))

    # 7. CQM (SA) - CQM->BQM conversion + Simulated Annealing
    start = time.perf_counter()
    cqm_constraints = {
        'budget': budget,
        'max_duration': max_duration,
        'max_cardinality': max_cardinality,
    }
    cqm_sa_optimizer = CQMPortfolioOptimizer(assets=assets, **cqm_constraints)
    cqm_sa_result = cqm_sa_optimizer.solve_sa(num_reads=qubo_num_reads, seed=seed)
    cqm_sa_time = (time.perf_counter() - start) * 1000
    cqm_sa_selection = [i for i, x in enumerate(cqm_sa_result['selection']) if x == 1]
    # Only count as feasible if CQM reports it as feasible
    if cqm_sa_result['is_feasible']:
        results.append(make_result("CQM (SA)", cqm_sa_selection, cqm_sa_time, cqm_sa_result['energy']))
    else:
        # Report infeasible result
        total_price = float(np.sum(prices[cqm_sa_selection])) if cqm_sa_selection else 0.0
        total_duration = float(np.sum(durations[cqm_sa_selection])) if cqm_sa_selection else 0.0
        total_score = float(np.sum(scores[cqm_sa_selection])) if cqm_sa_selection else 0.0
        results.append(BenchmarkResult(
            solver_name="CQM (SA)",
            selected_assets=[asset_ids[i] for i in cqm_sa_selection],
            total_score=total_score,
            total_price=total_price,
            total_duration=total_duration,
            num_selected=len(cqm_sa_selection),
            energy=cqm_sa_result['energy'],
            runtime_ms=cqm_sa_time,
            budget_satisfied=total_price <= budget,
            duration_satisfied=total_duration <= max_duration,
            cardinality_satisfied=len(cqm_sa_selection) <= max_cardinality,
            is_feasible=False
        ))

    # 8. NL (Exact) - Combination enumeration via dwave-optimization model
    start = time.perf_counter()
    nl_constraints = {
        'budget': budget,
        'max_duration': max_duration,
        'max_cardinality': max_cardinality,
    }
    nl_optimizer = NLPortfolioOptimizer(assets=assets, **nl_constraints)
    try:
        nl_result = nl_optimizer.solve_exact()
        nl_time = (time.perf_counter() - start) * 1000
        nl_selection = [i for i, x in enumerate(nl_result['selection']) if x == 1]
        results.append(make_result("NL (Exact)", nl_selection, nl_time, nl_result['energy']))
    except ValueError:
        # Too many states to enumerate
        nl_time = (time.perf_counter() - start) * 1000
        results.append(BenchmarkResult(
            solver_name="NL (Exact)",
            selected_assets=[],
            total_score=0.0,
            total_price=0.0,
            total_duration=0.0,
            num_selected=0,
            energy=0.0,
            runtime_ms=nl_time,
            budget_satisfied=False,
            duration_satisfied=False,
            cardinality_satisfied=False,
            is_feasible=False
        ))

    # 9. QHD-QP (Classical) — QHDOPT with QUBO matrix via QP interface
    if _HAS_QHDOPT:
        start = time.perf_counter()
        qhd_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
            'lambda_budget': constraints.get('lambda_budget', 2.0),
            'lambda_duration': constraints.get('lambda_duration', 10.0),
            'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
        }
        qhd_optimizer = QHDPortfolioOptimizer(assets=assets, **qhd_constraints)
        try:
            qhd_qp_result = qhd_optimizer.solve_qp(num_shots=50)
            qhd_qp_time = (time.perf_counter() - start) * 1000
            qhd_qp_selection = [i for i, x in enumerate(qhd_qp_result['selection']) if x == 1]
            results.append(make_result(
                "QHD-QP", qhd_qp_selection, qhd_qp_time, qhd_qp_result.get('energy', 0.0)
            ))
        except Exception:
            qhd_qp_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="QHD-QP", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=qhd_qp_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="QHD-QP", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 10. QHD-SymPy (Classical) — QHDOPT with native formulation + binary penalty
    if _HAS_QHDOPT:
        start = time.perf_counter()
        qhd_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
            'lambda_budget': constraints.get('lambda_budget', 2.0),
            'lambda_duration': constraints.get('lambda_duration', 10.0),
            'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
        }
        qhd_sp_optimizer = QHDPortfolioOptimizer(assets=assets, **qhd_constraints)
        try:
            qhd_sp_result = qhd_sp_optimizer.solve_sympy(num_shots=50)
            qhd_sp_time = (time.perf_counter() - start) * 1000
            qhd_sp_selection = [i for i, x in enumerate(qhd_sp_result['selection']) if x == 1]
            results.append(make_result(
                "QHD-SymPy", qhd_sp_selection, qhd_sp_time, qhd_sp_result.get('energy', 0.0)
            ))
        except Exception:
            qhd_sp_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="QHD-SymPy", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=qhd_sp_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="QHD-SymPy", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 11. cuOpt-MILP — GPU-accelerated binary integer programming
    if _HAS_CUOPT:
        start = time.perf_counter()
        cuopt_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
            'lambda_budget': constraints.get('lambda_budget', 2.0),
            'lambda_duration': constraints.get('lambda_duration', 10.0),
            'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
        }
        cuopt_optimizer = CuOptPortfolioOptimizer(assets=assets, **cuopt_constraints)
        try:
            cuopt_milp_result = cuopt_optimizer.solve_milp()
            cuopt_milp_time = (time.perf_counter() - start) * 1000
            cuopt_milp_selection = [
                i for i, x in enumerate(cuopt_milp_result['selection']) if x == 1
            ]
            results.append(make_result(
                "cuOpt-MILP", cuopt_milp_selection, cuopt_milp_time,
                cuopt_milp_result.get('energy', 0.0)
            ))
        except Exception:
            cuopt_milp_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="cuOpt-MILP", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=cuopt_milp_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="cuOpt-MILP", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 12. cuOpt-QP — GPU QP solver with QUBO penalty matrix
    if _HAS_CUOPT:
        start = time.perf_counter()
        cuopt_qp_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
            'lambda_budget': constraints.get('lambda_budget', 2.0),
            'lambda_duration': constraints.get('lambda_duration', 10.0),
            'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
        }
        cuopt_qp_optimizer = CuOptPortfolioOptimizer(assets=assets, **cuopt_qp_constraints)
        try:
            cuopt_qp_result = cuopt_qp_optimizer.solve_qp()
            cuopt_qp_time = (time.perf_counter() - start) * 1000
            cuopt_qp_selection = [
                i for i, x in enumerate(cuopt_qp_result['selection']) if x == 1
            ]
            results.append(make_result(
                "cuOpt-QP", cuopt_qp_selection, cuopt_qp_time,
                cuopt_qp_result.get('energy', 0.0)
            ))
        except Exception:
            cuopt_qp_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="cuOpt-QP", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=cuopt_qp_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="cuOpt-QP", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 13. Phi-QUBO — PhiSolve QIHD with QUBO penalty matrix
    if _HAS_PHISOLVE:
        start = time.perf_counter()
        phi_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
            'lambda_budget': constraints.get('lambda_budget', 2.0),
            'lambda_duration': constraints.get('lambda_duration', 10.0),
            'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
        }
        phi_optimizer = PhiPortfolioOptimizer(assets=assets, **phi_constraints)
        try:
            # n_steps must be > 1000: OpenPhiSolve 0.2.0 bug (qihd.py:152)
            phi_qubo_result = phi_optimizer.solve_qubo(n_shots=100, n_steps=10000)
            phi_qubo_time = (time.perf_counter() - start) * 1000
            phi_qubo_selection = [
                i for i, x in enumerate(phi_qubo_result['selection']) if x == 1
            ]
            results.append(make_result(
                "Phi-QUBO", phi_qubo_selection, phi_qubo_time,
                phi_qubo_result.get('energy', 0.0)
            ))
        except Exception:
            phi_qubo_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="Phi-QUBO", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=phi_qubo_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="Phi-QUBO", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 14. Phi-MIQP — PhiSolve QIHD with native linear constraints
    if _HAS_PHISOLVE:
        start = time.perf_counter()
        phi_miqp_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
            'lambda_budget': constraints.get('lambda_budget', 2.0),
            'lambda_duration': constraints.get('lambda_duration', 10.0),
            'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
        }
        phi_miqp_optimizer = PhiPortfolioOptimizer(assets=assets, **phi_miqp_constraints)
        try:
            # n_steps must be > 1000: OpenPhiSolve 0.2.0 bug (qihd.py:152)
            phi_miqp_result = phi_miqp_optimizer.solve_miqp(n_shots=100, n_steps=10000)
            phi_miqp_time = (time.perf_counter() - start) * 1000
            phi_miqp_selection = [
                i for i, x in enumerate(phi_miqp_result['selection']) if x == 1
            ]
            results.append(make_result(
                "Phi-MIQP", phi_miqp_selection, phi_miqp_time,
                phi_miqp_result.get('energy', 0.0)
            ))
        except Exception:
            phi_miqp_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="Phi-MIQP", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=phi_miqp_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="Phi-MIQP", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 15. Pasqal-QUBO — qubosolver LocalEmulator on the QUBO penalty matrix
    if _HAS_PASQAL:
        start = time.perf_counter()
        pasqal_constraints = {
            'budget': budget,
            'max_duration': max_duration,
            'max_cardinality': max_cardinality,
            'lambda_budget': constraints.get('lambda_budget', 2.0),
            'lambda_duration': constraints.get('lambda_duration', 10.0),
            'lambda_cardinality': constraints.get('lambda_cardinality', 5.0),
        }
        pasqal_optimizer = PasqalPortfolioOptimizer(
            assets=assets, **pasqal_constraints
        )
        try:
            pasqal_qubo_result = pasqal_optimizer.solve_qubo()
            pasqal_qubo_time = (time.perf_counter() - start) * 1000
            pasqal_qubo_selection = [
                i for i, x in enumerate(pasqal_qubo_result['selection']) if x == 1
            ]
            results.append(make_result(
                "Pasqal-QUBO", pasqal_qubo_selection, pasqal_qubo_time,
                pasqal_qubo_result.get('energy', 0.0)
            ))
        except Exception:
            pasqal_qubo_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="Pasqal-QUBO", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=pasqal_qubo_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="Pasqal-QUBO", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 16. Pasqal-Slack — qubosolver on SlackPortfolioQUBO (slack-variable matrix)
    if _HAS_PASQAL:
        start = time.perf_counter()
        pasqal_optimizer = PasqalPortfolioOptimizer(
            assets=assets, **pasqal_constraints
        )
        try:
            pasqal_slack_result = pasqal_optimizer.solve_qubo_slack()
            pasqal_slack_time = (time.perf_counter() - start) * 1000
            pasqal_slack_selection = [
                i for i, x in enumerate(pasqal_slack_result['selection']) if x == 1
            ]
            results.append(make_result(
                "Pasqal-Slack", pasqal_slack_selection, pasqal_slack_time,
                pasqal_slack_result.get('energy', 0.0)
            ))
        except Exception:
            pasqal_slack_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="Pasqal-Slack", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=pasqal_slack_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="Pasqal-Slack", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 17. Pasqal-MIS — Max Independent Set on pairwise conflict graph (NOT comparable)
    if _HAS_PASQAL:
        start = time.perf_counter()
        pasqal_optimizer = PasqalPortfolioOptimizer(
            assets=assets, **pasqal_constraints
        )
        try:
            pasqal_mis_result = pasqal_optimizer.solve_mis()
            pasqal_mis_time = (time.perf_counter() - start) * 1000
            pasqal_mis_selection = [
                i for i, x in enumerate(pasqal_mis_result['selection']) if x == 1
            ]
            results.append(make_result(
                "Pasqal-MIS", pasqal_mis_selection, pasqal_mis_time,
                pasqal_mis_result.get('energy', 0.0)
            ))
        except Exception:
            pasqal_mis_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="Pasqal-MIS", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=pasqal_mis_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="Pasqal-MIS", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    # 18. Pasqal-Pulser — raw Rydberg adiabatic sequence (NOT comparable, n<=12)
    if _HAS_PASQAL:
        start = time.perf_counter()
        pasqal_optimizer = PasqalPortfolioOptimizer(
            assets=assets, **pasqal_constraints
        )
        try:
            pasqal_pulser_result = pasqal_optimizer.solve_pulser(n_shots=50)
            pasqal_pulser_time = (time.perf_counter() - start) * 1000
            pasqal_pulser_selection = [
                i for i, x in enumerate(pasqal_pulser_result['selection']) if x == 1
            ]
            results.append(make_result(
                "Pasqal-Pulser", pasqal_pulser_selection, pasqal_pulser_time,
                pasqal_pulser_result.get('energy', 0.0)
            ))
        except Exception:
            pasqal_pulser_time = (time.perf_counter() - start) * 1000
            results.append(BenchmarkResult(
                solver_name="Pasqal-Pulser", selected_assets=[], total_score=0.0,
                total_price=0.0, total_duration=0.0, num_selected=0,
                energy=0.0, runtime_ms=pasqal_pulser_time,
                budget_satisfied=False, duration_satisfied=False,
                cardinality_satisfied=False, is_feasible=False
            ))
    else:
        results.append(BenchmarkResult(
            solver_name="Pasqal-Pulser", selected_assets=[], total_score=0.0,
            total_price=0.0, total_duration=0.0, num_selected=0,
            energy=0.0, runtime_ms=0.0,
            budget_satisfied=False, duration_satisfied=False,
            cardinality_satisfied=False, is_feasible=False
        ))

    return results


def generate_random_problem(
    num_assets: int,
    seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate a random problem instance."""
    rng = np.random.default_rng(seed)

    assets = []
    for i in range(num_assets):
        assets.append({
            'id': f'A{i:02d}',
            'price': round(rng.uniform(0.5, 2.0), 2),
            'duration': int(rng.integers(1, 15)),
            'score': int(rng.integers(1, 20))
        })

    # Set constraints based on generated assets
    total_price = sum(a['price'] for a in assets)
    total_duration = sum(a['duration'] for a in assets)

    constraints = {
        'budget': round(total_price * 0.4, 2),  # Target ~40% of total
        'max_duration': int(total_duration * 0.5),  # Allow ~50% of total
        'max_cardinality': max(2, num_assets // 3),  # Select up to 1/3
        'lambda_budget': 2.0,
        'lambda_duration': 10.0,
        'lambda_cardinality': 5.0
    }

    return assets, constraints


def print_results_table(results: List[BenchmarkResult], title: str = "Benchmark Results"):
    """Print results in a formatted table."""
    print(f"\n{'=' * 110}")
    print(f" {title}")
    print(f"{'=' * 110}")

    # Find best feasible score for gap calculation
    feasible = [r for r in results if r.is_feasible]
    best_score = max(r.total_score for r in feasible) if feasible else 0.0

    # Header
    header = f"{'Solver':<15} {'Score':>8} {'Gap%':>7} {'Price':>8} {'Dur':>6} {'#':>3} {'Budget':>8} {'Dur':>6} {'Card':>6} {'Feas':>6} {'Time(ms)':>10}"
    print(header)
    print("-" * 110)

    for r in results:
        budget_str = "OK" if r.budget_satisfied else "FAIL"
        dur_str = "OK" if r.duration_satisfied else "FAIL"
        card_str = "OK" if r.cardinality_satisfied else "FAIL"
        feas_str = "YES" if r.is_feasible else "NO"

        # Calculate optimality gap
        if r.is_feasible and best_score > 0:
            gap = 100.0 * (best_score - r.total_score) / best_score
            gap_str = f"{gap:>6.1f}%"
        else:
            gap_str = "    N/A"

        row = (
            f"{r.solver_name:<15} "
            f"{r.total_score:>8.1f} "
            f"{gap_str} "
            f"{r.total_price:>8.2f} "
            f"{r.total_duration:>6.0f} "
            f"{r.num_selected:>3d} "
            f"{budget_str:>8} "
            f"{dur_str:>6} "
            f"{card_str:>6} "
            f"{feas_str:>6} "
            f"{r.runtime_ms:>10.2f}"
        )
        print(row)

    print("-" * 110)

    # Find best feasible solution
    if feasible:
        best = max(feasible, key=lambda r: r.total_score)
        print(f"Best feasible solution: {best.solver_name} with score {best.total_score:.1f}")
    else:
        print("No feasible solution found by any solver!")


def print_summary_stats(all_results: List[List[BenchmarkResult]], solver_names: List[str]):
    """Print summary statistics across multiple trials."""
    print(f"\n{'=' * 115}")
    print(" Summary Statistics (across all trials)")
    print(f"{'=' * 115}")

    # Aggregate by solver
    stats = {name: {'scores': [], 'times': [], 'gaps': [], 'feasible_count': 0, 'total': 0} for name in solver_names}

    # Calculate gaps relative to best feasible in each trial
    for trial_results in all_results:
        feasible = [r for r in trial_results if r.is_feasible]
        best_score = max(r.total_score for r in feasible) if feasible else 0.0

        for r in trial_results:
            stats[r.solver_name]['total'] += 1
            stats[r.solver_name]['times'].append(r.runtime_ms)
            if r.is_feasible:
                stats[r.solver_name]['scores'].append(r.total_score)
                stats[r.solver_name]['feasible_count'] += 1
                if best_score > 0:
                    gap = 100.0 * (best_score - r.total_score) / best_score
                    stats[r.solver_name]['gaps'].append(gap)

    header = f"{'Solver':<15} {'Avg Score':>10} {'Avg Gap%':>10} {'Feas Rate':>10} {'Avg Time':>12} {'Std Time':>10}"
    print(header)
    print("-" * 115)

    for name in solver_names:
        s = stats[name]
        if s['scores']:
            avg_score = np.mean(s['scores'])
        else:
            avg_score = 0.0

        if s['gaps']:
            avg_gap = np.mean(s['gaps'])
            gap_str = f"{avg_gap:>9.1f}%"
        else:
            gap_str = "      N/A"

        feas_rate = s['feasible_count'] / s['total'] if s['total'] > 0 else 0.0
        avg_time = np.mean(s['times']) if s['times'] else 0.0
        std_time = np.std(s['times']) if s['times'] else 0.0

        row = (
            f"{name:<15} "
            f"{avg_score:>10.2f} "
            f"{gap_str} "
            f"{feas_rate:>10.1%} "
            f"{avg_time:>12.2f} "
            f"{std_time:>10.2f}"
        )
        print(row)

    print("-" * 115)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SimplePortfolioQUBO against classical optimizers"
    )
    parser.add_argument(
        "--problem-set",
        choices=["simple", "random"],
        default="simple",
        help="Problem set to use (default: simple)"
    )
    parser.add_argument(
        "--num-assets",
        type=int,
        default=5,
        help="Number of assets for random problems (default: 5)"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials to run (default: 1)"
    )
    parser.add_argument(
        "--qubo-reads",
        type=int,
        default=1000,
        help="Number of QUBO annealing reads (default: 1000)"
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=1000,
        help="Number of random samples (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-qubits",
        action="store_true",
        help="Skip qubit requirements table"
    )
    parser.add_argument(
        "--show-qubo",
        action="store_true",
        help="Print the QUBO matrix and Ising/compression stats before "
             "solving (the same Q consumed by SA, cuOpt, Phi, and Pasqal)"
    )
    parser.add_argument(
        "--show-slack-qubo",
        action="store_true",
        help="Print the slack-variable QUBO matrix and its qubit budget "
             "(the matrix consumed by the Pasqal-Slack path)"
    )

    args = parser.parse_args()

    print("=" * 100)
    print(" SimplePortfolioQUBO Benchmark")
    print("=" * 100)
    print(f"Problem set: {args.problem_set}")
    print(f"Number of trials: {args.num_trials}")
    print(f"QUBO reads: {args.qubo_reads}")
    print(f"Random samples: {args.random_samples}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    all_results = []
    solver_names = ["QUBO (SA)", "QUBO (QPU)", "QUBO (Filtered)", "QUBO (HighPen)", "QUBO (Slack)", "QUBO (Slack QPU)", "QUBO (Slack+Filt)", "Brute Force", "Greedy", "Random", "ILP (scipy)", "CQM (Exact)", "CQM (SA)", "NL (Exact)", "QHD-QP", "QHD-SymPy", "cuOpt-MILP", "cuOpt-QP", "Phi-QUBO", "Phi-MIQP", "Pasqal-QUBO", "Pasqal-Slack", "Pasqal-MIS", "Pasqal-Pulser"]

    for trial in range(args.num_trials):
        if args.problem_set == "simple":
            assets = get_example_assets()
            constraints = get_example_constraints()
            title = f"Simple Example Problem (Trial {trial + 1}/{args.num_trials})"
        else:
            trial_seed = args.seed + trial if args.seed is not None else None
            assets, constraints = generate_random_problem(args.num_assets, seed=trial_seed)
            title = f"Random Problem (n={args.num_assets}, Trial {trial + 1}/{args.num_trials})"

        # Print problem info and qubit table for first trial
        if trial == 0:
            print(f"\nProblem Configuration:")
            print(f"  Assets: {len(assets)}")
            print(f"  Budget target: ${constraints['budget']:.2f}")
            print(f"  Max duration: {constraints['max_duration']}")
            print(f"  Max cardinality: {constraints['max_cardinality']}")

            if not args.no_qubits:
                qubit_info = compute_qubit_requirements(assets, constraints)
                print_qubit_table(qubit_info, len(assets))

            if args.show_qubo:
                print_qubo_info(assets, constraints)

            if args.show_slack_qubo:
                print_slack_qubo_info(assets, constraints)

        trial_seed = args.seed + trial if args.seed is not None else None
        results = run_benchmark(
            assets=assets,
            constraints=constraints,
            qubo_num_reads=args.qubo_reads,
            random_samples=args.random_samples,
            seed=trial_seed
        )

        all_results.append(results)
        print_results_table(results, title)

    # Print summary if multiple trials
    if args.num_trials > 1:
        print_summary_stats(all_results, solver_names)


if __name__ == "__main__":
    main()
