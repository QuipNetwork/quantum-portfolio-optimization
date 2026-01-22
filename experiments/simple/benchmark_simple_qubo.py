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
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations

from simple_portfolio_qubo import SimplePortfolioQUBO, get_example_assets, get_example_constraints
from slack_portfolio_qubo import SlackPortfolioQUBO
from cqm_portfolio import CQMPortfolioOptimizer


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
    solver_names = ["QUBO (SA)", "QUBO (Filtered)", "QUBO (HighPen)", "QUBO (Slack)", "QUBO (Slack+Filt)", "Brute Force", "Greedy", "Random", "ILP (scipy)", "CQM (Exact)", "CQM (SA)"]

    for trial in range(args.num_trials):
        if args.problem_set == "simple":
            assets = get_example_assets()
            constraints = get_example_constraints()
            title = f"Simple Example Problem (Trial {trial + 1}/{args.num_trials})"
        else:
            trial_seed = args.seed + trial if args.seed is not None else None
            assets, constraints = generate_random_problem(args.num_assets, seed=trial_seed)
            title = f"Random Problem (n={args.num_assets}, Trial {trial + 1}/{args.num_trials})"

        # Print problem info for first trial
        if trial == 0:
            print(f"\nProblem Configuration:")
            print(f"  Assets: {len(assets)}")
            print(f"  Budget target: ${constraints['budget']:.2f}")
            print(f"  Max duration: {constraints['max_duration']}")
            print(f"  Max cardinality: {constraints['max_cardinality']}")

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
