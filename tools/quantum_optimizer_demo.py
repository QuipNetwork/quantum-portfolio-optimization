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

#!/usr/bin/env python3
"""
Quantum Portfolio Optimizer Demo

Demonstrates the Independent Clusters Approach for portfolio optimization using quantum annealing.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qpo.optimizers.quantum import IndependentClustersOptimizer
from clustering import HierarchicalClusterer


def load_portfolio_data(csv_path: str) -> pd.DataFrame:
    """Load portfolio price data and compute returns."""
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    returns = data.pct_change().dropna()
    return returns


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_result(result):
    """Print optimization result."""
    print_header("OPTIMIZATION RESULT")

    print(f"\nStatus: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    if not result.success:
        print(f"Error: {result.message}")
        return

    print(f"Message: {result.message}")
    print(f"Runtime: {result.runtime:.3f}s")

    print_header("PORTFOLIO METRICS")
    print(f"Expected Annual Return: {result.expected_return*100:>8.2f}%")
    print(f"Annual Volatility:      {result.volatility*100:>8.2f}%")
    print(f"Sharpe Ratio:           {result.sharpe_ratio:>8.3f}")

    print_header("CLUSTERING STATISTICS")
    stats = result.cluster_stats
    print(f"Number of clusters:     {stats['n_clusters']:>8d}")
    print(f"Total assets:           {stats['total_assets']:>8d}")
    print(f"Cluster sizes:          {stats['min_cluster_size']:>8d} (min), "
          f"{stats['max_cluster_size']:>8d} (max), "
          f"{stats['avg_cluster_size']:>8.1f} (avg)")

    print_header("PORTFOLIO WEIGHTS")
    weights = result.weights[result.weights > 0].sort_values(ascending=False)
    print(f"\nNon-zero assets: {len(weights)} / {len(result.weights)}")
    print(f"Sparsity: {(result.weights == 0).sum() / len(result.weights) * 100:.1f}%\n")

    print(f"{'Ticker':<10} {'Weight':>10} {'Allocation':>12}")
    print("-" * 35)
    for ticker, weight in weights.items():
        print(f"{ticker:<10} {weight:>10.6f} {weight*100:>10.2f}%")

    print(f"\n{'Total':<10} {weights.sum():>10.6f} {weights.sum()*100:>10.2f}%")

    print_header("SOLVER INFORMATION")
    solver_info = result.solver_info
    print(f"Solver type:            {solver_info['solver_type']}")
    print(f"Number of reads:        {solver_info.get('num_reads', 'N/A')}")
    print(f"Clusters solved:        {solver_info['n_clusters_solved']}")
    print(f"Total solve time:       {solver_info['total_solve_time']:.3f}s")
    print(f"Avg solve time/cluster: {solver_info['avg_solve_time']:.3f}s")


def compare_strategies(returns: pd.DataFrame, config: dict):
    """Compare different aggregation strategies."""
    print_header("STRATEGY COMPARISON")

    strategies = ['concatenate', 'proportional', 'uniform']
    results = {}

    for strategy in strategies:
        print(f"\nRunning {strategy} strategy...", end=" ")
        optimizer = IndependentClustersOptimizer(**{**config, 'aggregation_strategy': strategy})
        result = optimizer.optimize(returns)
        results[strategy] = result
        print(f"✓ ({result.runtime:.2f}s)")

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<15} {'Return':>10} {'Risk':>10} {'Sharpe':>10} {'Non-zero':>10} {'Runtime':>10}")
    print("-" * 80)

    for strategy, result in results.items():
        if result.success:
            print(f"{strategy:<15} "
                  f"{result.expected_return*100:>9.2f}% "
                  f"{result.volatility*100:>9.2f}% "
                  f"{result.sharpe_ratio:>10.3f} "
                  f"{(result.weights > 0).sum():>10d} "
                  f"{result.runtime:>9.3f}s")
        else:
            print(f"{strategy:<15} {'FAILED':>10}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Quantum portfolio optimizer demo using Independent Clusters Approach'
    )

    parser.add_argument(
        '--portfolio-csv',
        type=str,
        required=True,
        help='Path to portfolio CSV file (prices)'
    )

    parser.add_argument(
        '--max-cluster-size',
        type=int,
        default=18,
        help='Maximum assets per cluster (default: 18)'
    )

    parser.add_argument(
        '--n-bits',
        type=int,
        default=10,
        help='Binary discretization bits (default: 10)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Return coefficient (higher = more aggressive, default: 1.0)'
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='Risk coefficient (higher = more conservative, default: 1.0)'
    )

    parser.add_argument(
        '--lambda-budget',
        type=float,
        default=10.0,
        help='Budget constraint penalty (default: 10.0)'
    )

    parser.add_argument(
        '--solver',
        type=str,
        default='simulated',
        choices=['simulated', 'qpu', 'hybrid'],
        help='Solver type (default: simulated)'
    )

    parser.add_argument(
        '--num-reads',
        type=int,
        default=1000,
        help='Number of QPU samples (default: 1000)'
    )

    parser.add_argument(
        '--annealing-time',
        type=int,
        default=20,
        help='Annealing time in microseconds (default: 20)'
    )

    parser.add_argument(
        '--aggregation',
        type=str,
        default='concatenate',
        choices=['concatenate', 'proportional', 'uniform'],
        help='Aggregation strategy (default: concatenate)'
    )

    parser.add_argument(
        '--cardinality',
        type=int,
        default=None,
        help='Maximum non-zero assets (optional)'
    )

    parser.add_argument(
        '--compare-strategies',
        action='store_true',
        help='Compare all aggregation strategies'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Save weights to CSV file'
    )

    args = parser.parse_args()

    # Print configuration
    print_header("QUANTUM PORTFOLIO OPTIMIZER DEMO")
    print(f"\nPortfolio: {args.portfolio_csv}")
    print(f"Solver: {args.solver}")
    print(f"Max cluster size: {args.max_cluster_size}")
    print(f"Discretization: {args.n_bits} bits ({2**args.n_bits} levels)")
    print(f"Risk/Return: α={args.alpha}, β={args.beta}")
    print(f"Aggregation: {args.aggregation}")
    if args.cardinality:
        print(f"Cardinality constraint: {args.cardinality} assets")

    # Load data
    print("\nLoading portfolio data...", end=" ")
    returns = load_portfolio_data(args.portfolio_csv)
    print(f"✓ ({len(returns.columns)} assets, {len(returns)} days)")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    # Configuration
    config = {
        'max_cluster_size': args.max_cluster_size,
        'n_bits': args.n_bits,
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_budget': args.lambda_budget,
        'solver_type': args.solver,
        'num_reads': args.num_reads,
        'annealing_time': args.annealing_time,
        'aggregation_strategy': args.aggregation,
        'cardinality': args.cardinality
    }

    if args.compare_strategies:
        # Compare all strategies
        results = compare_strategies(returns, config)
        best_strategy = max(results.items(), key=lambda x: x[1].sharpe_ratio if x[1].success else -np.inf)
        print(f"\n✓ Best strategy: {best_strategy[0]} (Sharpe: {best_strategy[1].sharpe_ratio:.3f})")

    else:
        # Single optimization
        print("\nInitializing optimizer...")
        optimizer = IndependentClustersOptimizer(**config)

        print("Running optimization...")
        result = optimizer.optimize(returns)

        print_result(result)

        # Save weights if requested
        if args.output and result.success:
            result.weights.to_csv(args.output)
            print(f"\n✓ Weights saved to: {args.output}")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
