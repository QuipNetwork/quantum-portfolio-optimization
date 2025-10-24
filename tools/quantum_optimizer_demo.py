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
Discrete Levels Portfolio Optimizer Demo

Demonstrates the Discrete Levels Approach for portfolio optimization using quantum annealing.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer


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

    weights = result['weights']
    metrics = result['metrics']

    print(f"\nStatus: ✓ SUCCESS")
    print(f"Runtime: {metrics['runtime']:.3f}s")

    print_header("PORTFOLIO METRICS")
    print(f"Expected Annual Return: {metrics['return']*100:>8.2f}%")
    print(f"Annual Volatility:      {metrics['risk']*100:>8.2f}%")
    print(f"Sharpe Ratio:           {metrics['sharpe']:>8.3f}")

    print_header("CLUSTERING STATISTICS")
    print(f"Number of clusters:     {metrics['n_clusters']:>8d}")

    print_header("PORTFOLIO WEIGHTS")
    weights_filtered = weights[weights > 0].sort_values(ascending=False)
    print(f"\nNon-zero assets: {len(weights_filtered)} / {len(weights)}")
    print(f"Sparsity: {(weights == 0).sum() / len(weights) * 100:.1f}%\n")

    print(f"{'Ticker':<10} {'Weight':>10} {'Allocation':>12}")
    print("-" * 35)
    for ticker, weight in weights_filtered.items():
        print(f"{ticker:<10} {weight:>10.6f} {weight*100:>10.2f}%")

    print(f"\n{'Total':<10} {weights_filtered.sum():>10.6f} {weights_filtered.sum()*100:>10.2f}%")

    print_header("SOLVER INFORMATION")
    print(f"Runtime:                {metrics['runtime']:.3f}s")
    if 'qpu_access_time' in metrics:
        print(f"QPU access time:        {metrics['qpu_access_time']:.6f}s")
        print(f"Network latency:        {metrics['network_latency']:.3f}s")


def compare_parameters(returns: pd.DataFrame, config: dict):
    """Compare different parameter configurations."""
    print_header("PARAMETER COMPARISON")

    param_configs = [
        ('Balanced', {}),
        ('Aggressive', {'alpha': 20.0, 'beta': 1.0}),
        ('Conservative', {'alpha': 5.0, 'beta': 5.0}),
        ('High Spread', {'k_spread': 2.0})
    ]
    results = {}

    for name, params in param_configs:
        print(f"\nRunning {name} configuration...", end=" ")
        optimizer = DiscreteLevelsOptimizer(**{**config, **params})
        result = optimizer.optimize(returns)
        results[name] = result
        print(f"✓ ({result['metrics']['runtime']:.2f}s)")

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Config':<15} {'Return':>10} {'Risk':>10} {'Sharpe':>10} {'Non-zero':>10} {'Runtime':>10}")
    print("-" * 80)

    for name, result in results.items():
        metrics = result['metrics']
        weights = result['weights']
        print(f"{name:<15} "
              f"{metrics['return']*100:>9.2f}% "
              f"{metrics['risk']*100:>9.2f}% "
              f"{metrics['sharpe']:>10.3f} "
              f"{(weights > 0).sum():>10d} "
              f"{metrics['runtime']:>9.3f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Discrete Levels portfolio optimizer demo'
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
        default=19,
        help='Maximum assets per cluster (default: 19)'
    )

    parser.add_argument(
        '--n-levels',
        type=int,
        default=6,
        help='Number of discrete weight levels (default: 6)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=10.0,
        help='Return coefficient (higher = more aggressive, default: 10.0)'
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=2.0,
        help='Risk coefficient (higher = more conservative, default: 2.0)'
    )

    parser.add_argument(
        '--solver',
        type=str,
        default='simulated',
        choices=['simulated', 'qpu'],
        help='Solver type (default: simulated)'
    )

    parser.add_argument(
        '--num-reads',
        type=int,
        default=None,
        help='Number of samples (default: 256 for SA, 10 for QPU)'
    )

    parser.add_argument(
        '--k-spread',
        type=float,
        default=0.0,
        help='Concentration parameter (0.0=balanced, 1.0-3.0=concentrated, default: 0.0)'
    )

    parser.add_argument(
        '--compare-params',
        action='store_true',
        help='Compare different parameter configurations'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Save weights to CSV file'
    )

    args = parser.parse_args()

    # Print configuration
    print_header("DISCRETE LEVELS PORTFOLIO OPTIMIZER DEMO")
    print(f"\nPortfolio: {args.portfolio_csv}")
    print(f"Solver: {args.solver}")
    print(f"Max cluster size: {args.max_cluster_size}")
    print(f"Discretization: {args.n_levels} levels")
    print(f"Risk/Return: α={args.alpha}, β={args.beta}")
    print(f"Concentration: k={args.k_spread}")

    # Load data
    print("\nLoading portfolio data...", end=" ")
    returns = load_portfolio_data(args.portfolio_csv)
    print(f"✓ ({len(returns.columns)} assets, {len(returns)} days)")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    # Configuration
    config = {
        'max_cluster_size': args.max_cluster_size,
        'n_levels': args.n_levels,
        'alpha': args.alpha,
        'beta': args.beta,
        'solver_type': args.solver,
        'k_spread': args.k_spread,
        'l1_sparsity_penalty': 2.0,
        'use_thermometer_cutoff': True
    }

    if args.num_reads is not None:
        config['num_reads'] = args.num_reads

    if args.compare_params:
        # Compare parameter configurations
        results = compare_parameters(returns, config)
        best_config = max(results.items(), key=lambda x: x[1]['metrics']['sharpe'])
        print(f"\n✓ Best configuration: {best_config[0]} (Sharpe: {best_config[1]['metrics']['sharpe']:.3f})")

    else:
        # Single optimization
        print("\nInitializing optimizer...")
        optimizer = DiscreteLevelsOptimizer(**config)

        print("Running optimization...")
        result = optimizer.optimize(returns)

        print_result(result)

        # Save weights if requested
        if args.output:
            result['weights'].to_csv(args.output)
            print(f"\n✓ Weights saved to: {args.output}")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
