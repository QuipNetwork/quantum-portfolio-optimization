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
Test script for two-pass hierarchical quantum optimization.

Compares:
1. Original single-pass (equal cluster weighting)
2. Two-pass hierarchical (QUBO-optimized cluster allocation)
3. Classical Mean-Variance (global benchmark)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer, DiscreteLevelsOptimizerWrapper
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.backtest import Backtester
from clustering import CorrelationClusterer


def create_optimizers(solver_type='simulated'):
    """Create optimizers for comparison."""
    optimizers = {}

    # Shared clustering
    clusterer = CorrelationClusterer(
        max_cluster_size=18,
        target_cluster_size=10
    )

    # 1. Original Single-Pass Quantum (Baseline)
    quantum_single = DiscreteLevelsOptimizer(
        max_cluster_size=18,
        n_bits=10,
        alpha=1.0,
        beta=1.0,
        lambda_budget=10.0,
        solver_type=solver_type,
        num_reads=1000,
        annealing_time=20,
        aggregation_strategy='concatenate',
        clusterer=clusterer,
        use_two_pass=False  # Original approach
    )
    optimizers['Quantum (Single-Pass)'] = DiscreteLevelsOptimizerWrapper(quantum_single)

    # 2. Two-Pass Hierarchical Quantum (New)
    quantum_twopass = DiscreteLevelsOptimizer(
        max_cluster_size=18,
        n_bits=10,
        alpha=1.0,  # Pass 1: intra-cluster
        beta=1.0,
        lambda_budget=10.0,
        solver_type=solver_type,
        num_reads=1000,
        annealing_time=20,
        aggregation_strategy='concatenate',  # Ignored in two-pass mode
        clusterer=clusterer,
        use_two_pass=True,  # Enable two-pass
        inter_cluster_alpha=1.0,  # Pass 2: inter-cluster
        inter_cluster_beta=1.0
    )
    optimizers['Quantum (Two-Pass)'] = DiscreteLevelsOptimizerWrapper(quantum_twopass)

    # 3. Two-Pass with Aggressive Inter-Cluster
    quantum_twopass_aggressive = DiscreteLevelsOptimizer(
        max_cluster_size=18,
        n_bits=10,
        alpha=1.0,
        beta=1.0,
        lambda_budget=10.0,
        solver_type=solver_type,
        num_reads=1000,
        annealing_time=20,
        aggregation_strategy='concatenate',
        clusterer=clusterer,
        use_two_pass=True,
        inter_cluster_alpha=3.0,  # More aggressive cluster allocation
        inter_cluster_beta=1.0
    )
    optimizers['Quantum (Two-Pass Aggressive)'] = DiscreteLevelsOptimizerWrapper(quantum_twopass_aggressive)

    # 4. Classical Mean-Variance (Benchmark)
    optimizers['Mean-Variance (Global)'] = ClassicalOptimizer(gamma=1.0, method='cvxpy')

    return optimizers


def run_comparison(portfolio_csv: str, train_days=252, test_days=21, step_days=21):
    """Run backtest comparison."""
    print("="*80)
    print("TWO-PASS HIERARCHICAL QUANTUM OPTIMIZATION TEST")
    print("="*80)
    print()

    # Load data
    print(f"Loading portfolio data from: {portfolio_csv}")
    prices = pd.read_csv(portfolio_csv, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    print(f"  Data: {len(prices)} days, {len(prices.columns)} assets")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
    print()

    # Create optimizers
    print("Creating optimizers...")
    optimizers = create_optimizers(solver_type='simulated')
    for name in optimizers.keys():
        print(f"  ✓ {name}")
    print()

    # Run backtests
    print("="*80)
    print("RUNNING BACKTESTS")
    print("="*80)
    print()

    results = {}

    for name, optimizer in optimizers.items():
        print(f"[{list(optimizers.keys()).index(name)+1}/{len(optimizers)}] Running: {name}...")

        try:
            backtester = Backtester(
                optimizer,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                initial_capital=100000.0
            )

            result = backtester.run(prices, returns)
            results[name] = result

            metrics = result['metrics']
            print(f"  ✓ Complete")
            print(f"    Sharpe: {metrics['sharpe_ratio']:.3f} | "
                  f"Return: {metrics['annualized_return']*100:.1f}% | "
                  f"MaxDD: {metrics['max_drawdown']*100:.1f}% | "
                  f"Avg Runtime: {metrics['avg_runtime']:.4f}s | "
                  f"Total: {metrics['total_runtime']:.1f}s")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Summary Table
    print("="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print()

    if results:
        summary_data = []
        for name, result in results.items():
            m = result['metrics']
            summary_data.append({
                'Optimizer': name,
                'Return (%)': m['annualized_return'] * 100,
                'Volatility (%)': m['annualized_volatility'] * 100,
                'Sharpe Ratio': m['sharpe_ratio'],
                'Max Drawdown (%)': m['max_drawdown'] * 100,
                'Avg Runtime (s)': m['avg_runtime'],
                'Final Value ($)': m['final_value']
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print()

        # Analysis
        print("="*80)
        print("ANALYSIS")
        print("="*80)
        print()

        if 'Quantum (Single-Pass)' in results and 'Quantum (Two-Pass)' in results:
            single_return = results['Quantum (Single-Pass)']['metrics']['annualized_return']
            twopass_return = results['Quantum (Two-Pass)']['metrics']['annualized_return']
            improvement = ((twopass_return / single_return) - 1) * 100

            print(f"Two-Pass vs Single-Pass Improvement:")
            print(f"  Return: {single_return*100:.2f}% → {twopass_return*100:.2f}% "
                  f"(+{improvement:.1f}%)")

        if 'Quantum (Two-Pass)' in results and 'Mean-Variance (Global)' in results:
            twopass_return = results['Quantum (Two-Pass)']['metrics']['annualized_return']
            classical_return = results['Mean-Variance (Global)']['metrics']['annualized_return']
            gap = ((classical_return / twopass_return) - 1) * 100

            print(f"\nTwo-Pass vs Classical Gap:")
            print(f"  Return: {twopass_return*100:.2f}% vs {classical_return*100:.2f}% "
                  f"(Gap: {gap:.1f}%)")

        print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test two-pass hierarchical quantum optimization')
    parser.add_argument('--portfolio-csv', type=str, default='portfolio.csv',
                       help='Path to portfolio CSV file')
    parser.add_argument('--train-days', type=int, default=252,
                       help='Training window size')
    parser.add_argument('--test-days', type=int, default=21,
                       help='Holding period')
    parser.add_argument('--step-days', type=int, default=21,
                       help='Rebalancing frequency')

    args = parser.parse_args()

    run_comparison(
        args.portfolio_csv,
        args.train_days,
        args.test_days,
        args.step_days
    )
