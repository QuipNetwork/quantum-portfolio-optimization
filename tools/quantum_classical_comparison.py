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

"""Comprehensive comparison of quantum vs classical portfolio optimizers."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
load_dotenv(project_root / '.env')

from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.equal_weight import EqualWeightOptimizer
from qpo.optimizers.risk_parity import RiskParityOptimizer
from qpo.optimizers.regularized import L1RegularizedOptimizer, L2RegularizedOptimizer
from qpo.optimizers.backtest import Backtester
from qpo.utils.data_prep import load_portfolio_data as load_portfolio_data_util
from qpo.utils.topology_selection import select_optimal_template
from clustering import (
    CorrelationClusterer, AntiCorrelationClusterer, GraphClusterer, SectorClusterer,
    CovarianceClusterer, ReturnsClusterer, VolatilityClusterer,
    DTWClusterer, FactorClusterer, UniformClusterer
)
from tools.visualizations import generate_all_visualizations


def load_portfolio_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load portfolio price data and compute returns.

    This is a wrapper around qpo.utils.data_prep.load_portfolio_data
    that adds progress logging.

    Args:
        csv_path: Path to CSV file with price data

    Returns:
        (prices, returns) DataFrames
    """
    print(f"Loading portfolio data from: {csv_path}")

    # Use centralized utility (ensures consistent price->returns conversion)
    prices, returns = load_portfolio_data_util(csv_path, preprocess=False)

    print(f"  Data: {len(prices)} days, {len(prices.columns)} assets")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")

    return prices, returns


def find_qpu_template_params(returns: pd.DataFrame, n_levels: int = 10) -> Dict[str, Any]:
    """
    Auto-detect optimal QPU parameters based on available templates.

    Tests different max_cluster_size values to find one that produces
    a cluster count matching an available template.

    Args:
        returns: Returns DataFrame
        n_levels: Number of discrete levels (default 10)

    Returns:
        Dict with template_name, max_cluster_size, n_clusters, or None if no match
    """
    import json
    template_dir = Path(__file__).parent.parent / 'embeddings' / 'templates'
    n_assets = len(returns.columns)

    # Find templates matching n_levels
    templates = list(template_dir.glob(f'portfolio_*c_*a_{n_levels}l*.json'))
    if not templates:
        print(f"  ⚠ No templates found for n_levels={n_levels}")
        return None

    # Parse template specs
    template_specs = []
    for tpath in templates:
        try:
            with open(tpath, 'r') as f:
                data = json.load(f)
            config = data.get('configuration', {})
            n_clusters_total = config.get('n_clusters_total', 0)
            assets_per = config.get('assets_per_cluster', 0)
            n_asset_clusters = n_clusters_total - 1  # Subtract meta-cluster

            template_specs.append({
                'name': tpath.name,
                'n_asset_clusters': n_asset_clusters,
                'assets_per_cluster': assets_per,
                'total': n_clusters_total
            })
        except Exception:
            continue

    if not template_specs:
        return None

    # Test max_cluster_size values to find template match
    # IMPORTANT: Templates are designed for UniformClusterer (equal-sized clusters)
    # Not compatible with variable-sized clustering like CorrelationClusterer
    print(f"  Auto-detecting template for {n_assets} assets...")

    # Try to match each template by configuring UniformClusterer appropriately
    # Strategy: Use templates that are >= our portfolio size (pad with dummy assets)
    viable_templates = []

    for spec in template_specs:
        target_n_asset_clusters = spec['n_asset_clusters']
        expected_assets_per = spec['assets_per_cluster']
        template_total_assets = target_n_asset_clusters * expected_assets_per

        # Template must be >= our portfolio size (we'll pad if larger)
        if template_total_assets >= n_assets:
            viable_templates.append({
                'spec': spec,
                'total_assets': template_total_assets,
                'padding_needed': template_total_assets - n_assets
            })

    # Sort by least padding needed (prefer exact match)
    viable_templates.sort(key=lambda x: x['padding_needed'])

    for tmpl in viable_templates:
        spec = tmpl['spec']
        target_n_asset_clusters = spec['n_asset_clusters']
        expected_assets_per = spec['assets_per_cluster']
        padding_needed = tmpl['padding_needed']

        # Calculate clustering parameters
        # Distribute real assets evenly, will pad to reach expected size
        target_cluster_size = int(n_assets / target_n_asset_clusters) + 1
        max_cluster_size = expected_assets_per  # Use exact template size

        try:
            clusterer = UniformClusterer(
                max_cluster_size=max_cluster_size,
                target_cluster_size=target_cluster_size
            )
            clusters = clusterer.cluster(returns)
            n_clusters = len(clusters)

            # Check if we got the right number of clusters
            if n_clusters == target_n_asset_clusters:
                cluster_sizes = [len(tickers) for tickers in clusters.values()]
                min_size = min(cluster_sizes)
                max_size_actual = max(cluster_sizes)

                # Allow template to be larger (will pad with dummy assets)
                if max_size_actual <= expected_assets_per:
                    if padding_needed > 0:
                        print(f"  ✓ Found match: {spec['name']} (with {padding_needed} dummy assets)")
                    else:
                        print(f"  ✓ Found match: {spec['name']} (exact fit)")
                    print(f"    {n_clusters} clusters, sizes {min_size}-{max_size_actual}, template expects {expected_assets_per}")

                    return {
                        'template_name': spec['name'],
                        'max_cluster_size': max_cluster_size,
                        'target_cluster_size': target_cluster_size,
                        'n_clusters': n_clusters,
                        'n_levels': n_levels,
                        'expected_assets_per_cluster': expected_assets_per,
                        'expected_n_clusters': target_n_asset_clusters,
                        'padding_needed': padding_needed,
                        'use_uniform_clustering': True
                    }
        except ValueError:
            # This configuration doesn't work, try next template
            continue

    print(f"  ⚠ No template matches portfolio size")
    return None


def create_optimizers(
    solver_types: List[str] = ['simulated'],
    clustering_methods: List[str] = ['correlation'],
    include_classical: bool = True,
    max_cluster_size: int = 18,
    target_cluster_size: int = 10,
    max_clusters: int = None,
    min_cluster_size: int = 2,
    portfolio_info_csv: str = None,
    optimizer_filter: List[str] = None,
    returns: pd.DataFrame = None,
    k_spread: float = 0.0
) -> Dict[str, Any]:
    """
    Create all optimizer instances for comparison.

    Args:
        solver_types: List of quantum solver types to test
        clustering_methods: List of clustering methods to test
        include_classical: Include classical optimizers
        max_cluster_size: Max assets per cluster for quantum
        target_cluster_size: Target average cluster size
        portfolio_info_csv: Path to portfolio info CSV for sector clustering
        optimizer_filter: List of optimizer types to include (None = all)
        returns: Returns DataFrame for template detection
        k_spread: Concentration parameter for weight spreading (0.0-3.0)

    Returns:
        Dictionary of {optimizer_name: optimizer_instance}
    """
    optimizers = {}

    # Helper to check if optimizer should be included
    def should_include(opt_type: str) -> bool:
        if optimizer_filter is None:
            return True
        return opt_type in optimizer_filter

    # Clustering method factory
    def create_clusterer(method: str):
        """Create clusterer instance based on method name."""
        common_args = {
            'max_cluster_size': max_cluster_size,
            'target_cluster_size': target_cluster_size,
            'max_clusters': max_clusters,
            'min_cluster_size': min_cluster_size
        }

        if method == 'correlation':
            return CorrelationClusterer(**common_args)
        elif method == 'anti_correlation':
            return AntiCorrelationClusterer(**common_args)
        elif method == 'uniform':
            return UniformClusterer(**common_args)
        elif method == 'graph':
            return GraphClusterer(**common_args)
        elif method == 'sector':
            return SectorClusterer(**common_args, sector_map=portfolio_info_csv)
        elif method == 'covariance':
            return CovarianceClusterer(**common_args)
        elif method == 'returns':
            return ReturnsClusterer(**common_args)
        elif method == 'volatility':
            return VolatilityClusterer(**common_args)
        elif method == 'dtw':
            return DTWClusterer(**common_args)
        elif method == 'factor':
            return FactorClusterer(**common_args)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    # Quantum optimizers (different solver types × clustering methods)
    if should_include('quantum'):
        print("\nConfiguring Quantum Optimizers...")

        # Auto-select optimal template if returns DataFrame is provided
        template_params = None
        if returns is not None:
            num_assets = len(returns.columns)
            print(f"  Auto-selecting template for {num_assets} assets...")

            template = select_optimal_template(num_assets)
            if template:
                print(f"  ✓ Selected template: {template['template_name']}")
                print(f"    - Clusters: {template['cluster_size']}, Assets/cluster: {template['assets_per_cluster']}")
                print(f"    - Levels: {template['num_levels']}, Capacity: {template['capacity']}, Waste: {template['waste']}")

                template_params = {
                    'n_levels': template['num_levels'],
                    'max_cluster_size': template['assets_per_cluster'],
                    'expected_assets_per_cluster': template['assets_per_cluster'],
                    'expected_n_clusters': template['cluster_size'],
                    'auto_select_template': False,
                }
            else:
                print(f"  ⚠ No suitable template found, using fallback parameters")
                template_params = {
                    'n_levels': 6,
                    'max_cluster_size': max_cluster_size,
                    'auto_select_template': True,
                }

        for solver_type in solver_types:
            for clustering_method in clustering_methods:
                try:
                    clusterer = create_clusterer(clustering_method)

                    # Base parameters
                    opt_params = {
                        'alpha': 10,
                        'beta': 2,
                        'solver_type': solver_type,
                        'clusterer': clusterer,
                        'l1_sparsity_penalty': 5.0,
                        'use_thermometer_cutoff': True,
                    }

                    # Add template parameters if available
                    if template_params:
                        opt_params.update(template_params)
                    else:
                        # Fallback if no returns provided
                        opt_params.update({
                            'n_levels': 6,
                            'max_cluster_size': max_cluster_size,
                            'auto_select_template': True,
                        })

                    quantum_opt = DiscreteLevelsOptimizer(**opt_params)

                    name = f"Quantum ({solver_type.upper()}, {clustering_method.capitalize()})"

                    optimizers[name] = quantum_opt
                    print(f"  ✓ {name}")
                except Exception as e:
                    print(f"  ✗ Failed to create {solver_type}/{clustering_method}: {e}")

    # Classical optimizers
    if include_classical:
        print("\nConfiguring Classical Optimizers...")

        if should_include('risk-parity'):
            optimizers['Risk Parity'] = RiskParityOptimizer()
            print("  ✓ Risk Parity")

        if should_include('equal-weight'):
            optimizers['Equal-Weight'] = EqualWeightOptimizer()
            print("  ✓ Equal-Weight")

        if should_include('mean-variance'):
            optimizers['Mean-Variance'] = ClassicalOptimizer(gamma=1.0, method='cvxpy')
            print("  ✓ Mean-Variance")

        if should_include('l1'):
            optimizers['L1 (λ=0.01)'] = L1RegularizedOptimizer(gamma=1.0, lambda_l1=0.01)
            print("  ✓ L1 Regularized")

        if should_include('l2'):
            optimizers['L2 (λ=0.1)'] = L2RegularizedOptimizer(gamma=1.0, lambda_l2=0.1)
            print("  ✓ L2 Regularized")

    return optimizers


def run_comparison(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    optimizers: Dict[str, Any],
    train_days: int = 252,
    test_days: int = 21,
    step_days: int = 21,
    initial_capital: float = 100000.0
) -> Dict[str, Dict[str, Any]]:
    """
    Run backtest comparison for all optimizers.

    Args:
        prices: Price DataFrame
        returns: Returns DataFrame
        optimizers: Dictionary of optimizers
        train_days: Training window size
        test_days: Test/holding period
        step_days: Rebalancing frequency
        initial_capital: Starting capital

    Returns:
        Dictionary of {optimizer_name: backtest_result}
    """
    print("\n" + "="*80)
    print("RUNNING BACKTESTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Training window: {train_days} days (~{train_days//252} year)")
    print(f"  Holding period: {test_days} days (~{test_days//21} month)")
    print(f"  Rebalancing: Every {step_days} days")
    print(f"  Initial capital: ${initial_capital:,.0f}")

    # Calculate expected number of rebalances
    n_rebalances = (len(prices) - train_days - test_days) // step_days + 1
    print(f"  Expected rebalances: ~{n_rebalances}")

    results = {}

    for i, (name, optimizer) in enumerate(optimizers.items(), 1):
        print(f"\n[{i}/{len(optimizers)}] Running: {name}...")

        start_time = time.time()

        try:
            backtester = Backtester(
                optimizer,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                initial_capital=initial_capital
            )

            result = backtester.run(prices, returns)

            elapsed = time.time() - start_time

            # Add solver-specific metrics if available
            if len(result['weights_history']) > 0:
                solver_runtimes = []
                qpu_times = []
                network_times = []

                for w in result['weights_history']:
                    metrics = w.get('metrics', {})
                    if 'solver_only_runtime' in metrics:
                        solver_runtimes.append(metrics['solver_only_runtime'])
                    if 'qpu_access_time' in metrics:
                        qpu_times.append(metrics['qpu_access_time'])
                    if 'network_latency' in metrics:
                        network_times.append(metrics['network_latency'])

                if solver_runtimes:
                    result['metrics']['avg_solver_only_runtime'] = np.mean(solver_runtimes)
                if qpu_times:
                    result['metrics']['avg_qpu_access_time'] = np.mean(qpu_times)
                if network_times:
                    result['metrics']['avg_network_latency'] = np.mean(network_times)

            results[name] = result

            # Print summary
            m = result['metrics']
            print(f"  ✓ Complete in {elapsed:.1f}s")
            print(f"    Sharpe: {m['sharpe_ratio']:.3f} | "
                  f"Return: {m['annualized_return']*100:.1f}% | "
                  f"MaxDD: {m['max_drawdown']*100:.1f}% | "
                  f"Avg Runtime: {m['avg_runtime']:.4f}s")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Quantum vs Classical Portfolio Optimization Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--portfolio-csv', type=str, required=True,
                       help='Path to portfolio price CSV file')
    parser.add_argument('--portfolio-info-csv', type=str, default=None,
                       help='Path to portfolio info CSV (for sector clustering)')
    parser.add_argument('--train-days', type=int, default=252,
                       help='Training window size in days (default: 252 = 1 year)')
    parser.add_argument('--test-days', type=int, default=21,
                       help='Holding period in days (default: 21 = 1 month)')
    parser.add_argument('--step-days', type=int, default=21,
                       help='Rebalancing frequency in days (default: 21)')
    parser.add_argument('--solver-types', nargs='+', default=['simulated'],
                       choices=['simulated', 'qpu', 'hybrid'],
                       help='Quantum solver types to test (default: simulated)')
    parser.add_argument('--no-classical', action='store_true',
                       help='Exclude classical optimizers')
    parser.add_argument('--output-dir', type=str, default='output/quantum_benchmark',
                       help='Output directory for plots and results')
    parser.add_argument('--max-cluster-size', type=int, default=24,
                       help='Maximum cluster size for quantum optimizer (tuned for MV baseline matching)')
    parser.add_argument('--target-cluster-size', type=int, default=10,
                       help='Target average cluster size')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial portfolio capital (default: 100000)')
    parser.add_argument('--max-clusters', type=int, default=None,
                       help='Maximum number of clusters (hard constraint for fixed template)')
    parser.add_argument('--min-cluster-size', type=int, default=2,
                       help='Minimum assets per cluster (default: 2, avoids single-asset clusters)')
    parser.add_argument('--clustering-methods', nargs='+',
                       default=['all'],
                       choices=['all', 'correlation', 'anti_correlation', 'uniform', 'graph', 'sector', 'covariance',
                               'returns', 'volatility', 'dtw', 'factor'],
                       help='Clustering methods to test for quantum optimizer (default: all). Use "all" to run all methods.')
    parser.add_argument('--optimizers', nargs='+',
                       choices=['quantum', 'mean-variance', 'l1', 'l2', 'risk-parity', 'equal-weight'],
                       default=None,
                       help='Specific optimizers to run (default: all). Use to isolate and test individual optimizers.')
    parser.add_argument('--k-spread', type=float, default=0.0,
                       help='Concentration parameter for weight spreading (0.0=no spread, 1.0-3.0=moderate spread, default: 0.0)')

    args = parser.parse_args()

    # Handle "all" clustering methods
    all_clustering_methods = ['correlation', 'anti_correlation', 'uniform', 'graph', 'sector', 'covariance',
                              'returns', 'volatility', 'dtw', 'factor']
    if 'all' in args.clustering_methods:
        clustering_methods = all_clustering_methods
    else:
        clustering_methods = args.clustering_methods

    # Header
    print("="*80)
    print("QUANTUM VS CLASSICAL PORTFOLIO OPTIMIZATION BENCHMARK")
    print("="*80)

    # Load data
    prices, returns = load_portfolio_data(args.portfolio_csv)

    # Create optimizers
    optimizers = create_optimizers(
        solver_types=args.solver_types,
        clustering_methods=clustering_methods,
        include_classical=not args.no_classical,
        max_cluster_size=args.max_cluster_size,
        target_cluster_size=args.target_cluster_size,
        max_clusters=args.max_clusters,
        min_cluster_size=args.min_cluster_size,
        portfolio_info_csv=args.portfolio_info_csv,
        optimizer_filter=args.optimizers,
        returns=returns,  # Pass returns for QPU template auto-detection
        k_spread=args.k_spread
    )

    # Run comparison
    backtest_results = run_comparison(
        prices,
        returns,
        optimizers,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        initial_capital=args.initial_capital
    )

    # Generate visualizations
    if backtest_results:
        output_dir = Path(args.output_dir)
        generate_all_visualizations(
            backtest_results,
            output_dir=output_dir,
            generate_individual_weights=True
        )

        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        print(f"\n✓ All results saved to: {output_dir}/")
        print(f"  - portfolio_value_comparison.png")
        print(f"  - runtime_comparison.png")
        print(f"  - risk_return_scatter.png")
        print(f"  - comparison_metrics.csv")
        print(f"  - weights_*.png (per optimizer)")
    else:
        print("\n✗ No successful backtests completed")
        sys.exit(1)


if __name__ == '__main__':
    main()
