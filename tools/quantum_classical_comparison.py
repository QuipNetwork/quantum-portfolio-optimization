"""Comprehensive comparison of quantum vs classical portfolio optimizers."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qpo.optimizers.quantum import IndependentClustersOptimizer, QuantumOptimizerWrapper
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.equal_weight import EqualWeightOptimizer
from qpo.optimizers.risk_parity import RiskParityOptimizer
from qpo.optimizers.regularized import L1RegularizedOptimizer, L2RegularizedOptimizer
from qpo.optimizers.backtest import Backtester
from clustering import (
    CorrelationClusterer, GraphClusterer, SectorClusterer,
    CovarianceClusterer, ReturnsClusterer, VolatilityClusterer,
    DTWClusterer, FactorClusterer
)
from tools.visualizations import generate_all_visualizations


def load_portfolio_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load portfolio price data and compute returns.

    Args:
        csv_path: Path to CSV file with price data

    Returns:
        (prices, returns) DataFrames
    """
    print(f"Loading portfolio data from: {csv_path}")
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()

    print(f"  Data: {len(prices)} days, {len(prices.columns)} assets")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")

    return prices, returns


def create_optimizers(
    solver_types: List[str] = ['simulated'],
    clustering_methods: List[str] = ['correlation'],
    include_classical: bool = True,
    max_cluster_size: int = 18,
    target_cluster_size: int = 10,
    portfolio_info_csv: str = None,
    optimizer_filter: List[str] = None
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
            'target_cluster_size': target_cluster_size
        }

        if method == 'correlation':
            return CorrelationClusterer(**common_args)
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
        for solver_type in solver_types:
            for clustering_method in clustering_methods:
                try:
                    clusterer = create_clusterer(clustering_method)

                    quantum_opt = IndependentClustersOptimizer(
                        max_cluster_size=max_cluster_size,
                        n_bits=10,
                        alpha=1.0,  # Return coefficient
                        beta=1.0,   # Risk coefficient
                        lambda_budget=10.0,
                        solver_type=solver_type,
                        num_reads=1000,
                        annealing_time=20,
                        aggregation_strategy='proportional',
                        clusterer=clusterer
                    )

                    # Wrap for Backtester compatibility
                    wrapped = QuantumOptimizerWrapper(quantum_opt)

                    name = f"Quantum ({solver_type.upper()}, {clustering_method.capitalize()})"
                    optimizers[name] = wrapped
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
                for w in result['weights_history']:
                    if 'solver_only_runtime' in w.get('metrics', {}):
                        solver_runtimes.append(w['metrics']['solver_only_runtime'])

                if solver_runtimes:
                    result['metrics']['avg_solver_only_runtime'] = np.mean(solver_runtimes)

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
    parser.add_argument('--max-cluster-size', type=int, default=18,
                       help='Maximum cluster size for quantum optimizer')
    parser.add_argument('--target-cluster-size', type=int, default=10,
                       help='Target average cluster size')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial portfolio capital (default: 100000)')
    parser.add_argument('--clustering-methods', nargs='+',
                       default=['all'],
                       choices=['all', 'correlation', 'graph', 'sector', 'covariance',
                               'returns', 'volatility', 'dtw', 'factor'],
                       help='Clustering methods to test for quantum optimizer (default: all). Use "all" to run all methods.')
    parser.add_argument('--optimizers', nargs='+',
                       choices=['quantum', 'mean-variance', 'l1', 'l2', 'risk-parity', 'equal-weight'],
                       default=None,
                       help='Specific optimizers to run (default: all). Use to isolate and test individual optimizers.')

    args = parser.parse_args()

    # Handle "all" clustering methods
    all_clustering_methods = ['correlation', 'graph', 'sector', 'covariance',
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
        portfolio_info_csv=args.portfolio_info_csv,
        optimizer_filter=args.optimizers
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
