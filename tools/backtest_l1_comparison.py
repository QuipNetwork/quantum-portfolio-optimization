#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Backtest comparison of different L1 sparsity penalty values.

Tests λ = [0, 0.5, 1, 2, 3, 5] to find optimal sparsity-performance tradeoff.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from qpo.utils.data_prep import load_portfolio_data
from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer
from qpo.optimizers.backtest import Backtester


def run_l1_backtest_comparison(
    portfolio_csv: str = 'portfolio.csv',
    train_days: int = 126,  # 6 months
    test_days: int = 21,    # 1 month
    step_days: int = 21,    # Monthly rebalancing
    lambda_values: list = None
):
    """
    Run backtest comparing different L1 penalty values.

    Args:
        portfolio_csv: Path to portfolio CSV
        train_days: Training window (days)
        test_days: Test/holding period (days)
        step_days: Rebalancing frequency (days)
        lambda_values: List of L1 penalty values to test
    """
    if lambda_values is None:
        lambda_values = [0, 0.5, 1, 2, 3, 5]

    print("="*80)
    print("L1 SPARSITY PENALTY BACKTEST COMPARISON")
    print("="*80)
    print(f"\nTesting λ values: {lambda_values}")
    print(f"Train window: {train_days} days")
    print(f"Test period: {test_days} days")
    print(f"Rebalancing: Every {step_days} days")
    print()

    # Load data
    print("Loading portfolio data...")
    prices, returns = load_portfolio_data(portfolio_csv)

    # Use recent data for faster backtesting
    lookback_days = 252  # ~1 year only
    prices = prices.iloc[-lookback_days:]
    returns = returns.iloc[-lookback_days:]

    print(f"Data loaded: {len(prices)} days, {len(prices.columns)} assets")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print()

    # Run backtests for each lambda value
    results = []

    for lam in lambda_values:
        print(f"\n{'='*80}")
        print(f"Testing λ = {lam}")
        print(f"{'='*80}")

        # Create optimizer with this lambda
        # Match detailed_optimizer_comparison.py configuration
        from clustering import CorrelationClusterer

        clusterer = CorrelationClusterer(
            max_cluster_size=15,
            target_cluster_size=14
        )

        optimizer = DiscreteLevelsOptimizer(
            n_levels=4,
            alpha=10.0,
            beta=5.0,
            thermometer_penalty=10.0,
            solver_type='simulated',
            num_reads=256,
            num_sweeps=256,
            clusterer=clusterer,
            l1_sparsity_penalty=lam,
            use_thermometer_cutoff=False
        )

        # Create backtester
        backtester = Backtester(
            optimizer=optimizer,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            initial_capital=100000.0
        )

        # Run backtest
        try:
            backtest_result = backtester.run(prices, returns)

            # Extract metrics
            portfolio_values = backtest_result['portfolio_values']
            weights_history = backtest_result['weights_history']
            metrics = backtest_result['metrics']

            # Compute average holdings per rebalance
            # weights_history is a list of dicts with 'weights' key
            weight_series_list = [pd.Series(w['weights']) for w in weights_history]

            avg_holdings = np.mean([
                (w > 1e-6).sum() for w in weight_series_list
            ])

            # Compute average max weight
            avg_max_weight = np.mean([
                w.max() for w in weight_series_list
            ])

            # Compute average Gini coefficient
            def gini(w):
                w_sorted = np.sort(w.values)
                n = len(w)
                index = np.arange(1, n + 1)
                return (2 * np.sum(index * w_sorted)) / (n * np.sum(w_sorted)) - (n + 1) / n

            avg_gini = np.mean([gini(w) for w in weight_series_list])

            # Print summary
            print(f"\n📊 Results for λ={lam}:")
            print(f"  Total Return:     {metrics['total_return']:+.2%}")
            print(f"  Annualized Ret:   {metrics['annualized_return']:+.2%}")
            print(f"  Volatility:       {metrics['volatility']:.2%}")
            print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:     {metrics['max_drawdown']:.2%}")
            print(f"  Avg Holdings:     {avg_holdings:.1f}")
            print(f"  Avg Max Weight:   {avg_max_weight:.2%}")
            print(f"  Avg Gini:         {avg_gini:.3f}")
            print(f"  Rebalances:       {len(weights_history)}")
            print(f"  Avg Runtime:      {np.mean(backtest_result['runtimes']):.2f}s")

            results.append({
                'lambda': lam,
                'total_return': metrics['total_return'],
                'annualized_return': metrics['annualized_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'avg_holdings': avg_holdings,
                'avg_max_weight': avg_max_weight,
                'avg_gini': avg_gini,
                'n_rebalances': len(weights_history),
                'avg_runtime': np.mean(backtest_result['runtimes']),
                'final_value': portfolio_values.iloc[-1]
            })

        except Exception as e:
            print(f"❌ Error with λ={lam}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    if not results:
        print("\n❌ No results to display - all backtests failed!")
        return None

    df = pd.DataFrame(results)

    # Format for display
    print("\nPerformance Metrics:")
    print(df[['lambda', 'total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown']].to_string(index=False))

    print("\n\nPortfolio Characteristics:")
    print(df[['lambda', 'avg_holdings', 'avg_max_weight', 'avg_gini']].to_string(index=False))

    print("\n\nOperational Metrics:")
    print(df[['lambda', 'n_rebalances', 'avg_runtime']].to_string(index=False))

    # Identify best performers
    print("\n" + "="*80)
    print("BEST PERFORMERS")
    print("="*80)

    if len(df) > 0:
        best_return_idx = df['total_return'].idxmax()
        best_sharpe_idx = df['sharpe_ratio'].idxmax()
        best_drawdown_idx = df['max_drawdown'].idxmax()
        most_sparse_idx = df['avg_holdings'].idxmin()

        if pd.notna(best_return_idx):
            best_return = df.loc[best_return_idx]
            print(f"\n🏆 Highest Return:    λ={best_return['lambda']:.1f}  ({best_return['total_return']:+.2%})")

        if pd.notna(best_sharpe_idx):
            best_sharpe = df.loc[best_sharpe_idx]
            print(f"⭐ Best Sharpe:       λ={best_sharpe['lambda']:.1f}  ({best_sharpe['sharpe_ratio']:.2f})")

        if pd.notna(best_drawdown_idx):
            best_drawdown = df.loc[best_drawdown_idx]
            print(f"🛡️  Smallest Drawdown: λ={best_drawdown['lambda']:.1f}  ({best_drawdown['max_drawdown']:.2%})")

        if pd.notna(most_sparse_idx):
            most_sparse = df.loc[most_sparse_idx]
            print(f"🎯 Most Sparse:       λ={most_sparse['lambda']:.1f}  ({most_sparse['avg_holdings']:.1f} holdings)")

    # Save results
    output_file = Path('backtest_l1_comparison_results.csv')
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Backtest L1 penalty comparison')
    parser.add_argument('--portfolio', type=str, default='portfolio.csv',
                       help='Path to portfolio CSV')
    parser.add_argument('--train-days', type=int, default=126,
                       help='Training window size (days)')
    parser.add_argument('--test-days', type=int, default=21,
                       help='Test/holding period (days)')
    parser.add_argument('--step-days', type=int, default=21,
                       help='Rebalancing frequency (days)')
    parser.add_argument('--lambda-values', type=float, nargs='+',
                       default=[0, 0.5, 1, 2, 3, 5],
                       help='L1 penalty values to test')

    args = parser.parse_args()

    results = run_l1_backtest_comparison(
        portfolio_csv=args.portfolio,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        lambda_values=args.lambda_values
    )
