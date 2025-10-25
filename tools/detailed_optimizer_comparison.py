#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Detailed comparison of MeanVariance vs QPU optimizers with diagnostics.

Shows:
- YTD and period returns
- Top 10 picks with weights
- Cluster structure and weights (for QPU)
- Individual asset weights within each cluster (for QPU)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / '.env')

from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.backtest import Backtester
from qpo.utils.data_prep import load_portfolio_data
from qpo.utils.topology_selection import select_optimal_template
from clustering import UniformClusterer, CorrelationClusterer


class DetailedOptimizerComparison:
    """Run detailed comparison between optimizers with diagnostics."""

    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        """
        Initialize comparison framework.

        Args:
            prices: Price DataFrame (dates × tickers)
            returns: Returns DataFrame (dates × tickers)
        """
        self.prices = prices
        self.returns = returns
        self.results = {}

    def _format_top_picks(self, weights: pd.Series, n: int = 10) -> str:
        """Format top N picks with weights."""
        top_picks = weights.nlargest(n)
        lines = []
        for ticker, weight in top_picks.items():
            lines.append(f"    {ticker:8s} ({weight:6.2%})")
        return "\n".join(lines)

    def _extract_qpu_cluster_details(self, optimizer: DiscreteLevelsOptimizer,
                                     returns_window: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract cluster structure from QPU optimizer.

        Returns cluster weights and per-cluster asset breakdowns.
        """
        # Get cluster assignments
        clusters = optimizer.clusterer.cluster(returns_window)

        # Run optimization to get weights
        result = optimizer.optimize(returns_window)
        weights = result['weights']

        # Group weights by cluster
        cluster_details = {}
        cluster_totals = {}

        for cluster_id, tickers in clusters.items():
            cluster_weights = weights[weights.index.isin(tickers)]
            cluster_total = cluster_weights.sum()

            cluster_totals[cluster_id] = cluster_total
            cluster_details[cluster_id] = {
                'total_weight': cluster_total,
                'tickers': tickers,
                'weights': cluster_weights.sort_values(ascending=False)
            }

        return {
            'cluster_totals': pd.Series(cluster_totals).sort_values(ascending=False),
            'cluster_details': cluster_details,
            'n_clusters': len(clusters)
        }

    def _print_qpu_diagnostics(self, cluster_info: Dict[str, Any], test_returns: pd.DataFrame):
        """Print detailed QPU cluster diagnostics with return information."""
        print("\n  📊 CLUSTER STRUCTURE:")
        print(f"    Total clusters: {cluster_info['n_clusters']}")
        print("\n  🎯 CLUSTER WEIGHTS (Meta-cluster allocation):")

        cluster_totals = cluster_info['cluster_totals']
        for cluster_id, weight in cluster_totals.items():
            print(f"    {cluster_id:15s} {weight:6.2%}")

        print("\n  🔍 INDIVIDUAL ASSET WEIGHTS BY CLUSTER:")

        # Compute YTD and period returns for all assets
        ytd_returns = self.returns.iloc[-252:].mean() * 252  # Annualized YTD
        period_returns = test_returns.sum()  # Total return over test period

        cluster_details = cluster_info['cluster_details']
        for cluster_id in cluster_totals.index:  # Print in order of weight
            details = cluster_details[cluster_id]
            print(f"\n    {cluster_id} (total: {details['total_weight']:.2%}):")
            print(f"      {'Ticker':<8s} {'Weight':>6s}  {'%Clust':>6s}  {'YTD Ret':>8s}  {'Period':>8s}")
            print(f"      {'-'*8} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}")

            # Print top assets in this cluster
            for ticker, weight in details['weights'].head(10).items():
                pct_of_cluster = (weight / details['total_weight'] * 100) if details['total_weight'] > 0 else 0
                ytd_ret = ytd_returns.get(ticker, 0)
                period_ret = period_returns.get(ticker, 0)

                print(f"      {ticker:<8s} {weight:5.2%}  {pct_of_cluster:5.1f}%  {ytd_ret:+7.2%}  {period_ret:+7.2%}")

            if len(details['weights']) > 10:
                print(f"      ... and {len(details['weights']) - 10} more assets")

    def run_single_period(self, train_days: int = 252, test_days: int = 21,
                         qpu_solver: str = 'simulated',
                         qpu_clusterer: str = 'uniform',
                         qpu_params: Dict = None) -> Dict[str, Any]:
        """
        Run single-period comparison (most recent data).

        Args:
            train_days: Training window size
            test_days: Test period size
            qpu_solver: 'simulated' or 'qpu'
            qpu_clusterer: 'uniform' or 'correlation'
            qpu_params: Dict with n_levels, max_cluster_size, etc.

        Returns:
            Comparison results with diagnostics
        """
        print("="*80)
        print("DETAILED OPTIMIZER COMPARISON - SINGLE PERIOD")
        print("="*80)

        # Use most recent data
        total_days = train_days + test_days
        if len(self.returns) < total_days:
            raise ValueError(f"Need at least {total_days} days, have {len(self.returns)}")

        train_returns = self.returns.iloc[-total_days:-test_days]
        test_returns = self.returns.iloc[-test_days:]

        train_start = train_returns.index[0]
        train_end = train_returns.index[-1]
        test_start = test_returns.index[0]
        test_end = test_returns.index[-1]

        print(f"\nTraining period: {train_start.date()} to {train_end.date()} ({len(train_returns)} days)")
        print(f"Testing period:  {test_start.date()} to {test_end.date()} ({len(test_returns)} days)")
        print(f"Assets: {len(self.returns.columns)}")

        # Configure QPU optimizer
        if qpu_params is None:
            qpu_params = {}

        # Use auto-select template (no manual selection needed)
        num_assets = len(train_returns.columns)
        print(f"\n📐 Will auto-select template for {num_assets} assets during optimization...")

        n_levels = qpu_params.get('n_levels', 8)
        max_cluster_size = qpu_params.get('max_cluster_size', 16)
        auto_select_template = True  # Let the optimizer auto-select

        alpha = qpu_params.get('alpha', 10.0)
        beta = qpu_params.get('beta', 2.0)

        # Create clusterer
        if qpu_clusterer == 'uniform':
            clusterer = UniformClusterer(
                max_cluster_size=max_cluster_size,
                target_cluster_size=max_cluster_size
            )
        elif qpu_clusterer == 'correlation':
            clusterer = CorrelationClusterer(
                max_cluster_size=max_cluster_size,
                target_cluster_size=max_cluster_size
            )
        else:
            raise ValueError(f"Unknown clusterer: {qpu_clusterer}")

        # Create optimizers
        print(f"\n📐 Configuring optimizers...")
        print(f"  QPU: solver={qpu_solver}, clusterer={qpu_clusterer}, n_levels={n_levels}, "
              f"alpha={alpha}, beta={beta}")

        mv_optimizer = ClassicalOptimizer(
            gamma=1.0,
            method='cvxpy'
        )

        qpu_optimizer = DiscreteLevelsOptimizer(
            n_levels=n_levels,
            alpha=alpha,
            beta=beta,
            thermometer_penalty=10.0,
            solver_type=qpu_solver,
            clusterer=clusterer,
            max_cluster_size=max_cluster_size,
            auto_select_template=auto_select_template,
            l1_sparsity_penalty=qpu_params.get('l1_sparsity_penalty', 2.0),
            use_thermometer_cutoff=qpu_params.get('use_thermometer_cutoff', True)
        )

        # Run MeanVariance
        print("\n" + "="*80)
        print("OPTIMIZER RESULTS")
        print("="*80)

        mv_result = mv_optimizer.optimize(train_returns)
        mv_weights = mv_result['weights']

        # Run QPU
        # Extract cluster details BEFORE optimization
        cluster_info = self._extract_qpu_cluster_details(qpu_optimizer, train_returns)
        qpu_weights = qpu_optimizer.optimize(train_returns)['weights']

        # Compute returns
        mv_period_return = (test_returns * mv_weights).sum(axis=1).sum()
        mv_ytd_return = (self.returns.iloc[-252:] * mv_weights).sum(axis=1).sum()
        qpu_period_return = (test_returns * qpu_weights).sum(axis=1).sum()
        qpu_ytd_return = (self.returns.iloc[-252:] * qpu_weights).sum(axis=1).sum()

        # Print returns side by side
        print(f"\n📈 RETURNS:")
        print(f"{'Metric':<20} {'MeanVariance':>15} {'QPU':>15}")
        print("-"*52)
        print(f"{'Period Return':<20} {mv_period_return:>14.2%} {qpu_period_return:>14.2%}")
        print(f"{'YTD Return':<20} {mv_ytd_return:>14.2%} {qpu_ytd_return:>14.2%}")

        # Print top 10 picks side by side with returns
        print(f"\n🏆 TOP 10 PICKS:")
        mv_top = mv_weights.nlargest(10)
        qpu_top = qpu_weights.nlargest(10)

        # Compute returns for display
        ytd_returns = self.returns.iloc[-252:].mean() * 252  # Annualized YTD
        period_returns = test_returns.sum()  # Total return over test period

        print(f"{'Rank':<6} {'MeanVariance':<12} {'Weight':>7} {'YTD':>8} {'Period':>8}  {'│':^3}  "
              f"{'QPU':<12} {'Weight':>7} {'YTD':>8} {'Period':>8}")
        print("-"*100)

        for i in range(10):
            mv_ticker = mv_top.index[i] if i < len(mv_top) else ""
            mv_weight = mv_top.iloc[i] if i < len(mv_top) else 0
            mv_ytd = ytd_returns.get(mv_ticker, 0) if mv_ticker else 0
            mv_period = period_returns.get(mv_ticker, 0) if mv_ticker else 0

            qpu_ticker = qpu_top.index[i] if i < len(qpu_top) else ""
            qpu_weight = qpu_top.iloc[i] if i < len(qpu_top) else 0
            qpu_ytd = ytd_returns.get(qpu_ticker, 0) if qpu_ticker else 0
            qpu_period = period_returns.get(qpu_ticker, 0) if qpu_ticker else 0

            print(f"{i+1:<6} {mv_ticker:<12} {mv_weight:>6.2%} {mv_ytd:>+7.2%} {mv_period:>+7.2%}  {'│':^3}  "
                  f"{qpu_ticker:<12} {qpu_weight:>6.2%} {qpu_ytd:>+7.2%} {qpu_period:>+7.2%}")

        # Print cluster diagnostics
        self._print_qpu_diagnostics(cluster_info, test_returns)

        # Summary comparison
        print("\n" + "="*80)
        print("📊 SUMMARY COMPARISON")
        print("="*80)
        print(f"\n{'Metric':<20} {'MeanVariance':>15} {'QPU':>15} {'Difference':>15}")
        print("-"*70)
        print(f"{'Period Return':<20} {mv_period_return:>14.2%} {qpu_period_return:>14.2%} {qpu_period_return-mv_period_return:>+14.2%}")
        print(f"{'YTD Return':<20} {mv_ytd_return:>14.2%} {qpu_ytd_return:>14.2%} {qpu_ytd_return-mv_ytd_return:>+14.2%}")
        print(f"{'Non-zero weights':<20} {(mv_weights > 1e-6).sum():>15} {(qpu_weights > 1e-6).sum():>15} {(qpu_weights > 1e-6).sum() - (mv_weights > 1e-6).sum():>+15}")
        print(f"{'Weight std dev':<20} {mv_weights.std():>15.4f} {qpu_weights.std():>15.4f} {qpu_weights.std()-mv_weights.std():>+15.4f}")

        return {
            'mean_variance': {
                'weights': mv_weights,
                'period_return': mv_period_return,
                'ytd_return': mv_ytd_return
            },
            'qpu': {
                'weights': qpu_weights,
                'period_return': qpu_period_return,
                'ytd_return': qpu_ytd_return,
                'cluster_info': cluster_info
            }
        }

    def run_rolling_backtest(self, train_days: int = 252, test_days: int = 21,
                            step_days: int = 21, n_periods: int = 12,
                            qpu_solver: str = 'simulated',
                            qpu_params: Dict = None) -> Dict[str, Any]:
        """
        Run rolling window backtest with detailed tracking.

        Args:
            train_days: Training window
            test_days: Test period
            step_days: Rebalancing frequency
            n_periods: Number of periods to test
            qpu_solver: 'simulated' or 'qpu'
            qpu_params: QPU configuration

        Returns:
            Full backtest results with period-by-period breakdown
        """
        print("="*80)
        print("ROLLING BACKTEST COMPARISON")
        print("="*80)

        if qpu_params is None:
            qpu_params = {}

        # Use auto-select template (no manual selection needed)
        num_assets = len(self.returns.columns)
        print(f"\nWill auto-select template for {num_assets} assets during optimization...")

        n_levels = qpu_params.get('n_levels', 8)
        max_cluster_size = qpu_params.get('max_cluster_size', 16)

        # Create optimizers
        mv_optimizer = ClassicalOptimizer(gamma=1.0, method='cvxpy')

        qpu_optimizer = DiscreteLevelsOptimizer(
            n_levels=n_levels,
            alpha=qpu_params.get('alpha', 10.0),
            beta=qpu_params.get('beta', 2.0),
            thermometer_penalty=10.0,
            solver_type=qpu_solver,
            clusterer=UniformClusterer(max_cluster_size=max_cluster_size, target_cluster_size=max_cluster_size//2),
            max_cluster_size=max_cluster_size,
            auto_select_template=True,  # Let the optimizer auto-select
            l1_sparsity_penalty=qpu_params.get('l1_sparsity_penalty', 2.0),
            use_thermometer_cutoff=qpu_params.get('use_thermometer_cutoff', True)
        )

        # Run backtests
        mv_backtester = Backtester(mv_optimizer, train_days, test_days, step_days)
        qpu_backtester = Backtester(qpu_optimizer, train_days, test_days, step_days)

        print(f"\nRunning MeanVariance backtest...")
        mv_results = mv_backtester.run(self.prices, self.returns)

        print(f"Running QPU backtest...")
        qpu_results = qpu_backtester.run(self.prices, self.returns)

        # Compute period returns
        mv_values = pd.Series(mv_results['portfolio_values'], index=mv_results['rebalance_dates'])
        qpu_values = pd.Series(qpu_results['portfolio_values'], index=qpu_results['rebalance_dates'])

        mv_period_returns = mv_values.pct_change().dropna()
        qpu_period_returns = qpu_values.pct_change().dropna()

        # Get weights history
        mv_weights_history = mv_results['weights_history']
        qpu_weights_history = qpu_results['weights_history']
        rebalance_dates = mv_results['rebalance_dates']

        # Print detailed period-by-period results with top picks
        print("\n" + "="*80)
        print("PERIOD-BY-PERIOD RESULTS WITH TOP PICKS")
        print("="*80)

        # Limit to last n_periods
        start_idx = max(0, len(rebalance_dates) - n_periods)

        for i in range(start_idx, len(rebalance_dates)):
            date = rebalance_dates[i]

            # Get returns for this period
            if i > 0 and date in mv_period_returns.index:
                mv_ret = mv_period_returns[date]
                qpu_ret = qpu_period_returns[date]
                diff = qpu_ret - mv_ret
            else:
                mv_ret = qpu_ret = diff = 0.0

            print(f"\n{'='*80}")
            print(f"Period ending {date.date()}  │  MV: {mv_ret:+.2%}  QPU: {qpu_ret:+.2%}  Diff: {diff:+.2%}")
            print(f"{'='*80}")

            # Get weights for this period
            # weights_history contains dicts with 'weights' key
            mv_weights_dict = mv_weights_history[i]
            qpu_weights_dict = qpu_weights_history[i]

            # Extract weights Series
            if isinstance(mv_weights_dict, dict) and 'weights' in mv_weights_dict:
                mv_weights = pd.Series(mv_weights_dict['weights'])
            else:
                mv_weights = pd.Series(mv_weights_dict) if isinstance(mv_weights_dict, dict) else mv_weights_dict

            if isinstance(qpu_weights_dict, dict) and 'weights' in qpu_weights_dict:
                qpu_weights = pd.Series(qpu_weights_dict['weights'])
            else:
                qpu_weights = pd.Series(qpu_weights_dict) if isinstance(qpu_weights_dict, dict) else qpu_weights_dict

            # Top 5 picks for this period
            mv_top5 = mv_weights.nlargest(5)
            qpu_top5 = qpu_weights.nlargest(5)

            # Compute period returns for assets
            # Get the test window for this period using iloc to avoid timezone issues
            test_start_idx = i

            # Find the date position in the returns index
            try:
                date_pos = self.returns.index.get_loc(date)
            except KeyError:
                # If exact date not found, use nearest
                date_pos = self.returns.index.get_indexer([date], method='nearest')[0]

            if test_start_idx < len(rebalance_dates) - 1:
                next_date = rebalance_dates[test_start_idx + 1]
                try:
                    next_date_pos = self.returns.index.get_loc(next_date)
                except KeyError:
                    next_date_pos = self.returns.index.get_indexer([next_date], method='nearest')[0]

                if date_pos >= 0 and next_date_pos > date_pos:
                    period_data = self.returns.iloc[date_pos:next_date_pos+1]
                    if len(period_data) > 1:
                        asset_period_returns = period_data.iloc[1:].sum()  # Skip first day (rebalance day)
                    else:
                        asset_period_returns = pd.Series(0, index=self.returns.columns)
                else:
                    asset_period_returns = pd.Series(0, index=self.returns.columns)
            else:
                # Last period - use remaining data
                if date_pos >= 0 and date_pos < len(self.returns) - 1:
                    period_data = self.returns.iloc[date_pos:]
                    if len(period_data) > 1:
                        asset_period_returns = period_data.iloc[1:].sum()
                    else:
                        asset_period_returns = pd.Series(0, index=self.returns.columns)
                else:
                    asset_period_returns = pd.Series(0, index=self.returns.columns)

            # Print top picks side by side
            print(f"\n{'Rank':<6} {'MeanVariance':<12} {'Weight':>7} {'Period':>8}  {'│':^3}  "
                  f"{'QPU':<12} {'Weight':>7} {'Period':>8}")
            print("-"*70)

            for j in range(5):
                mv_ticker = mv_top5.index[j] if j < len(mv_top5) else ""
                mv_weight = mv_top5.iloc[j] if j < len(mv_top5) else 0
                mv_asset_ret = asset_period_returns.get(mv_ticker, 0) if mv_ticker else 0

                qpu_ticker = qpu_top5.index[j] if j < len(qpu_top5) else ""
                qpu_weight = qpu_top5.iloc[j] if j < len(qpu_top5) else 0
                qpu_asset_ret = asset_period_returns.get(qpu_ticker, 0) if qpu_ticker else 0

                print(f"{j+1:<6} {mv_ticker:<12} {mv_weight:>6.2%} {mv_asset_ret:>+7.2%}  {'│':^3}  "
                      f"{qpu_ticker:<12} {qpu_weight:>6.2%} {qpu_asset_ret:>+7.2%}")

            # Show portfolio statistics
            mv_nonzero = (mv_weights > 1e-6).sum()
            qpu_nonzero = (qpu_weights > 1e-6).sum()
            print(f"\nHoldings: MV={mv_nonzero}, QPU={qpu_nonzero}  │  "
                  f"Concentration (std): MV={mv_weights.std():.4f}, QPU={qpu_weights.std():.4f}")

        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY TABLE - ALL PERIODS")
        print("="*80)
        print(f"\n{'Period End':<12} {'MV Return':>12} {'QPU Return':>12} {'Difference':>12}")
        print("-"*50)

        for date in mv_period_returns.index[-n_periods:]:
            if date in qpu_period_returns.index:
                mv_ret = mv_period_returns[date]
                qpu_ret = qpu_period_returns[date]
                diff = qpu_ret - mv_ret
                print(f"{str(date.date()):<12} {mv_ret:>11.2%} {qpu_ret:>11.2%} {diff:>+11.2%}")

        print("\n" + "="*80)
        print("CUMULATIVE RESULTS")
        print("="*80)

        mv_total = mv_values.iloc[-1] / mv_values.iloc[0] - 1
        qpu_total = qpu_values.iloc[-1] / qpu_values.iloc[0] - 1

        mv_sharpe = mv_period_returns.mean() / mv_period_returns.std() * np.sqrt(252/test_days)
        qpu_sharpe = qpu_period_returns.mean() / qpu_period_returns.std() * np.sqrt(252/test_days)

        print(f"\n{'Metric':<25} {'MeanVariance':>15} {'QPU':>15}")
        print("-"*60)
        print(f"{'Total Return':<25} {mv_total:>14.2%} {qpu_total:>14.2%}")
        print(f"{'Sharpe Ratio':<25} {mv_sharpe:>15.3f} {qpu_sharpe:>15.3f}")
        print(f"{'Avg Period Return':<25} {mv_period_returns.mean():>14.2%} {qpu_period_returns.mean():>14.2%}")
        print(f"{'Period Volatility':<25} {mv_period_returns.std():>14.2%} {qpu_period_returns.std():>14.2%}")

        return {
            'mean_variance': mv_results,
            'qpu': qpu_results
        }


def main():
    """Run comparison analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Detailed optimizer comparison')
    parser.add_argument('--portfolio', type=str, default='portfolio.csv',
                       help='Portfolio CSV file')
    parser.add_argument('--mode', type=str, choices=['single', 'rolling'], default='single',
                       help='Comparison mode')
    parser.add_argument('--solver', type=str, choices=['simulated', 'qpu'], default='simulated',
                       help='QPU solver type')
    parser.add_argument('--clusterer', type=str, choices=['uniform', 'correlation'], default='uniform',
                       help='Clustering method')
    parser.add_argument('--n-levels', type=int, default=6,
                       help='Number of discrete levels')
    parser.add_argument('--max-cluster-size', type=int, default=16,
                       help='Maximum cluster size')
    parser.add_argument('--alpha', type=float, default=10.0,
                       help='Return coefficient')
    parser.add_argument('--beta', type=float, default=2.0,
                       help='Risk coefficient')
    parser.add_argument('--l1-sparsity-penalty', type=float, default=3.0,
                       help='L1 sparsity penalty (λ > 0 promotes fewer holdings, try 0.01-1.0)')
    parser.add_argument('--use-thermometer-cutoff', action='store_true',
                       help='Keep only assets with maximum thermometer level')
    parser.add_argument('--train-days', type=int, default=252,
                       help='Training window size')
    parser.add_argument('--test-days', type=int, default=21,
                       help='Test period size')
    parser.add_argument('--n-periods', type=int, default=12,
                       help='Number of periods for rolling backtest')
    parser.add_argument('--output', type=str, default=None,
                       help='Save output to file (optional)')

    args = parser.parse_args()

    # Redirect output to file if specified
    if args.output:
        import sys
        output_file = open(args.output, 'w')
        sys.stdout = output_file

    # Load data
    portfolio_path = project_root / args.portfolio
    if not portfolio_path.exists():
        print(f"❌ Portfolio file not found: {portfolio_path}")
        return 1

    prices, returns = load_portfolio_data(str(portfolio_path))

    # Create comparison framework
    comparison = DetailedOptimizerComparison(prices, returns)

    # QPU parameters
    qpu_params = {
        'n_levels': args.n_levels,
        'max_cluster_size': args.max_cluster_size,
        'alpha': args.alpha,
        'beta': args.beta,
        'l1_sparsity_penalty': args.l1_sparsity_penalty,
        'use_thermometer_cutoff': args.use_thermometer_cutoff
    }

    # Run comparison
    if args.mode == 'single':
        comparison.run_single_period(
            train_days=args.train_days,
            test_days=args.test_days,
            qpu_solver=args.solver,
            qpu_clusterer=args.clusterer,
            qpu_params=qpu_params
        )
    else:
        comparison.run_rolling_backtest(
            train_days=args.train_days,
            test_days=args.test_days,
            n_periods=args.n_periods,
            qpu_solver=args.solver,
            qpu_params=qpu_params
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
