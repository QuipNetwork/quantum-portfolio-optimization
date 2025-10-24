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

"""Visualization tools for portfolio optimization comparison."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10


def plot_portfolio_values(backtest_results: Dict[str, Dict[str, Any]],
                          output_path: Optional[Path] = None,
                          title: str = "Portfolio Value Over Time"):
    """
    Plot portfolio values over time for all optimizers.

    Args:
        backtest_results: {optimizer_name: backtest_result}
        output_path: Path to save plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(14, 8))

    for name, result in backtest_results.items():
        portfolio_values = result['portfolio_values']
        # Normalize to start at 100
        normalized = (portfolio_values / portfolio_values.iloc[0]) * 100
        plt.plot(normalized.index, normalized.values, label=name, linewidth=2)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (Starting = 100)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_runtime_comparison(backtest_results: Dict[str, Dict[str, Any]],
                            output_path: Optional[Path] = None,
                            use_solver_only: bool = False,
                            title: str = "Runtime Comparison"):
    """
    Plot average runtime per rebalancing for all optimizers.

    Args:
        backtest_results: {optimizer_name: backtest_result}
        output_path: Path to save plot (optional)
        use_solver_only: If True, use solver_only_runtime for quantum methods
        title: Plot title
    """
    names = []
    runtimes = []
    colors = []

    for name, result in backtest_results.items():
        names.append(name)

        # Extract runtime
        if use_solver_only and 'solver_only_runtime' in result['metrics']:
            # Use solver-only time for quantum methods
            runtime = result['metrics'].get('avg_solver_only_runtime',
                                          result['metrics']['avg_runtime'])
        else:
            runtime = result['metrics']['avg_runtime']

        runtimes.append(runtime)

        # Color by type
        if 'Quantum' in name or 'QPU' in name:
            colors.append('#e74c3c')  # Red for quantum
        elif 'Classical' in name or 'Mean-Variance' in name:
            colors.append('#3498db')  # Blue for classical
        elif 'Risk Parity' in name:
            colors.append('#2ecc71')  # Green
        elif 'Equal' in name:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#95a5a6')  # Gray

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(names)), runtimes, color=colors, alpha=0.8, edgecolor='black')

    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Average Runtime per Rebalancing (seconds)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, runtime) in enumerate(zip(bars, runtimes)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{runtime:.4f}s',
                ha='center', va='bottom', fontsize=9)

    # Add speedup annotations relative to slowest
    max_runtime = max(runtimes)
    slowest_idx = runtimes.index(max_runtime)

    for i, runtime in enumerate(runtimes):
        if i != slowest_idx:
            speedup = max_runtime / runtime
            plt.text(i, runtime * 0.5, f'{speedup:.1f}x',
                    ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_weights_heatmap(backtest_result: Dict[str, Any],
                         optimizer_name: str,
                         output_path: Optional[Path] = None,
                         top_n_assets: int = 20):
    """
    Plot weight allocation over time as a heatmap.

    Args:
        backtest_result: Single backtest result
        optimizer_name: Name of optimizer (for title)
        output_path: Path to save plot (optional)
        top_n_assets: Show only top N most traded assets
    """
    weights_history = backtest_result['weights_history']

    if not weights_history:
        print(f"  ⚠ No weight history for {optimizer_name}")
        return

    # Extract dates and weights
    dates = [w['date'] for w in weights_history]
    all_assets = set()
    for w in weights_history:
        all_assets.update(w['weights'].keys())

    all_assets = sorted(all_assets)

    # Build weight matrix
    weight_matrix = np.zeros((len(weights_history), len(all_assets)))
    for i, w in enumerate(weights_history):
        for j, asset in enumerate(all_assets):
            weight_matrix[i, j] = w['weights'].get(asset, 0.0)

    # Find top N most traded assets (by total weight across time)
    total_weights = weight_matrix.sum(axis=0)
    top_indices = np.argsort(total_weights)[-top_n_assets:][::-1]

    # Filter to top assets
    top_assets = [all_assets[i] for i in top_indices]
    top_weights = weight_matrix[:, top_indices]

    # Plot stacked area chart
    plt.figure(figsize=(14, 8))
    plt.stackplot(range(len(dates)), top_weights.T,
                 labels=top_assets, alpha=0.8)

    plt.xlabel('Rebalancing Period', fontsize=12)
    plt.ylabel('Portfolio Weight', fontsize=12)
    plt.title(f'{optimizer_name} - Weight Allocation Over Time (Top {top_n_assets} Assets)',
             fontsize=14, fontweight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

    # Set x-ticks to dates (sample every few periods)
    step = max(1, len(dates) // 10)
    xtick_positions = range(0, len(dates), step)
    xtick_labels = [dates[i].strftime('%Y-%m-%d') for i in xtick_positions]
    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right')

    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_risk_return_scatter(backtest_results: Dict[str, Dict[str, Any]],
                             output_path: Optional[Path] = None,
                             title: str = "Risk-Return Profile"):
    """
    Plot risk-return scatter plot for all optimizers.

    Args:
        backtest_results: {optimizer_name: backtest_result}
        output_path: Path to save plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(12, 8))

    names = []
    returns = []
    volatilities = []
    sharpe_ratios = []

    for name, result in backtest_results.items():
        metrics = result['metrics']
        names.append(name)
        returns.append(metrics['annualized_return'] * 100)  # Convert to percentage
        volatilities.append(metrics['volatility'] * 100)
        sharpe_ratios.append(metrics['sharpe_ratio'])

    # Color by Sharpe ratio
    sharpe_array = np.array(sharpe_ratios)
    scatter = plt.scatter(volatilities, returns, c=sharpe_array,
                         s=200, alpha=0.7, cmap='RdYlGn',
                         edgecolors='black', linewidth=1.5)

    # Add labels
    for i, name in enumerate(names):
        plt.annotate(name, (volatilities[i], returns[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                         facecolor='white', alpha=0.7))

    # Add Sharpe ratio reference lines (diagonal lines from origin)
    max_vol = max(volatilities) * 1.1
    for sharpe in [0.5, 1.0, 1.5, 2.0]:
        x = np.linspace(0, max_vol, 100)
        y = sharpe * x
        plt.plot(x, y, '--', alpha=0.3, color='gray', linewidth=1)
        plt.text(max_vol * 0.95, sharpe * max_vol * 0.95,
                f'Sharpe={sharpe}',
                fontsize=8, alpha=0.5, rotation=np.degrees(np.arctan(sharpe)))

    plt.xlabel('Volatility (Annualized %)', fontsize=12)
    plt.ylabel('Return (Annualized %)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def generate_metrics_table(backtest_results: Dict[str, Dict[str, Any]],
                           output_path: Optional[Path] = None,
                           sort_by: str = 'sharpe_ratio') -> pd.DataFrame:
    """
    Generate performance metrics comparison table.

    Args:
        backtest_results: {optimizer_name: backtest_result}
        output_path: Path to save CSV (optional)
        sort_by: Metric to sort by (default: sharpe_ratio)

    Returns:
        DataFrame with comparison metrics
    """
    rows = []

    for name, result in backtest_results.items():
        metrics = result['metrics']

        # Get solver-specific metrics if available
        solver_runtime = metrics.get('avg_solver_only_runtime', metrics['avg_runtime'])
        qpu_time = metrics.get('avg_qpu_access_time', None)
        network_time = metrics.get('avg_network_latency', None)

        # Calculate other runtime (total - solver - network if available)
        other_runtime = metrics['avg_runtime']
        if qpu_time is not None and network_time is not None:
            other_runtime = other_runtime - qpu_time - network_time

        row = {
            'Optimizer': name,
            'Return (%)': metrics['annualized_return'] * 100,
            'Volatility (%)': metrics['volatility'] * 100,
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Sortino Ratio': metrics['sortino_ratio'],
            'Max Drawdown (%)': metrics['max_drawdown'] * 100,
            'Win Rate (%)': metrics['win_rate'] * 100,
            'Avg Runtime (s)': metrics['avg_runtime'],
            'QPU Time (ms)': f"{qpu_time * 1000:.1f}" if qpu_time is not None else '-',
            'Network (s)': f"{network_time:.3f}" if network_time is not None else '-',
            'Other (s)': f"{other_runtime:.3f}" if qpu_time is not None else '-',
            'Solver Only (s)': solver_runtime if solver_runtime != metrics['avg_runtime'] else '-',
            'N Rebalances': metrics['n_rebalances'],
            'Final Value ($)': metrics['final_value']
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by specified metric
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    elif 'Sharpe Ratio' in df.columns:
        df = df.sort_values(by='Sharpe Ratio', ascending=False)

    if output_path:
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"  ✓ Saved: {output_path}")

    return df


def generate_all_visualizations(backtest_results: Dict[str, Dict[str, Any]],
                                output_dir: Path,
                                generate_individual_weights: bool = True):
    """
    Generate all visualization plots and save to output directory.

    Args:
        backtest_results: {optimizer_name: backtest_result}
        output_dir: Directory to save plots
        generate_individual_weights: Whether to generate per-optimizer weight plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Portfolio value comparison
    print("\n1. Portfolio Value Comparison...")
    plot_portfolio_values(
        backtest_results,
        output_path=output_dir / 'portfolio_value_comparison.png',
        title='Portfolio Value Over Time - All Optimizers'
    )

    # 2. Runtime comparison
    print("\n2. Runtime Comparison...")
    plot_runtime_comparison(
        backtest_results,
        output_path=output_dir / 'runtime_comparison.png',
        use_solver_only=True,
        title='Average Runtime per Rebalancing (Solver Only)'
    )

    plot_runtime_comparison(
        backtest_results,
        output_path=output_dir / 'runtime_total_comparison.png',
        use_solver_only=False,
        title='Average Runtime per Rebalancing (Total)'
    )

    # 3. Weight allocation (individual plots)
    if generate_individual_weights:
        print("\n3. Weight Allocation Heatmaps...")
        for name, result in backtest_results.items():
            safe_name = name.replace(' ', '_').replace('/', '_').lower()
            plot_weights_heatmap(
                result,
                optimizer_name=name,
                output_path=output_dir / f'weights_{safe_name}.png',
                top_n_assets=15
            )

    # 4. Risk-return scatter
    print("\n4. Risk-Return Scatter Plot...")
    plot_risk_return_scatter(
        backtest_results,
        output_path=output_dir / 'risk_return_scatter.png',
        title='Risk-Return Profile - All Optimizers'
    )

    # 5. Metrics table
    print("\n5. Performance Metrics Table...")
    df = generate_metrics_table(
        backtest_results,
        output_path=output_dir / 'comparison_metrics.csv',
        sort_by='Sharpe Ratio'
    )

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    print(f"\n✓ All visualizations saved to: {output_dir}/")
