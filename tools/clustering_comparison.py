"""Clustering method comparison tool.

Compares all available clustering methods on the same dataset and generates
visualizations and performance metrics.

Usage:
    # Basic usage with portfolio price data:
    python tools/clustering_comparison.py --portfolio-csv portfolio.csv

    # With portfolio metadata for sector clustering:
    python tools/clustering_comparison.py --portfolio-csv portfolio.csv --portfolio-info-csv stock_info.csv

    # Complete workflow (fetch info first, then compare):
    qpo fetch-info --input tickers.txt --output stock_info.csv
    python tools/clustering_comparison.py --portfolio-csv portfolio.csv --portfolio-info-csv stock_info.csv

    # With custom constraints:
    python tools/clustering_comparison.py --portfolio-csv portfolio.csv --max-cluster-size 12

    # Generate synthetic data for testing:
    python tools/clustering_comparison.py --seed 42
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from clustering import (
    PortfolioClustering,
    CorrelationClusterer,
    CovarianceClusterer,
    ReturnsClusterer,
    VolatilityClusterer,
    SectorClusterer,
    FactorClusterer,
    DTWClusterer,
    GraphClusterer,
)


def generate_synthetic_returns(n_assets=50, n_days=252, seed=42):
    """Generate synthetic returns with varying characteristics."""
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n_days)
    tickers = [f'ASSET_{i:03d}' for i in range(n_assets)]

    returns_data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0003 + (i / n_assets) * 0.0005
        vol = 0.01 + (i / n_assets) * 0.02
        returns = np.random.normal(drift, vol, n_days)
        returns_data[ticker] = returns

    return pd.DataFrame(returns_data, index=dates)


def compute_cluster_quality_metrics(returns, clusters):
    """Compute quality metrics for a clustering result."""
    metrics = {}

    # Basic statistics
    cluster_sizes = [len(tickers) for tickers in clusters.values()]
    metrics['n_clusters'] = len(clusters)
    metrics['min_size'] = min(cluster_sizes) if cluster_sizes else 0
    metrics['max_size'] = max(cluster_sizes) if cluster_sizes else 0
    metrics['avg_size'] = np.mean(cluster_sizes) if cluster_sizes else 0
    metrics['std_size'] = np.std(cluster_sizes) if cluster_sizes else 0

    # Intra-cluster correlation (higher = more similar within cluster)
    intra_corrs = []
    for cluster_id, tickers in clusters.items():
        if len(tickers) > 1:
            cluster_returns = returns[tickers]
            corr_matrix = cluster_returns.corr()
            # Get upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            intra_corrs.extend(corr_matrix.values[mask])

    metrics['avg_intra_correlation'] = np.mean(intra_corrs) if intra_corrs else 0
    metrics['std_intra_correlation'] = np.std(intra_corrs) if intra_corrs else 0

    # Inter-cluster correlation (lower = more different between clusters)
    inter_corrs = []
    cluster_list = list(clusters.values())
    for i in range(len(cluster_list)):
        for j in range(i+1, len(cluster_list)):
            cluster_i = returns[cluster_list[i]]
            cluster_j = returns[cluster_list[j]]
            # Average returns for each cluster
            avg_i = cluster_i.mean(axis=1)
            avg_j = cluster_j.mean(axis=1)
            inter_corrs.append(avg_i.corr(avg_j))

    metrics['avg_inter_correlation'] = np.mean(inter_corrs) if inter_corrs else 0
    metrics['std_inter_correlation'] = np.std(inter_corrs) if inter_corrs else 0

    # Separation score (higher = better separation)
    if metrics['avg_intra_correlation'] > 0:
        metrics['separation_score'] = (
            metrics['avg_intra_correlation'] - metrics['avg_inter_correlation']
        ) / metrics['avg_intra_correlation']
    else:
        metrics['separation_score'] = 0

    return metrics


def run_clustering_comparison(returns, max_cluster_size=18, target_cluster_size=1, sector_map=None):
    """Run all clustering methods and collect results."""
    results = {}

    methods = {
        'Hierarchical': PortfolioClustering(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size),
        'Correlation': CorrelationClusterer(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size),
        'Covariance': CovarianceClusterer(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size, distance_metric='euclidean'),
        'Covariance (Spectral)': CovarianceClusterer(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size, distance_metric='spectral'),
        'Returns': ReturnsClusterer(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size),
        'Volatility': VolatilityClusterer(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size),
        'Factor (3)': FactorClusterer(n_factors=3, max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size),
        'Factor (5)': FactorClusterer(n_factors=5, max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size),
    }

    # Add optional advanced clustering methods (with error handling)
    try:
        methods['DTW'] = DTWClusterer(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size)
    except ImportError:
        print("  ⚠ DTW clustering skipped (tslearn not available)")

    try:
        methods['Graph'] = GraphClusterer(max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size)
    except ImportError:
        print("  ⚠ Graph clustering skipped (networkx not available)")

    # Add sector clustering if sector data is available
    if sector_map:
        methods['Sector'] = SectorClusterer(sector_map=sector_map, max_cluster_size=max_cluster_size, target_cluster_size=target_cluster_size)

    for name, clusterer in methods.items():
        print(f"Running {name} clustering...")
        try:
            start_time = time.time()

            if name == 'Hierarchical':
                clusters = clusterer.cluster(returns.corr())
            else:
                clusters = clusterer.cluster(returns)

            runtime = time.time() - start_time

            # Compute quality metrics
            metrics = compute_cluster_quality_metrics(returns, clusters)
            metrics['runtime'] = runtime
            metrics['success'] = True
            metrics['clusters'] = clusters

            results[name] = metrics
            print(f"  ✓ Completed in {runtime:.3f}s - {metrics['n_clusters']} clusters")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[name] = {
                'success': False,
                'error': str(e),
                'runtime': 0,
            }

    return results


def plot_comparison(results, output_path=None):
    """Generate comparison visualizations."""
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}

    if not successful_results:
        print("No successful clustering results to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Clustering Method Comparison', fontsize=16, fontweight='bold')

    methods = list(successful_results.keys())

    # 1. Number of clusters
    ax = axes[0, 0]
    n_clusters = [successful_results[m]['n_clusters'] for m in methods]
    ax.bar(methods, n_clusters, color='steelblue')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Cluster Count')
    ax.tick_params(axis='x', rotation=45)

    # 2. Cluster size statistics
    ax = axes[0, 1]
    avg_sizes = [successful_results[m]['avg_size'] for m in methods]
    max_sizes = [successful_results[m]['max_size'] for m in methods]
    min_sizes = [successful_results[m]['min_size'] for m in methods]

    x = np.arange(len(methods))
    width = 0.25
    ax.bar(x - width, min_sizes, width, label='Min', color='lightcoral')
    ax.bar(x, avg_sizes, width, label='Avg', color='steelblue')
    ax.bar(x + width, max_sizes, width, label='Max', color='seagreen')
    ax.set_ylabel('Cluster Size')
    ax.set_title('Cluster Size Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.axhline(y=18, color='red', linestyle='--', linewidth=1, label='Max allowed (18)')

    # 3. Runtime comparison
    ax = axes[0, 2]
    runtimes = [successful_results[m]['runtime'] for m in methods]
    bars = ax.bar(methods, runtimes, color='darkorange')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Computational Performance')
    ax.tick_params(axis='x', rotation=45)

    # Add values on bars
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{runtime:.3f}s', ha='center', va='bottom', fontsize=9)

    # 4. Intra-cluster correlation
    ax = axes[1, 0]
    intra_corrs = [successful_results[m]['avg_intra_correlation'] for m in methods]
    ax.bar(methods, intra_corrs, color='mediumpurple')
    ax.set_ylabel('Average Correlation')
    ax.set_title('Intra-Cluster Correlation (higher = more similar)')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 1])

    # 5. Inter-cluster correlation
    ax = axes[1, 1]
    inter_corrs = [successful_results[m]['avg_inter_correlation'] for m in methods]
    ax.bar(methods, inter_corrs, color='lightseagreen')
    ax.set_ylabel('Average Correlation')
    ax.set_title('Inter-Cluster Correlation (lower = better separation)')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([-1, 1])

    # 6. Separation score
    ax = axes[1, 2]
    sep_scores = [successful_results[m]['separation_score'] for m in methods]
    colors = ['green' if s > 0 else 'red' for s in sep_scores]
    ax.bar(methods, sep_scores, color=colors, alpha=0.7)
    ax.set_ylabel('Separation Score')
    ax.set_title('Cluster Separation Quality (higher = better)')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def plot_cluster_heatmaps(returns, results, output_dir=None):
    """Generate correlation heatmaps for each clustering method."""
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}

    n_methods = len(successful_results)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Correlation Matrices Ordered by Clusters', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    corr_matrix = returns.corr()

    for idx, (name, result) in enumerate(successful_results.items()):
        if idx >= len(axes):
            break

        clusters = result['clusters']

        # Reorder tickers by cluster
        ordered_tickers = []
        cluster_boundaries = [0]
        for cluster_id in sorted(clusters.keys()):
            tickers = clusters[cluster_id]
            ordered_tickers.extend(tickers)
            cluster_boundaries.append(cluster_boundaries[-1] + len(tickers))

        reordered_corr = corr_matrix.loc[ordered_tickers, ordered_tickers]

        ax = axes[idx]
        im = ax.imshow(reordered_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Draw cluster boundaries
        for boundary in cluster_boundaries[1:-1]:
            ax.axhline(y=boundary-0.5, color='black', linewidth=2)
            ax.axvline(x=boundary-0.5, color='black', linewidth=2)

        ax.set_title(f'{name}\n({result["n_clusters"]} clusters)', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(len(successful_results), len(axes)):
        axes[idx].axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04, label='Correlation')

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'cluster_heatmaps.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmaps saved to: {output_path}")
    else:
        plt.show()


def generate_report(returns, results, output_path=None):
    """Generate detailed comparison report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CLUSTERING COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nDataset: {len(returns.columns)} assets, {len(returns)} days")
    report_lines.append(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    report_lines.append("\n" + "=" * 80)

    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    failed_results = {k: v for k, v in results.items() if not v.get('success', False)}

    # Summary table
    report_lines.append("\nSUMMARY TABLE")
    report_lines.append("-" * 88)

    header = f"{'Method':<20} {'Clusters':>10} {'Min':>8} {'Max':>8} {'Avg':>8} {'Runtime':>10} {'Intra-Corr':>12} {'Sep Score':>12}"
    report_lines.append(header)
    report_lines.append("-" * 88)

    for name, result in successful_results.items():
        row = (
            f"{name:<20} "
            f"{result['n_clusters']:>10d} "
            f"{result['min_size']:>8d} "
            f"{result['max_size']:>8d} "
            f"{result['avg_size']:>8.1f} "
            f"{result['runtime']:>9.3f}s "
            f"{result['avg_intra_correlation']:>12.3f} "
            f"{result['separation_score']:>12.3f}"
        )
        report_lines.append(row)

    # Detailed metrics
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DETAILED METRICS")
    report_lines.append("=" * 80)

    for name, result in successful_results.items():
        report_lines.append(f"\n{name}:")
        report_lines.append("-" * 40)
        report_lines.append(f"  Number of clusters: {result['n_clusters']}")
        report_lines.append(f"  Cluster sizes: min={result['min_size']}, max={result['max_size']}, "
                          f"avg={result['avg_size']:.2f}, std={result['std_size']:.2f}")
        report_lines.append(f"  Intra-cluster correlation: {result['avg_intra_correlation']:.4f} "
                          f"± {result['std_intra_correlation']:.4f}")
        report_lines.append(f"  Inter-cluster correlation: {result['avg_inter_correlation']:.4f} "
                          f"± {result['std_inter_correlation']:.4f}")
        report_lines.append(f"  Separation score: {result['separation_score']:.4f}")
        report_lines.append(f"  Runtime: {result['runtime']:.3f}s")

    # Failed methods
    if failed_results:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("FAILED METHODS")
        report_lines.append("=" * 80)
        for name, result in failed_results.items():
            report_lines.append(f"\n{name}: {result.get('error', 'Unknown error')}")

    # Recommendations
    report_lines.append("\n" + "=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)

    if successful_results:
        # Best separation
        best_sep = max(successful_results.items(), key=lambda x: x[1]['separation_score'])
        report_lines.append(f"\n✓ Best cluster separation: {best_sep[0]} "
                          f"(score: {best_sep[1]['separation_score']:.3f})")

        # Fastest
        fastest = min(successful_results.items(), key=lambda x: x[1]['runtime'])
        report_lines.append(f"✓ Fastest method: {fastest[0]} ({fastest[1]['runtime']:.3f}s)")

        # Most balanced sizes
        best_balance = min(successful_results.items(), key=lambda x: x[1]['std_size'])
        report_lines.append(f"✓ Most balanced cluster sizes: {best_balance[0]} "
                          f"(std: {best_balance[1]['std_size']:.2f})")

    report_lines.append("\n" + "=" * 80)

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    print(report)

    return report


def load_portfolio_data(csv_path: str) -> pd.DataFrame:
    """
    Load portfolio price data from CSV and compute returns.

    Args:
        csv_path: Path to CSV file with price data (Date, TICKER1, TICKER2, ...)

    Returns:
        DataFrame of returns (T × N)
    """
    print(f"Loading portfolio data from: {csv_path}")
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    n_days = len(data)
    n_assets = len(data.columns)

    print(f"  Portfolio: {n_assets} assets, {n_days} days of prices")
    print("  Computing returns...")

    returns = data.pct_change().dropna()

    print(f"  Computed {len(returns)} days of returns for {n_assets} assets")

    return returns


def load_portfolio_info(csv_path: str) -> Optional[Dict[str, str]]:
    """
    Load portfolio metadata from CSV file (e.g., from 'qpo fetch-info').

    Args:
        csv_path: Path to CSV file with columns: symbol, sector, industry, etc.

    Returns:
        Dictionary mapping ticker -> sector, or None if file doesn't exist
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Portfolio info file not found: {csv_path}")
        return None

    print(f"Loading portfolio info from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)

        if 'symbol' not in df.columns:
            print("  Warning: CSV must have 'symbol' column. Skipping sector clustering.")
            return None

        if 'sector' not in df.columns:
            print("  Warning: CSV must have 'sector' column. Skipping sector clustering.")
            return None

        # Create sector map
        sector_map = {}
        for _, row in df.iterrows():
            symbol = str(row['symbol']).strip().upper()
            sector = row['sector']

            # Handle NaN or empty values
            if pd.isna(sector) or sector == '':
                sector_map[symbol] = 'Unknown'
            else:
                sector_map[symbol] = str(sector).strip()

        print(f"  Loaded sector data for {len(sector_map)} tickers")
        return sector_map

    except Exception as e:
        print(f"  Error loading portfolio info: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Compare clustering methods on portfolio data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With portfolio price data
  %(prog)s --portfolio-csv portfolio.csv

  # With portfolio metadata for sector clustering
  %(prog)s --portfolio-csv portfolio.csv --portfolio-info-csv stock_info.csv

  # Generate synthetic data for testing
  %(prog)s --seed 42

  # Custom output directory and cluster size
  %(prog)s --portfolio-csv portfolio.csv --max-cluster-size 12 --output-dir results/
        """
    )

    parser.add_argument('--portfolio-csv', type=str,
                       help='Path to CSV file with portfolio price data (Date, TICKER1, TICKER2, ...)')
    parser.add_argument('--portfolio-info-csv', type=str,
                       help='Path to CSV file with portfolio metadata (symbol, sector, industry, ...)')
    parser.add_argument('--max-cluster-size', type=int, default=18,
                       help='Maximum cluster size constraint (default: 18)')
    parser.add_argument('--target-cluster-size', type=int, default=None,
                       help='Target average cluster size (default: max/2). Lower values produce more clusters.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for synthetic data generation (default: 42)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for reports and plots (default: output)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load or generate returns
    if args.portfolio_csv:
        returns = load_portfolio_data(args.portfolio_csv)
    else:
        # Generate synthetic data with default dimensions
        n_assets = 50
        n_days = 252
        print(f"No portfolio CSV provided. Generating synthetic data...")
        print(f"  {n_assets} assets, {n_days} days (use --seed to change random seed)")
        returns = generate_synthetic_returns(n_assets, n_days, args.seed)

    # Load portfolio info if provided, or look for default file
    sector_map = None
    if args.portfolio_info_csv:
        sector_map = load_portfolio_info(args.portfolio_info_csv)
    else:
        # Try to find default portfolio-info.csv in project root
        default_info_path = project_root / 'portfolio-info.csv'
        if default_info_path.exists():
            print(f"\n📊 Found portfolio info file: {default_info_path}")
            sector_map = load_portfolio_info(default_info_path)

    # Run comparison
    print("\nRunning clustering comparison...")
    print("=" * 80)
    results = run_clustering_comparison(returns,
                                       max_cluster_size=args.max_cluster_size,
                                       target_cluster_size=args.target_cluster_size,
                                       sector_map=sector_map)

    # Generate outputs
    print("\nGenerating visualizations and report...")
    plot_comparison(results, output_path=output_dir / 'comparison_metrics.png')
    plot_cluster_heatmaps(returns, results, output_dir=output_dir)
    generate_report(returns, results, output_path=output_dir / 'comparison_report.txt')

    print(f"\nAll outputs saved to: {output_dir}/")
    print("  - comparison_metrics.png")
    print("  - cluster_heatmaps.png")
    print("  - comparison_report.txt")


if __name__ == '__main__':
    main()
