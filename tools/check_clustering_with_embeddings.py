#!/usr/bin/env python3
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

"""Test all clustering methods against all available QPU topologies.

This script tests each clustering method to see if it can fit into each
available hardware topology configuration. Produces a compatibility matrix
showing which clustering methods work with which topology constraints.

Usage:
    python tools/test_clustering_topology_matrix.py --portfolio-csv portfolio.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from clustering import (
    CorrelationClusterer,
    AntiCorrelationClusterer,
    CovarianceClusterer,
    ReturnsClusterer,
    VolatilityClusterer,
    FactorClusterer,
    DTWClusterer,
    GraphClusterer,
    UniformClusterer,
    SectorClusterer,
)


# Define available topologies from embeddings/templates/
# Extracted from actual embedding files in embeddings/templates/
# Format: (num_clusters, total_assets, levels, description)
# Note: "12c × 12a" means 12 clusters with max 12 assets per cluster
AVAILABLE_TOPOLOGIES = [
    (12, 144, 8, "12c × 12a (8 levels)"),
    (12, 192, 6, "12c × 16a (6 levels)"),
    (13, 117, 10, "13c × 9a (10 levels)"),
    (13, 117, 4, "13c × 9a (4 levels)"),
    (13, 117, 8, "13c × 9a (8 levels)"),
    (14, 196, 6, "14c × 14a (6 levels)"),
    (16, 128, 6, "16c × 8a (6 levels)"),
    (17, 289, 4, "17c × 17a (4 levels)"),
]


def load_portfolio_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load portfolio price data and compute returns."""
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return prices, returns


def test_clustering_method(
    method_name: str,
    method_class,
    returns: pd.DataFrame,
    num_clusters: int,
    max_cluster_size: int,
    verbose: bool = False
) -> Tuple[bool, str, Dict]:
    """
    Test if a clustering method can fit into a topology.

    Returns:
        (success, message, stats) tuple
    """
    n_assets = len(returns.columns)
    target_size = max(1, n_assets // num_clusters)

    try:
        # Create clusterer with topology constraints
        clusterer = method_class(
            max_cluster_size=max_cluster_size,
            max_clusters=num_clusters,
            target_cluster_size=target_size
        )

        # Run clustering (sweep is enabled by default)
        clusters = clusterer.cluster(returns)

        # Check results
        actual_num = len(clusters)
        cluster_sizes = [len(tickers) for tickers in clusters.values()]
        max_size = max(cluster_sizes)

        # Validate constraints
        if actual_num > num_clusters:
            return False, f"Too many clusters: {actual_num} > {num_clusters}", {}

        if max_size > max_cluster_size:
            return False, f"Cluster too large: {max_size} > {max_cluster_size}", {}

        # Calculate quality metrics
        intra_corrs = []
        for cluster_id, tickers in clusters.items():
            if len(tickers) > 1:
                cluster_returns = returns[tickers]
                corr_matrix = cluster_returns.corr()
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                intra_corrs.extend(corr_matrix.values[mask])

        avg_intra_corr = np.mean(intra_corrs) if intra_corrs else np.nan

        stats = {
            'num_clusters': actual_num,
            'avg_size': np.mean(cluster_sizes),
            'max_size': max_size,
            'avg_correlation': avg_intra_corr,
        }

        return True, f"{actual_num} clusters, avg={stats['avg_size']:.1f}", stats

    except Exception as e:
        return False, f"Error: {str(e)[:50]}", {}


def print_compatibility_matrix(results: Dict[str, Dict[Tuple[int, int], Tuple[bool, str]]]):
    """
    Print a compatibility matrix showing which methods work with which topologies.

    Args:
        results: {method_name: {(num_clusters, max_size): (success, message), ...}}
    """
    print(f"\n{'='*100}")
    print(f"CLUSTERING METHOD × TOPOLOGY COMPATIBILITY MATRIX")
    print(f"{'='*100}\n")

    # Get all topologies
    topologies = sorted(set(
        topo for method_results in results.values()
        for topo in method_results.keys()
    ))

    # Print header
    method_col_width = 20
    topo_col_width = 16

    header = f"{'Method':<{method_col_width}}"
    for num_c, total_a, levels, desc in AVAILABLE_TOPOLOGIES:
        # Extract the compact name from description (e.g., "12c × 12a")
        compact_name = desc.split('(')[0].strip().replace(' ', '')
        header += compact_name.center(topo_col_width)
    print(header)
    print("-" * len(header))

    # Print each method's results
    for method_name in sorted(results.keys()):
        row = f"{method_name:<{method_col_width}}"

        for num_c, total_assets, levels, desc in AVAILABLE_TOPOLOGIES:
            # Calculate max cluster size from topology
            max_s = total_assets // num_c

            # Get result for this topology
            success, message, _ = results[method_name].get((num_c, max_s), (False, "N/A", {}))

            # Format result
            if success:
                symbol = "✓"
                # Extract cluster count from message if available
                if "clusters" in message:
                    try:
                        n_clusters = message.split()[0]
                        display = f"{symbol} ({n_clusters})"
                    except:
                        display = f"{symbol}"
                else:
                    display = f"{symbol}"
            else:
                symbol = "✗"
                display = f"{symbol}"

            row += display.center(topo_col_width)

        print(row)

    print("\n" + "="*len(header))
    print("\nLegend:")
    print("  ✓ = Method fits topology (sweep enabled by default)")
    print("  ✗ = Method cannot fit topology")
    print("  Number in parentheses = actual clusters generated")
    print("\nTopology Details:")
    for num_c, total_assets, levels, desc in AVAILABLE_TOPOLOGIES:
        max_per_cluster = total_assets // num_c
        print(f"  {desc}: {num_c} clusters, up to {max_per_cluster} assets/cluster")


def print_detailed_results(results: Dict[str, Dict[Tuple[int, int], Tuple[bool, str, Dict]]]):
    """Print detailed statistics for each successful configuration."""
    print(f"\n{'='*100}")
    print(f"DETAILED RESULTS")
    print(f"{'='*100}\n")

    for method_name in sorted(results.keys()):
        print(f"\n{method_name}:")
        print(f"{'-'*100}")

        for (num_c, max_s), (success, message, stats) in sorted(results[method_name].items()):
            if success and stats:
                print(f"  Topology: {num_c} clusters × {max_s} max size")
                print(f"    Result: {message}")
                print(f"    Avg correlation: {stats['avg_correlation']:.3f}")
            elif success:
                print(f"  Topology: {num_c} clusters × {max_s} max size → ✓ {message}")


def main():
    parser = argparse.ArgumentParser(
        description='Test all clustering methods against all QPU topologies',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--portfolio-csv', type=str, required=True,
                       help='Path to CSV file with portfolio price data')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output for each test')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed statistics')

    args = parser.parse_args()

    # Load data
    print(f"Loading portfolio data from: {args.portfolio_csv}")
    prices, returns = load_portfolio_data(args.portfolio_csv)
    n_assets = len(returns.columns)
    n_days = len(returns)

    print(f"  Assets: {n_assets}")
    print(f"  Trading days: {n_days}")
    print(f"  Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    # Define clustering methods to test
    method_map = {
        'Correlation': CorrelationClusterer,
        'AntiCorrelation': AntiCorrelationClusterer,
        'Covariance': CovarianceClusterer,
        'Returns': ReturnsClusterer,
        'Volatility': VolatilityClusterer,
        'Factor': FactorClusterer,
        'DTW': DTWClusterer,
        'Graph': GraphClusterer,
        'Uniform': UniformClusterer,
        # 'Sector': SectorClusterer,  # Excluded - too slow without pre-cached data
    }

    # Test all methods against all topologies
    results = {}
    total_tests = len(method_map) * len(AVAILABLE_TOPOLOGIES)
    test_count = 0

    print(f"\nTesting {len(method_map)} clustering methods against {len(AVAILABLE_TOPOLOGIES)} topologies...")
    print(f"Total tests: {total_tests}\n")

    for method_name, method_class in method_map.items():
        results[method_name] = {}

        for num_clusters, total_capacity, levels, desc in AVAILABLE_TOPOLOGIES:
            test_count += 1

            # Check if topology can fit our assets
            if total_capacity < n_assets:
                if args.verbose:
                    print(f"[{test_count}/{total_tests}] {method_name} × {desc}: SKIP (capacity too small)")
                results[method_name][(num_clusters, total_capacity // num_clusters)] = (
                    False, "Topology too small", {}
                )
                continue

            # Calculate max cluster size from topology
            max_cluster_size = total_capacity // num_clusters

            if args.verbose:
                print(f"[{test_count}/{total_tests}] Testing {method_name} × {desc}...", end=" ")

            success, message, stats = test_clustering_method(
                method_name,
                method_class,
                returns,
                num_clusters,
                max_cluster_size,
                verbose=args.verbose
            )

            results[method_name][(num_clusters, max_cluster_size)] = (success, message, stats)

            if args.verbose:
                print(f"{'✓' if success else '✗'} {message}")

    # Print compatibility matrix
    print_compatibility_matrix(results)

    # Print detailed results if requested
    if args.detailed:
        print_detailed_results(results)

    # Summary statistics
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")

    total_success = sum(
        1 for method_results in results.values()
        for success, _, _ in method_results.values()
        if success
    )

    print(f"\nTotal successful configurations: {total_success}/{total_tests}")
    print(f"Success rate: {total_success/total_tests*100:.1f}%")

    # Per-method success rate
    print("\nPer-method success:")
    for method_name in sorted(results.keys()):
        method_success = sum(1 for success, _, _ in results[method_name].values() if success)
        method_total = len(results[method_name])
        print(f"  {method_name:<20} {method_success}/{method_total} topologies")

    # Per-topology success rate
    print("\nPer-topology success:")
    for num_c, total_assets, levels, desc in AVAILABLE_TOPOLOGIES:
        max_s = total_assets // num_c
        topo_success = sum(
            1 for method_results in results.values()
            if (num_c, max_s) in method_results and method_results[(num_c, max_s)][0]
        )
        topo_total = len(results)
        print(f"  {desc:<35} {topo_success}/{topo_total} methods")

    print(f"\n{'='*100}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
