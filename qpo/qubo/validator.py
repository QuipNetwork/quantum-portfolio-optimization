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

"""Validator for clustering outputs to ensure QPU API compatibility."""

from typing import Dict, List, Any
import pandas as pd


def validate_clustering_for_qpu(
    clusters: Dict[str, List[str]],
    returns: pd.DataFrame,
    max_cluster_size: int,
    max_clusters: int = None,
    min_cluster_size: int = 2
) -> Dict[str, Any]:
    """
    Validate clustering output conforms to QPU API expectations.

    Args:
        clusters: Dictionary mapping cluster IDs to lists of ticker symbols
        returns: Returns DataFrame (T × N) - used to verify tickers exist
        max_cluster_size: Maximum assets per cluster (hardware constraint)
        max_clusters: Maximum number of clusters (template constraint)
        min_cluster_size: Minimum assets per cluster (efficiency constraint)

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'stats': Dict[str, Any]
        }

    Raises:
        ValueError: If critical validation failures occur
    """
    errors = []
    warnings = []
    stats = {}

    # 1. Type validation
    if not isinstance(clusters, dict):
        errors.append(f"Clusters must be a dict, got {type(clusters).__name__}")
        return {'valid': False, 'errors': errors, 'warnings': warnings, 'stats': stats}

    # 2. Check clusters is not empty
    if len(clusters) == 0:
        errors.append("Clusters dictionary is empty")

    # 3. Validate cluster structure
    all_tickers = []
    cluster_sizes = []

    for cluster_id, tickers in clusters.items():
        # Check cluster_id is string
        if not isinstance(cluster_id, str):
            errors.append(f"Cluster ID must be string, got {type(cluster_id).__name__}: {cluster_id}")

        # Check tickers is list
        if not isinstance(tickers, list):
            errors.append(f"Cluster '{cluster_id}' tickers must be list, got {type(tickers).__name__}")
            continue

        # Check list contains strings
        for ticker in tickers:
            if not isinstance(ticker, str):
                errors.append(f"Cluster '{cluster_id}' contains non-string ticker: {type(ticker).__name__}")
                break

        cluster_sizes.append(len(tickers))
        all_tickers.extend(tickers)

    # 4. Check for duplicate tickers across clusters
    seen_tickers = set()
    duplicate_tickers = set()
    for ticker in all_tickers:
        if ticker in seen_tickers:
            duplicate_tickers.add(ticker)
        seen_tickers.add(ticker)

    if duplicate_tickers:
        errors.append(f"Tickers appear in multiple clusters: {sorted(duplicate_tickers)[:5]}...")

    # 5. Check all tickers exist in returns DataFrame
    returns_tickers = set(returns.columns)
    missing_tickers = seen_tickers - returns_tickers
    if missing_tickers:
        errors.append(f"Tickers not in returns DataFrame: {sorted(missing_tickers)[:5]}...")

    # 6. Check all returns tickers are clustered
    unclustered_tickers = returns_tickers - seen_tickers
    if unclustered_tickers:
        warnings.append(f"{len(unclustered_tickers)} tickers not assigned to any cluster: {sorted(unclustered_tickers)[:5]}...")

    # 7. Validate size constraints
    if cluster_sizes:
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)
        avg_size = sum(cluster_sizes) / len(cluster_sizes)

        stats['n_clusters'] = len(clusters)
        stats['n_tickers'] = len(all_tickers)
        stats['min_cluster_size'] = min_size
        stats['max_cluster_size'] = max_size
        stats['avg_cluster_size'] = avg_size
        stats['cluster_sizes'] = sorted(cluster_sizes)

        # Check max_cluster_size constraint
        if max_size > max_cluster_size:
            violating_clusters = [cid for cid, tickers in clusters.items() if len(tickers) > max_cluster_size]
            errors.append(
                f"max_cluster_size constraint violated: {len(violating_clusters)} clusters exceed {max_cluster_size} "
                f"(max observed: {max_size}). Violating clusters: {violating_clusters[:5]}..."
            )

        # Check min_cluster_size constraint
        if min_size < min_cluster_size:
            small_clusters = [cid for cid, tickers in clusters.items() if len(tickers) < min_cluster_size]
            errors.append(
                f"min_cluster_size constraint violated: {len(small_clusters)} clusters below {min_cluster_size} "
                f"(min observed: {min_size}). Small clusters: {small_clusters[:5]}..."
            )

        # Check max_clusters constraint
        if max_clusters is not None and len(clusters) > max_clusters:
            errors.append(
                f"max_clusters constraint violated: {len(clusters)} clusters created, "
                f"but max_clusters={max_clusters}"
            )

        # Warn about very unbalanced clusters
        if max_size > 2 * avg_size:
            warnings.append(
                f"Highly unbalanced cluster sizes detected: max={max_size}, avg={avg_size:.1f}. "
                f"This may lead to inefficient QPU usage."
            )

    # 8. Check for empty clusters
    empty_clusters = [cid for cid, tickers in clusters.items() if len(tickers) == 0]
    if empty_clusters:
        errors.append(f"Empty clusters detected: {empty_clusters}")

    # Determine validity
    valid = len(errors) == 0

    return {
        'valid': valid,
        'errors': errors,
        'warnings': warnings,
        'stats': stats
    }


def print_validation_report(validation_result: Dict[str, Any]):
    """Pretty print validation results."""
    print("="*70)
    print("CLUSTERING VALIDATION REPORT")
    print("="*70)

    stats = validation_result.get('stats', {})
    if stats:
        print("\nStatistics:")
        print(f"  Clusters: {stats.get('n_clusters', 'N/A')}")
        print(f"  Tickers: {stats.get('n_tickers', 'N/A')}")
        print(f"  Cluster sizes: min={stats.get('min_cluster_size', 'N/A')}, "
              f"max={stats.get('max_cluster_size', 'N/A')}, "
              f"avg={stats.get('avg_cluster_size', 0):.1f}")
        print(f"  Distribution: {stats.get('cluster_sizes', [])}")

    errors = validation_result.get('errors', [])
    if errors:
        print(f"\n✗ ERRORS ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")

    warnings = validation_result.get('warnings', [])
    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nResult: {'✓ VALID' if validation_result['valid'] else '✗ INVALID'}")
    print("="*70)
