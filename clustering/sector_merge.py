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

"""Sector merging utilities for combining similar sectors into super-sectors.

This module provides utilities to combine GICS sectors when the number of
sector-based clusters exceeds topology constraints.
"""

from typing import Dict, List, Set
from collections import defaultdict


# GICS Sector hierarchy - map of related sectors that can be merged
SECTOR_MERGE_HIERARCHY = {
    # Financial Services super-group
    'Financial': {
        'Financial Services',
        'Financial',
        'Banks',
        'Insurance',
        'Capital Markets',
        'Consumer Finance',
        'Diversified Financial Services',
    },

    # Technology super-group
    'Technology': {
        'Technology',
        'Information Technology',
        'Software',
        'Hardware',
        'Semiconductors',
        'IT Services',
        'Communications Equipment',
        'Electronic Equipment',
    },

    # Healthcare super-group
    'Healthcare': {
        'Healthcare',
        'Health Care',
        'Biotechnology',
        'Pharmaceuticals',
        'Health Care Equipment',
        'Health Care Providers',
        'Life Sciences Tools',
    },

    # Consumer super-group
    'Consumer': {
        'Consumer Cyclical',
        'Consumer Defensive',
        'Consumer Discretionary',
        'Consumer Staples',
        'Retail',
        'Automobiles',
        'Household Products',
        'Personal Products',
        'Textiles',
        'Hotels',
        'Restaurants',
        'Leisure',
    },

    # Industrial super-group
    'Industrial': {
        'Industrials',
        'Industrial',
        'Aerospace & Defense',
        'Construction',
        'Machinery',
        'Airlines',
        'Transportation',
        'Building Products',
        'Electrical Equipment',
    },

    # Energy & Materials super-group
    'Energy & Materials': {
        'Energy',
        'Oil & Gas',
        'Coal',
        'Materials',
        'Basic Materials',
        'Chemicals',
        'Metals & Mining',
        'Paper & Forest Products',
    },

    # Communication & Media super-group
    'Communication': {
        'Communication Services',
        'Communications',
        'Telecommunication Services',
        'Media',
        'Entertainment',
        'Interactive Media',
        'Wireless Telecommunication',
    },

    # Utilities & Real Estate super-group
    'Utilities & Real Estate': {
        'Utilities',
        'Real Estate',
        'Equity Real Estate',
        'Real Estate Management',
        'Electric Utilities',
        'Gas Utilities',
        'Water Utilities',
    },
}


def build_sector_to_supersector_map() -> Dict[str, str]:
    """
    Build a mapping from individual sectors to their super-sector.

    Returns:
        Dict mapping sector name to super-sector name
    """
    sector_map = {}
    for supersector, sectors in SECTOR_MERGE_HIERARCHY.items():
        for sector in sectors:
            sector_map[sector] = supersector

    return sector_map


def merge_sectors_by_hierarchy(sector_clusters: Dict[str, List[str]],
                               max_clusters: int = None) -> Dict[str, List[str]]:
    """
    Merge sector-based clusters using the sector hierarchy.

    Combines related sectors (e.g., all financial sectors) when the number
    of sector clusters exceeds the maximum allowed.

    Args:
        sector_clusters: Dict mapping sector names to lists of tickers
        max_clusters: Maximum number of clusters allowed (optional)

    Returns:
        Merged cluster dictionary with combined sectors
    """
    if max_clusters is None or len(sector_clusters) <= max_clusters:
        # No merging needed
        return sector_clusters

    # Build reverse mapping
    sector_to_super = build_sector_to_supersector_map()

    # Group clusters by super-sector
    super_clusters = defaultdict(list)
    unmapped_clusters = {}

    for sector, tickers in sector_clusters.items():
        if sector in sector_to_super:
            supersector = sector_to_super[sector]
            super_clusters[supersector].extend(tickers)
        else:
            # Sector not in hierarchy - keep separate for now
            unmapped_clusters[sector] = tickers

    # Convert to regular dict
    merged = {name: tickers for name, tickers in super_clusters.items()}
    merged.update(unmapped_clusters)

    return merged


def merge_smallest_sectors(sector_clusters: Dict[str, List[str]],
                          target_clusters: int) -> Dict[str, List[str]]:
    """
    Merge sectors by combining the smallest clusters.

    Simple greedy algorithm: repeatedly merge the two smallest clusters
    until we reach the target number.

    Args:
        sector_clusters: Dict mapping sector names to lists of tickers
        target_clusters: Target number of clusters

    Returns:
        Merged cluster dictionary
    """
    if len(sector_clusters) <= target_clusters:
        return sector_clusters

    # Convert to list of (name, tickers) for easier manipulation
    clusters = [(name, list(tickers)) for name, tickers in sector_clusters.items()]

    while len(clusters) > target_clusters:
        # Sort by size
        clusters.sort(key=lambda x: len(x[1]))

        # Merge two smallest
        name1, tickers1 = clusters.pop(0)
        name2, tickers2 = clusters.pop(0)

        # Create merged cluster
        merged_name = f"{name1}+{name2}"
        merged_tickers = tickers1 + tickers2

        clusters.append((merged_name, merged_tickers))

    return {name: tickers for name, tickers in clusters}


def smart_sector_merge(sector_clusters: Dict[str, List[str]],
                      max_clusters: int,
                      max_cluster_size: int) -> Dict[str, List[str]]:
    """
    Smart sector merging that considers both cluster count and size constraints.

    Strategy:
    1. First try hierarchy-based merging (keeps related sectors together)
    2. If still too many clusters, use greedy smallest-first merging
    3. Validate max_cluster_size constraint

    Args:
        sector_clusters: Dict mapping sector names to lists of tickers
        max_clusters: Maximum number of clusters
        max_cluster_size: Maximum assets per cluster

    Returns:
        Merged cluster dictionary

    Raises:
        ValueError: If constraints cannot be satisfied
    """
    # Step 1: Try hierarchy-based merging
    merged = merge_sectors_by_hierarchy(sector_clusters, max_clusters)

    # Step 2: If still too many, use greedy merging
    if len(merged) > max_clusters:
        merged = merge_smallest_sectors(merged, max_clusters)

    # Step 3: Validate size constraints (skip if max_cluster_size is infinite)
    if max_cluster_size != float('inf'):
        for cluster_name, tickers in merged.items():
            if len(tickers) > max_cluster_size:
                raise ValueError(
                    f"Cluster '{cluster_name}' has {len(tickers)} assets, "
                    f"exceeds max_cluster_size={max_cluster_size}. "
                    f"Cannot merge sectors without violating size constraint."
                )

    return merged


def get_sector_cluster_stats(sector_clusters: Dict[str, List[str]]) -> Dict:
    """
    Get statistics about sector clustering.

    Args:
        sector_clusters: Dict mapping sector names to lists of tickers

    Returns:
        Statistics dictionary
    """
    sizes = [len(tickers) for tickers in sector_clusters.values()]

    return {
        'num_sectors': len(sector_clusters),
        'total_assets': sum(sizes),
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'avg_size': sum(sizes) / len(sizes) if sizes else 0,
        'sectors': list(sector_clusters.keys()),
    }
