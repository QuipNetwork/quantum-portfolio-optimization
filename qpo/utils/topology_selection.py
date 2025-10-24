#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Automatic topology selection for DiscreteLevelsOptimizer.

This module provides utilities to automatically select the best embedding template
based on portfolio size and available precomputed templates.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import re


def select_optimal_template(
    num_assets: int,
    template_dir: Optional[Path] = None,
    n_levels: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Automatically select the best embedding template for a given portfolio size.

    Scans the embeddings/templates/ directory for files matching the pattern
    portfolio_<cluster_size>c_<assets_per_cluster>a_<num_levels>l and selects
    the template with maximum num_levels among those with sufficient capacity.

    Template filename format: portfolio_<C>c_<A>a_<L>l.json
    - C: number of asset clusters (excludes meta-cluster)
    - A: assets per cluster
    - L: number of discrete levels

    Selection algorithm:
    1. Parse all available templates
    2. Calculate capacity = cluster_size × assets_per_cluster
    3. Filter to templates where capacity ≥ num_assets
    4. Among filtered templates, select the one with maximum num_levels
    5. If multiple templates have same max num_levels, prefer minimum capacity
       (least waste)

    Args:
        num_assets: Number of assets in the portfolio
        template_dir: Directory containing templates (default: embeddings/templates/)
        n_levels: If specified, only consider templates with this num_levels

    Returns:
        Dict with template parameters or None if no suitable template found:
        {
            'template_name': str,          # Filename of selected template
            'cluster_size': int,           # Number of asset clusters (C)
            'assets_per_cluster': int,     # Assets per cluster (A)
            'num_levels': int,             # Number of discrete levels (L)
            'capacity': int,               # Total capacity (C × A)
            'waste': int,                  # Excess capacity (capacity - num_assets)
        }

    Example:
        >>> # For a 50-asset portfolio
        >>> params = select_optimal_template(50)
        >>> # Might return: {'cluster_size': 10, 'assets_per_cluster': 5,
        >>>                  'num_levels': 10, 'capacity': 50, 'waste': 0}
    """
    # Default template directory
    if template_dir is None:
        template_dir = Path(__file__).parent.parent.parent / 'embeddings' / 'templates'

    template_dir = Path(template_dir)
    if not template_dir.exists():
        return None

    # Scan for template files matching pattern: portfolio_*c_*a_*l*.json
    pattern = re.compile(r'portfolio_(\d+)c_(\d+)a_(\d+)l.*\.json$')

    candidates = []
    for template_file in template_dir.glob('portfolio_*.json'):
        match = pattern.match(template_file.name)
        if match:
            cluster_size = int(match.group(1))
            assets_per_cluster = int(match.group(2))
            num_levels_template = int(match.group(3))

            # Skip if n_levels specified and doesn't match
            if n_levels is not None and num_levels_template != n_levels:
                continue

            capacity = cluster_size * assets_per_cluster
            waste = capacity - num_assets

            # Only consider templates with sufficient capacity
            if capacity >= num_assets:
                candidates.append({
                    'template_name': template_file.name,
                    'cluster_size': cluster_size,
                    'assets_per_cluster': assets_per_cluster,
                    'num_levels': num_levels_template,
                    'capacity': capacity,
                    'waste': waste,
                })

    if not candidates:
        return None

    # Select template with maximum num_levels, breaking ties by minimum waste
    # Sort by: (1) num_levels descending, (2) waste ascending
    candidates.sort(key=lambda x: (-x['num_levels'], x['waste']))

    return candidates[0]


def get_template_parameters_for_optimizer(
    num_assets: int,
    template_dir: Optional[Path] = None,
    n_levels: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get template parameters formatted for DiscreteLevelsOptimizer initialization.

    This is a convenience wrapper around select_optimal_template that formats
    the result for direct use in DiscreteLevelsOptimizer constructor.

    Args:
        num_assets: Number of assets in the portfolio
        template_dir: Directory containing templates
        n_levels: If specified, only consider templates with this num_levels

    Returns:
        Dict with parameters for DiscreteLevelsOptimizer:
        {
            'n_levels': int,
            'max_cluster_size': int,
            'expected_assets_per_cluster': int,
            'expected_n_clusters': int,
            'auto_select_template': False,  # We already selected manually
        }

    Raises:
        ValueError: If no suitable template found

    Example:
        >>> params = get_template_parameters_for_optimizer(50, n_levels=6)
        >>> optimizer = DiscreteLevelsOptimizer(**params, alpha=10, beta=2)
    """
    template = select_optimal_template(num_assets, template_dir, n_levels)

    if template is None:
        available_templates = []
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent / 'embeddings' / 'templates'

        template_dir = Path(template_dir)
        if template_dir.exists():
            available_templates = list(template_dir.glob('portfolio_*.json'))

        error_msg = (
            f"No suitable template found for {num_assets} assets"
            + (f" with n_levels={n_levels}" if n_levels else "")
            + f"\n\nAvailable templates in {template_dir}:\n"
        )

        if available_templates:
            for t in available_templates[:10]:  # Show first 10
                error_msg += f"  - {t.name}\n"
            if len(available_templates) > 10:
                error_msg += f"  ... and {len(available_templates) - 10} more\n"
        else:
            error_msg += "  (none found)\n"
            error_msg += (
                f"\nTo generate templates, run:\n"
                f"  python tools/generate_portfolio_embedding_template.py \\\n"
                f"    --solver Advantage2_system1.6 \\\n"
                f"    --num-clusters <C> \\\n"
                f"    --cluster-size <A> \\\n"
                f"    --n-levels {n_levels or 6}"
            )

        raise ValueError(error_msg)

    return {
        'n_levels': template['num_levels'],
        'max_cluster_size': template['assets_per_cluster'],
        'expected_assets_per_cluster': template['assets_per_cluster'],
        'expected_n_clusters': template['cluster_size'],
        'auto_select_template': False,  # We already did the selection
    }