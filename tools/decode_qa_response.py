#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Decode quantum annealing solver response and display portfolio weights.

This tool takes a solver response (JSON) and the corresponding metadata file
to decode the solution and display portfolio weights and performance metrics.

Example usage:
    # From curl response saved to file
    curl -X POST https://qpu-1.nodes.quip.network/solve \\
        -H "Content-Type: application/json" \\
        -d @train.json > response.json

    python tools/decode_qa_response.py \\
        --response response.json \\
        --metadata train.metadata.json \\
        --portfolio-csv portfolio.csv
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qpo.utils.data_prep import load_portfolio_data


def decode_sample(
    sample: List[int],
    variable_mapping: Dict[str, int],
    reverse_mapping: Dict[str, str],
    n_levels: int,
    cluster_info: List[Dict] = None
) -> Dict[str, float]:
    """
    Decode a sample to portfolio weights.

    Args:
        sample: Sample from solver (Ising spin format: -1/+1, or binary: 0/1)
        variable_mapping: Maps template variable names to indices
        reverse_mapping: Maps template variables back to TICKER_LEVEL format
        n_levels: Number of discrete weight levels
        cluster_info: Cluster information (needed for meta-cluster decoding)

    Returns:
        Dict mapping ticker to weight
    """
    # Convert from Ising spin format (-1/+1) to binary (0/1) if needed
    if -1 in sample or 1 in sample and 0 not in sample:
        # Ising format: -1 = 0, +1 = 1
        binary_sample = [(s + 1) // 2 for s in sample]
    else:
        binary_sample = sample

    # Reverse the variable_mapping to get index -> template_var
    idx_to_template_var = {idx: var for var, idx in variable_mapping.items()}

    # Group by ticker/cluster
    ticker_values = {}
    meta_cluster_values = {}

    for idx, value in enumerate(binary_sample):
        if value == 0:
            continue

        template_var = idx_to_template_var.get(idx)
        if template_var is None:
            continue

        # Map back to TICKER_LEVEL format
        original_var = reverse_mapping.get(template_var)
        if original_var is None:
            continue

        # Parse TICKER_LEVEL or CLUSTER_ID_LEVEL
        parts = original_var.rsplit('_', 1)
        if len(parts) != 2:
            continue

        ticker = parts[0]
        level_str = parts[1]

        if not level_str.isdigit():
            continue

        level = int(level_str)

        # Check if this is a meta-cluster variable
        # Template format: META_X_Y where X is cluster index
        is_meta = False
        if cluster_info and template_var and template_var.startswith('META_'):
            # Parse META_X_Y to get cluster index
            parts = template_var.split('_')
            if len(parts) >= 3:
                try:
                    meta_idx = int(parts[1])
                    # Find the meta-cluster and get the cluster_id at this index
                    for cluster in cluster_info:
                        if cluster['id'] == 'META_CLUSTER':
                            if meta_idx < len(cluster['tickers']):
                                cluster_id = cluster['tickers'][meta_idx]
                                if cluster_id not in meta_cluster_values:
                                    meta_cluster_values[cluster_id] = []
                                meta_cluster_values[cluster_id].append(level)
                                is_meta = True
                            break
                except ValueError:
                    pass

        if not is_meta:
            if ticker not in ticker_values:
                ticker_values[ticker] = []
            ticker_values[ticker].append(level)

    # Convert thermometer encoding to weights
    K = 2**8 - 1  # 255
    start_bit = 8 - n_levels

    # Decode meta-cluster weights (cluster allocations)
    meta_weights = {}
    for cluster_id, levels in meta_cluster_values.items():
        weight = 0.0
        for level in levels:
            weight += 2**(start_bit + level) / K
        meta_weights[cluster_id] = weight

    # Normalize meta-cluster weights
    total_meta = sum(meta_weights.values())
    if total_meta > 0:
        meta_weights = {k: v / total_meta for k, v in meta_weights.items()}

    # Decode asset weights within clusters
    cluster_weights = {}
    for ticker, levels in ticker_values.items():
        if ticker.startswith('_DUMMY_') or ticker.startswith('_EMPTY_'):
            continue

        weight = 0.0
        for level in levels:
            weight += 2**(start_bit + level) / K
        cluster_weights[ticker] = weight

    # Apply meta-cluster allocation to get final weights
    if meta_weights and cluster_info:
        # Find which cluster each asset belongs to
        ticker_to_cluster = {}
        for cluster in cluster_info:
            if cluster['id'] != 'META_CLUSTER':
                for ticker in cluster['tickers']:
                    ticker_to_cluster[ticker] = cluster['id']

        # Scale asset weights by meta-cluster allocation
        final_weights = {}
        for ticker, asset_weight in cluster_weights.items():
            cluster_id = ticker_to_cluster.get(ticker)
            if cluster_id and cluster_id in meta_weights:
                final_weights[ticker] = asset_weight * meta_weights[cluster_id]
            else:
                final_weights[ticker] = asset_weight

        # Normalize
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}
        return final_weights
    else:
        # No meta-cluster, just normalize asset weights
        total_weight = sum(cluster_weights.values())
        if total_weight > 0:
            return {k: v / total_weight for k, v in cluster_weights.items()}
        return cluster_weights


def compute_portfolio_metrics(
    weights: Dict[str, float],
    returns: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute portfolio performance metrics.

    Args:
        weights: Portfolio weights
        returns: Historical returns DataFrame

    Returns:
        Dict with performance metrics
    """
    if not weights:
        return {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'num_holdings': 0
        }

    # Filter to assets in portfolio
    tickers = list(weights.keys())
    tickers = [t for t in tickers if t in returns.columns]

    if not tickers:
        return {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'num_holdings': 0
        }

    # Create weight vector
    w = pd.Series({t: weights[t] for t in tickers})
    w = w / w.sum()  # Renormalize

    # Compute statistics
    mu = returns[tickers].mean() * 252  # Annualized
    Sigma = returns[tickers].cov() * 252

    portfolio_return = (w * mu).sum()
    portfolio_variance = (w.T @ Sigma @ w)
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0.0

    return {
        'expected_return': float(portfolio_return),
        'volatility': float(portfolio_volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'num_holdings': len([w for w in weights.values() if w > 1e-6])
    }


def display_weights(weights: Dict[str, float], top_n: int = 20):
    """Display portfolio weights in a formatted table."""
    if not weights:
        print("  No weights (empty portfolio)")
        return

    # Sort by weight descending
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    # Display top N
    print(f"\n  Top {min(top_n, len(sorted_weights))} Holdings:")
    print(f"  {'Ticker':<10} {'Weight':>10} {'%':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*8}")

    for ticker, weight in sorted_weights[:top_n]:
        if weight > 1e-6:  # Skip near-zero weights
            print(f"  {ticker:<10} {weight:>10.6f} {weight*100:>7.2f}%")

    if len(sorted_weights) > top_n:
        remaining = sorted_weights[top_n:]
        remaining_weight = sum(w for _, w in remaining)
        print(f"  {'...':<10} {remaining_weight:>10.6f} {remaining_weight*100:>7.2f}%")

    print(f"  {'-'*10} {'-'*10} {'-'*8}")
    total = sum(weights.values())
    print(f"  {'TOTAL':<10} {total:>10.6f} {total*100:>7.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Decode quantum annealing solver response and display portfolio weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
    # 1. Submit job and save response
    curl -X POST https://qpu-1.nodes.quip.network/solve \\
        -H "Content-Type: application/json" \\
        -d @backtest_jobs/train.json > response.json

    # 2. Decode response
    python tools/decode_qa_response.py \\
        --response response.json \\
        --metadata backtest_jobs/train.metadata.json \\
        --portfolio-csv portfolio.csv
        """
    )

    parser.add_argument('--response', type=str, required=True,
                       help='Solver response JSON file')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Job metadata JSON file')
    parser.add_argument('--portfolio-csv', type=str, required=True,
                       help='Portfolio price CSV file (for computing metrics)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top holdings to display (default: 20)')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Which sample to decode (default: 0 = best energy)')

    args = parser.parse_args()

    print("="*80)
    print("QUANTUM ANNEALING RESPONSE DECODER")
    print("="*80)

    # Load response
    print(f"\n[1/4] Loading solver response from: {args.response}")
    with open(args.response, 'r') as f:
        response = json.load(f)

    # Validate response format
    if 'samples' not in response or 'energies' not in response:
        print("  ✗ Invalid response format (missing 'samples' or 'energies')")
        sys.exit(1)

    n_samples = len(response['samples'])
    print(f"  ✓ Loaded {n_samples} samples")

    if 'transaction_id' in response:
        print(f"  Transaction ID: {response['transaction_id']}")
    if 'status' in response:
        print(f"  Status: {response['status']}")

    # Load metadata
    print(f"\n[2/4] Loading metadata from: {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)

    n_levels = metadata['parameters']['n_levels']
    variable_mapping = metadata['variable_mapping']
    reverse_mapping = metadata['reverse_mapping']

    print(f"  ✓ Problem configuration:")
    print(f"    Discrete levels: {n_levels}")
    print(f"    Variables: {len(variable_mapping)}")
    print(f"    Alpha: {metadata['parameters']['alpha']}")
    print(f"    Beta: {metadata['parameters']['beta']}")

    if 'train_start_date' in metadata:
        print(f"  ✓ Training period: {metadata['train_start_date']} to {metadata['train_end_date']}")
        print(f"    Test period: {metadata['test_start_date']} to {metadata['test_end_date']}")

    # Load portfolio data
    print(f"\n[3/4] Loading portfolio data from: {args.portfolio_csv}")
    prices, returns = load_portfolio_data(args.portfolio_csv, preprocess=False)
    print(f"  ✓ Loaded {len(returns.columns)} assets")

    # Extract training period returns if metadata specifies dates
    if 'train_start_date' in metadata:
        train_start = pd.Timestamp(metadata['train_start_date'])
        train_end = pd.Timestamp(metadata['train_end_date'])
        returns_train = returns.loc[train_start:train_end]
        print(f"  ✓ Using training period returns ({len(returns_train)} days)")
    else:
        returns_train = returns
        print(f"  ✓ Using full return series ({len(returns_train)} days)")

    # Decode samples
    print(f"\n[4/4] Decoding solutions...")

    # Find best solution (lowest energy)
    energies = response['energies']
    best_idx = int(np.argmin(energies))

    print(f"  Best solution: sample {best_idx} (energy: {energies[best_idx]:.6f})")

    # Decode requested sample (default: best)
    sample_idx = args.sample_idx if args.sample_idx < n_samples else best_idx
    sample = response['samples'][sample_idx]
    energy = energies[sample_idx]

    if sample_idx != best_idx:
        print(f"  Decoding sample {sample_idx} (energy: {energy:.6f})")

    cluster_info = metadata.get('cluster_info', [])

    weights = decode_sample(
        sample=sample,
        variable_mapping=variable_mapping,
        reverse_mapping=reverse_mapping,
        n_levels=n_levels,
        cluster_info=cluster_info
    )

    # Apply return adjustment (matches quantum_classical_comparison.py)
    # This is how the benchmark achieves concentration even with uniform QPU weights
    print(f"\n  Applying return adjustment (weight × return scoring)...")
    weights_raw = weights.copy()

    mu_train = returns_train.mean() * 252  # Annualized returns
    weight_pct = pd.Series(weights) * 100
    return_pct = mu_train[[t for t in weights.keys() if t in mu_train.index]] * 100
    scores = weight_pct * return_pct.abs()  # Use abs() to handle negative returns

    weights_adjusted = (scores / scores.sum()).to_dict()
    weights = weights_adjusted

    print(f"  ✓ Return adjustment applied (top assets amplified by return)")

    # Compute metrics
    metrics = compute_portfolio_metrics(weights, returns_train)

    # Display results
    print("\n" + "="*80)
    print("PORTFOLIO SOLUTION")
    print("="*80)

    print(f"\nPerformance Metrics:")
    print(f"  Expected Return:  {metrics['expected_return']*100:>7.2f}%")
    print(f"  Volatility:       {metrics['volatility']*100:>7.2f}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>7.3f}")
    print(f"  Num Holdings:     {metrics['num_holdings']:>7d}")

    display_weights(weights, top_n=args.top_n)

    # Energy statistics
    print(f"\nEnergy Statistics:")
    print(f"  Best energy:    {min(energies):>12.6f}")
    print(f"  Worst energy:   {max(energies):>12.6f}")
    print(f"  Mean energy:    {np.mean(energies):>12.6f}")
    print(f"  Std energy:     {np.std(energies):>12.6f}")

    # Count unique solutions
    unique_energies = len(set(energies))
    print(f"  Unique solutions: {unique_energies}/{n_samples}")

    print()


if __name__ == '__main__':
    main()
